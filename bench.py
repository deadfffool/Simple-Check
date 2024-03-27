import time
import argparse
import os
import shutil
import math
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import horovod.torch as hvd
from collections import OrderedDict
import util.global_value as glo

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

# compression
import compression

glo.init()

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', metavar='DIR', default="cifar100", type=str,
                    help='path to dataset')
parser.add_argument('--model-net', default='resnet50',type=str, help='net type')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--warmup-epochs', type=float, default=1,
                    help='number of warmup epochs')
parser.add_argument('--lr', '--learning-rate', default=0.0125, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--noeval', action='store_true', help = 'not run evaluation phase')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='seed for initializing training')
parser.add_argument('--download-data', action='store_true', default=False,
                    help='download data for training')
parser.add_argument("--classes", default=1000, type=int)
parser.add_argument("--cache_size", default=0, type=int)

best_acc1 = 0
args = parser.parse_args()

def main():
    global best_acc1, args
   
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    args.allreduce_batch_size = args.batch_size * args.batches_per_allreduce

    hvd.init()
    torch.manual_seed(args.seed)
    
    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    start_full = time.time()
    
    time_stat = OrderedDict()
    start = time.time()
    
    # Horovod: limit # of CPU threads to be used per worker.
    # torch.set_num_threads(8)
    
    # Set up standard model.
    if args.model_net == 'resnet50':
        model = models.resnet50()
    elif args.model_net == 'resnet101':
        model = models.resnet101()
    elif args.model_net == 'vgg16':
        model = models.vgg16_bn()
    elif args.model_net == 'vgg19':
        model = models.vgg19_bn()
    else:
        print("Model net ERROR!")
        sys.exit()
    if args.cuda:
        # Move model to GPU.
        model.cuda()
    
    # By default, Adasum doesn't need scaling up learning rate.
    # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
    lr_scaler = args.batch_size * hvd.size()
    
    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.CrossEntropyLoss()
    
    cudnn.benchmark = True

    kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    # imagenet
    # Remember to modify args.train_dir and args.val_dir!!!
    if args.dataset == 'imagenet':
        train_dataset = \
            datasets.ImageFolder('/data/dataset/cv/imagenet_0908/train',
                                transform=transforms.Compose([
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
                                ]))
    # CIFAR100
    elif args.dataset == 'cifar100':
        train_dataset = \
            datasets.CIFAR100('~/cifar100/train',
                                train=True,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(15),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                                        std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
                                ]))
    
    # Horovod: use DistributedSampler to partition data among workers. Manually specify
    # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, **kwargs)
    
    # Imagenet
    if args.dataset == 'imagenet':
        val_dataset = \
            datasets.ImageFolder('/data/dataset/cv/imagenet_0908/val',
                                transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
                                ])) 
    # CIFAR100
    elif args.dataset == 'cifar100':
        val_dataset = \
            datasets.CIFAR100('~/cifar100/validation',
                                train=False,
                                download=True,
                                transform=transforms.Compose([
                                    #  transforms.Resize(256),
                                    #  transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                                        std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
                                ]))    
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             sampler=val_sampler, **kwargs)

    optimizer = torch.optim.SGD(model.parameters(), lr=(args.lr * lr_scaler),
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    
    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    
    # Compression
    comm_params = {
    'comm_mode':'allgather_fast',
    'compressor':'topk',
    'memory':'residual',
    'send_size_aresame':True,
    'model_named_parameters': model.named_parameters()
    }
    
    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = compression.DistributedOptimizer(optimizer, comm_params=comm_params, named_parameters=model.named_parameters())
    
    # optionally resume from a checkpoint at rank 0, then broadcast weights to other workers
    if args.resume and hvd.rank() == 0:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # Horovod: broadcast start_epoch from rank 0 to other ranks
            args.start_epoch = hvd.broadcast(torch.tensor(args.start_epoch), root_rank=0,
                                             name='start_epoch').item()
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    dur_setup = time.time() - start
    time_stat["setup_time"] = dur_setup
    train_ep = AverageMeter('Train Time', ':6.3f')
    
    for epoch in range(args.start_epoch, args.epochs):
        print("epoch {}".format(epoch))
        start_ep = time.time()
        train_sampler.set_epoch(epoch)
        
        glo.set_value('epoch', epoch)

        # train for one epoch
        avg_train_time = train(train_loader, model, criterion, optimizer, epoch, args)
        train_ep.update(avg_train_time)
        
        # evaluate on validation set
        if args.noeval:
            acc1 = 0
        else:
            acc1 = validate(val_loader, model, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if hvd.rank() == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'net': args.model_net,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
            if epoch == args.epochs - 1:
                print('##Top-1 {0}\n'
                    '##Perf  {1}'.format(acc1, hvd.size() * args.allreduce_batch_size / train_ep.avg))
        
        dur_ep = time.time() - start_ep
        print("epoch {} takes {}s".format(epoch, dur_ep))
        time_stat["epoch" + str(epoch)] = dur_ep
    
    dur_full = time.time() - start_full
    if hvd.rank() == 0:
        with open("time.txt", 'w') as f:
            for k, t in time_stat.items():
                print("Time stat {} : {}s".format(k, t))
                f.write(str(t))
                f.write("\n")
        print("Total time for all {} epochs = {}s".format(args.epochs-args.start_epoch, dur_full))


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    train_loader_len = int(len(train_loader))
    progress = ProgressMeter(
        train_loader_len,
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    
    def trace_handler(prof):
        print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
        prof.export_chrome_trace("pytorch-trace/pytorch-trace" + str(prof.step_num) + ".json")
    
    if True:
    # event1 = torch.cuda.Event()
        all_iterations = train_loader_len
        # for i, (images, target) in enumerate(train_loader):
        for i, data in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            
            adjust_learning_rate(optimizer, train_loader, epoch, i, args, train_loader_len)
            
            input_var, target_var = data
            target_var = target_var.squeeze().cuda().long()
            images = Variable(input_var).cuda(non_blocking=True)
            target = Variable(target_var).cuda(non_blocking=True)
            
            glo.set_value('iter_this_epoch', i)
            data_index = i * args.allreduce_batch_size
            glo.set_value('data_index', data_index)
            
            # compute output
            k = 0
            for j in range(0, len(images), args.batch_size):
                # s2 = time.time()
                optimizer.zero_grad()
                images_batch = images[j:j + args.batch_size]
                target_batch = target[j:j + args.batch_size]
                output = model(images_batch)
                # s3 = time.time()
                # print('forward time {}, {}'.format(s3 - s2, s3 - end))
                # s2 = time.time()
                # loss = F.cross_entropy(output, target_batch)
                loss = criterion(output, target_batch)
                acc1, acc5 = accuracy(output, target_batch, topk=(1, 5))
                # loss_val = loss.detach().to(device='cpu', non_blocking=True)
                loss_val = loss.data
                # loss_val = loss.item()
                # torch.cuda.current_stream().synchronize()
                # event1.record()
                # event1.synchronize()
                # s3 = time.time()
                # print('get loss in cpu {}, {}'.format(s3 - s2, s3 - end))
                # s2 = time.time()
                # losses.update(loss.item(), images_batch.size(0))
                losses.update(loss_val, images_batch.size(0))
                # s3 = time.time()
                # print('upate losses {}, {}'.format(s3 - s2, s3 - end))
                # s2 = time.time()
                top1.update(acc1[0], images_batch.size(0))
                top5.update(acc5[0], images_batch.size(0))
                # Average gradients among sub-batches
                loss.div_(math.ceil(float(len(images)) / args.batch_size))
                loss.backward()
                # s3 = time.time()
                # print('backward time {}, {}'.format(s3 - s2, s3 - end))
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                # if i % args.print_freq == 0:
                print('iter {} of {}, batch time: {}'.format(i * args.batches_per_allreduce + k, all_iterations, time.time() - end))
                end = time.time()
                k = k + 1
                
            # if i % args.print_freq == 0:
                # progress.display(i * args.batches_per_allreduce + k)
            
                # p.step()
    progress.display(all_iterations)
    return batch_time.avg


def validate(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.cuda:
                images, target = images.cuda(), target.cuda()
            # compute output
            output = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, train_loader, epoch, batch_idx, args, train_loader_len):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / train_loader_len
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * hvd.size() * args.batches_per_allreduce * lr_adj

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
