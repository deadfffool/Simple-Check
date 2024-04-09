import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from compression.compressor.topk import TopKCompressor
import compression


model = models.resnet50().cuda()
comm_params = {
'comm_mode':'allgather',
'compressor':'topk',
'compress_ratio' : 0.01,
'memory':'residual',
'send_size_aresame':True,
'model_named_parameters': model.named_parameters(),
'checkpoint': True
}
optimizer = torch.optim.SGD(model.parameters(), lr=0.0125*256, momentum=0.9, weight_decay=1e-4)
# optimizer = compression.DistributedOptimizer(optimizer, comm_params=comm_params, named_parameters=model.named_parameters())

checkpoint = torch.load('./diff/checkpoint_1.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])

# tensor_compressed = torch.load('./diff/checkpoint_0-0.pth.tar')
# topk = TopKCompressor(0.01,0)

# for key in tensor_compressed.keys():
#     tensor = topk.decompress(tensor_compressed[key]['tensors'],tensor_compressed[key]['ctx'],None)
#     for param_group in optimizer.param_groups:
#         for p in param_group['params']:
#             name = _parameter_names.get(p)
#             if(name == key):
#                 p.grad = tensor
#                 print(key)
#                 break
# optimizer.step() 