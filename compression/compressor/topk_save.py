import torch
from compression import Compressor

def sparsify(tensor, compress_ratio):
    tensor = tensor.flatten().cuda()
    len = tensor.numel()
    # compress_ratio=0.001
    # compress_ratio=0.05
    compress_ratio=0.01
    
    k = max(1, int(len * compress_ratio))
    _, indices = torch.topk(tensor.abs(), k, sorted=False,)
    values = torch.gather(tensor, 0, indices)
    return values, indices

def desparsify(tensors, numel):
    values, indices = tensors
    if values.numel()==numel:
        return values

    tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
    tensor_decompressed.scatter_(0, indices, values)
    return tensor_decompressed


class TopKCompressor(Compressor):

    def __init__(self, compress_ratio, rank):
        super().__init__()

        self.compress_ratio = compress_ratio
        self.rank = rank

    def compress(self, tensor, name):
        tensors = sparsify(tensor, self.compress_ratio)
        ctx = tensor.numel(), tensor.size()
        return tensors, ctx

    def decompress(self, tensors, ctx, name):
        if ctx==None:
            tensor, = tensors
            return tensor
        numel, shape = ctx
        tensor_decompressed = desparsify(tensors, numel)
        return tensor_decompressed.view(shape)

    def decompress_add(self, tensors, ctx, name):
        numel, shape = ctx
        values, indices = tensors
        if values.numel()==numel:
            return values
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
        tensor_decompressed = tensor_decompressed.scatter_add(0, indices, values)
        return tensor_decompressed.view(shape)

