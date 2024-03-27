import torch
from compression import Compressor

class TopKCompressor(Compressor):

    def __init__(self, compress_ratio, rank):
        super().__init__()

        self.compress_ratio = compress_ratio
        self.rank = rank

    def compress(self, tensor, name):
        ctx = tensor.numel(), tensor.size()
        tensor = tensor.flatten().cuda()
        len = tensor.numel()
        k = max(1, int(len * self.compress_ratio))
        _, indices = torch.topk(tensor.abs(), k, sorted=False,)
        values = torch.gather(tensor, 0, indices)
        tensors = values, indices
        return tensors, ctx

    def decompress(self, tensors, ctx, name):
        numel, shape = ctx
        values, indices = tensors
        if values.numel()==numel:
            return values
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
        tensor_decompressed = tensor_decompressed.scatter_add(0, indices, values)
        return tensor_decompressed.view(shape)

