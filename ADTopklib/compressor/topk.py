import torch

from ADTopklib import Compressor
import random

# tensor稀疏化得到top-k的稀疏值
def sparsify(tensor, compress_ratio):
    # 将tensor展平成一维向量
    tensor = tensor.flatten().cuda()
    # tensor.numel()获取tensor中元素的个数
    len = tensor.numel()
    compress_ratio=0.001
    compress_ratio=0.01
    compress_ratio=0.05
    
    k = max(1, int(len * compress_ratio))
    # 获取top-k值的索引
    _, indices = torch.topk(tensor.abs(), k, sorted=False,)
    values = torch.gather(tensor, 0, indices)
    return values, indices

# tensor反稀疏化
def desparsify(tensors, numel):
    # tensors保存了稀疏值和系数值的索引
    values, indices = tensors
    
    # Awareness
    if values.numel()==numel:
        return values
    
    # 返回一个形状为为size,类型为torch.dtype,里面的每一个值都是0的tensor
    tensor_decompressed = torch.zeros(
        numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
    # 填充稀疏值
    tensor_decompressed.scatter_(0, indices, values)
    return tensor_decompressed


class TopKCompressor(Compressor):

    def __init__(self, compress_ratio, rank):
        super().__init__()

        self.compress_ratio = compress_ratio
        self.rank = rank

    # 抽象方法重载compress
    def compress(self, tensor, name):
        # if self.rank==0:
        #     print('--------------------')

        # if self.rank==0:
        #     print('tensor.shape=')
        #     print(tensor.shape)
        #     print('tensor=')
        #     print(tensor)
        # self.show_sparsify(tensor)

        tensors = sparsify(tensor, self.compress_ratio)
        # ctx 保存tensor的元素个数和shap,用于解压还原原始tensor的形状
        
        
        # if self.rank==0:
            
            # print('values.shape=')
            # print(values.shape)
            # print('values=')
            # print(values)

        ctx = tensor.numel(), tensor.size()
        return tensors, ctx

    def show_sparsify(self, tensor):
        # if self.rank==0:
        print('----------'+str(self.rank)+'----------')
        print(tensor.shape)
        tensor = tensor.flatten()
        print(tensor)

    def decompress(self, tensors, ctx, name):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        if ctx==None:
            tensor, = tensors
            return tensor
        numel, shape = ctx
        
        tensor_decompressed = desparsify(tensors, numel)
        # if self.rank==0:
            

        return tensor_decompressed.view(shape)

    def decompress_add(self, tensors, ctx, name):
        numel, shape = ctx
        values, indices = tensors
        if values.numel()==numel:
            return values
        # 返回一个形状为为size,类型为torch.dtype,里面的每一个值都是0的tensor
        tensor_decompressed = torch.zeros(
            numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
        # 填充稀疏值
        # if hvd.rank() == 0:
        #     print('values: ', values, 'indices: ', indices)
        # [a,b,    c,d]  [0,1,    0,2]
        # [c, b ,d ][a+c, b,d ]
        tensor_decompressed = tensor_decompressed.scatter_add(0, indices, values)
        return tensor_decompressed.view(shape)

