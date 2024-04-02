import horovod.torch as hvd

def find_duplicates(lst):
    seen = set()
    dups = set()
    for el in lst:
        if el in seen:
            dups.add(el)
        seen.add(el)
    return dups

# return communication mode: allreduce or allgather
def get_comm(params):
    comm_name = params.get('comm_mode', 'allreduce')
    return comm_name

def get_compressor(params):
    compress_name = params.get('compressor', 'none')
    compress_ratio = params.get('compress_ratio', 0.01)
    if compress_name == 'none':
        from compression.compressor.none import NoneCompressor
        compressor = NoneCompressor()
    elif compress_name == 'topk':
        from compression.compressor.topk import TopKCompressor
        compressor = TopKCompressor(compress_ratio,rank=hvd.rank())
    else:
        raise NotImplementedError(compressor)

    return compressor

def get_memory(params):
    memory_name = params.get('memory', 'none')

    if memory_name == 'none':
        from compression.memory.none import NoneMemory
        memory = NoneMemory()

    elif memory_name == 'residual':
        from compression.memory.residual import ResidualMemory
        memory = ResidualMemory()  
    else:
        raise NotImplementedError(memory)
    return memory

def get_config(params):
    send_size_aresame = params.get('send_size_aresame', True)
    return send_size_aresame

def get_check(params):
    check = params.get('checkpoint', False)
    return check

# Special case:
# All dim==1 tensor should not be compressed
# ResNet: EF on the 'fc' will harm the performance of ADTOPK and AllchannelTopK
# VGG16: 'features.0' should not be compressed
# VGG16: EF on the 'classifier.6' will harm the performance
# LSTM: 'rnn.weight_hh' should not be compressed
def check_not_compress(params, name, tensor):
    
    if tensor.dim() == 1:
        return True
    if 'features.0' in name:
        return True
    if 'rnn.weight_hh' in name:
        return True

    return False


def check_not_ef(params, name, tensor):

    compressor_name = params.get('compressor', 'none')

    if 'adtopk' in compressor_name or 'alldimensiontopk' in compressor_name:
        if 'fc' in name:
            return True
    
    if 'classifier.6' in name:
        return True
    return False 