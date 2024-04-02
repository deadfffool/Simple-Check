# HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 horovodrun -np 2 -H n20:1,n19:1 python baseline.py --noeval -b 256 --epochs 4
# HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 horovodrun -np 2 -H n20:1,n19:1 python differential_checkpoint.py --noeval -b 256 --epochs 4
# HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 horovodrun -np 2 -H n20:1,n19:1 python baseline2.py --noeval -b 256 --epochs 4
HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 horovodrun -np 2 -H n20:1,n19:1 python test.py --noeval -b 256 --epochs 2