HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 horovodrun -np 2 -H n20:1,n19:1 python bench.py --noeval -b 256 --epochs 1
rm checkpoint.pth.tar