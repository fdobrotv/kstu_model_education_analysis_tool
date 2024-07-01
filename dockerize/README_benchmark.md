cd benchmark
./initialize.sh
cd ..

sudo docker build -t my-long-benchmark-latest . -f ./Dockerfile.tensorflow.benchmark.latest

sudo docker run -v ./benchmark:/benchmark --gpus all -it my-long-benchmark-latest bash

cd /benchmark/dawn-bench-models/tensorflow/CIFAR10

To change GPU power limit:
    sudo nvidia-smi --power-limit 115

To get NUMA affinity
    nvidia-smi topo -m
    for i in $(nvidia-smi -x -q | grep "gpu id" | awk -F "\"" '{print $2}' | awk -F ":" '{print "0000:"  $2 ":"  $3}' | tr '[:upper:]' '[:lower:]'); do echo $i; cat "/sys/bus/pci/devices/$i/numa_node"; done

Resolve "RuntimeError: cuda runtime error (999)":
    apt-get install nvidia-modprobe -y
    nvidia-modprobe -u

    sudo rmmod -f nvidia_uvm;
    sudo rmmod -f nvidia;
    sudo modprobe nvidia;
    sudo modprobe nvidia_uvm;

    CUDA_VISIBLE_DEVICES=1 python -c 'import torch; print(torch.rand(2,3).cuda())'
    python -c 'import torch; torch.cuda.is_available()'