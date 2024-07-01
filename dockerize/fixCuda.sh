#!/usr/bin/bash

sudo rmmod -f nvidia_uvm
sudo rmmod -f nvidia_drm
sudo rmmod -f nvidia_modeset
sudo rmmod -f nvidia
sudo modprobe nvidia
sudo modprobe nvidia_modeset
sudo modprobe nvidia_drm
sudo modprobe nvidia_uvm

# sudo rmmod -f nvidia_uvm;
# sudo rmmod -f nvidia;
# sudo modprobe nvidia;
# sudo modprobe nvidia_uvm;
# sudo nvidia-smi -r
# sudo nvidia-smi --gpu-reset


sudo nvidia-smi -pm ENABLED
sudo nvidia-smi -c EXCLUSIVE_PROCESS
# sudo rmmod nvidia_uvm
# sudo modprobe nvidia_uvm

nvidia-smi pmon -c 1