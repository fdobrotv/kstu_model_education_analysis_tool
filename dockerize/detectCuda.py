#!/usr/bin/python3

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch.nn as nn
import torch.nn.functional as F
import torch
print(torch.__version__)

print(torch.cuda.is_available())

print(torch._C._cuda_getDeviceCount())