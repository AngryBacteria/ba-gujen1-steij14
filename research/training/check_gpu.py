import os
import setproctitle
import torch

# Variables
GPU_ID = 0
GPU_ID_STRING = "cuda:0"
MODEl_ID = "mistralai/Mistral-7B-v0.1"

# Setup
setproctitle.setproctitle("gujen1 - ba-mistralai - testing.py")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"


print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())