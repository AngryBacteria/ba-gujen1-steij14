import os
import setproctitle
import torch

# Variables
GPU_ID = 0
MODEl_ID = "mistralai/Mistral-7B-v0.1"

# Setup
setproctitle.setproctitle("gujen1 - ba-mistralai - testing.py")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"


print(f"Is CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
