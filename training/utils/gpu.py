import os

import pynvml
import setproctitle
import torch
from pynvml import NVMLError

from training.utils.printing import print_with_heading


def print_gpu_support(gpu_id: str):
    """Print information about the GPU."""
    setproctitle.setproctitle("gujen1 - ba-mistralai - testing.py")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    print_with_heading("GPU Information")
    print(f"Is CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    current_device = torch.cuda.current_device()
    print(f"Current device ID: {current_device}")
    print(f"Device name: {torch.cuda.get_device_name(current_device)}")

    memory_allocated = torch.cuda.memory_allocated(current_device)
    memory_reserved = torch.cuda.memory_reserved(current_device)
    total_memory = torch.cuda.get_device_properties(current_device).total_memory
    print(f"Memory allocated: {memory_allocated / 1e6} MB")
    print(f"Memory reserved: {memory_reserved / 1e6} MB")
    print(f"Total memory: {total_memory / 1e6} MB")
    print_with_heading()


def get_gpu_memory_usage(gpu_id: int) -> tuple[float, float]:
    """Get the memory usage and capacity in GB for the specified GPU."""
    try:
        # NVML
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        allocated = info.used / 1024.0**3
        capacity = info.total / 1024.0**3

        return allocated, capacity
    except NVMLError:
        return 0.0, 0.0
    finally:
        pynvml.nvmlShutdown()
