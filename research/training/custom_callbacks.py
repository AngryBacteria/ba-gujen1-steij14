from transformers import TrainerCallback
import pynvml


class GPUMemoryUsageCallback(TrainerCallback):
    def __init__(self, gpu_id=0):
        super().__init__()
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        self.peak_memory = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.peak_memory = 0

    def on_step_end(self, args, state, control, **kwargs):
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        mem_used = meminfo.used
        self.peak_memory = max(self.peak_memory, mem_used)

    def on_train_end(self, args, state, control, **kwargs):
        print(f"Peak GPU memory usage: {self.peak_memory / 1024 / 1024:.2f} MB")
        pynvml.nvmlShutdown()
