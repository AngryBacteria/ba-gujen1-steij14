from transformers import TrainerCallback

from research.training.utils.utils_gpu import get_gpu_memory_usage


class GPUMemoryUsageCallback(TrainerCallback):
    def __init__(self, gpu_id=0, every_step=False):
        super().__init__()
        self.handle = gpu_id
        self.peak_memory = 0.0
        self.total_capacity = None
        self.every_step = every_step

    def on_train_begin(self, args, state, control, **kwargs):
        self.peak_memory = 0.0
        _, self.total_capacity = get_gpu_memory_usage(self.handle)

    def on_step_end(self, args, state, control, **kwargs):
        allocated, _ = get_gpu_memory_usage(self.handle)
        self.peak_memory = max(self.peak_memory, allocated)
        if self.every_step:
            usage_percent = (allocated / self.total_capacity) * 100
            print(
                f"Current GPU memory usage: {allocated:.2f} MB ({usage_percent:.2f}%)"
            )

    def on_train_end(self, args, state, control, **kwargs):
        usage_percent = (self.peak_memory / self.total_capacity) * 100
        print(
            f"Peak GPU memory usage: {self.peak_memory:.2f} MB ({usage_percent:.2f}%)"
        )
