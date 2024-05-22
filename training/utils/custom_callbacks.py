from transformers import TrainerCallback

from shared.gpu_utils import get_cuda_memory_usage


class GPUMemoryUsageCallback(TrainerCallback):
    def __init__(self, gpu_id=0, logging_steps=16):
        super().__init__()
        self.handle = gpu_id
        self.peak_memory = 0.0
        self.total_capacity = None
        self.logging_steps = logging_steps
        self.current_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.peak_memory = 0.0
        _, self.total_capacity = get_cuda_memory_usage(self.handle)

    def on_step_end(self, args, state, control, **kwargs):
        allocated, _ = get_cuda_memory_usage(self.handle)
        self.peak_memory = max(self.peak_memory, allocated)
        self.current_step += 1

        if self.current_step % self.logging_steps == 0:
            usage_percent = (allocated / self.total_capacity) * 100
            print(
                f"Current GPU memory usage (step {self.current_step}): {allocated:.2f} GB ({usage_percent:.2f}%)"
            )

    def on_train_end(self, args, state, control, **kwargs):
        usage_percent = (self.peak_memory / self.total_capacity) * 100
        print(
            f"Peak GPU memory usage: {self.peak_memory:.2f} GB ({usage_percent:.2f}%)"
        )
