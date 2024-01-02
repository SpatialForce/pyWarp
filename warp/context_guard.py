import warp_runtime_py as wp

from warp.context import is_cuda_driver_initialized


class ContextGuard:
    def __init__(self, device):
        self.device = device

    def __enter__(self):
        if self.device.is_cuda:
            wp.cuda_context_push_current(self.device.context)
        elif is_cuda_driver_initialized():
            self.saved_context = wp.cuda_context_get_current()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.device.is_cuda:
            wp.cuda_context_pop_current()
        elif is_cuda_driver_initialized():
            wp.cuda_context_set_current(self.saved_context)