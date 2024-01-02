import warp_runtime_py as wp


# a simple allocator
# TODO: use a pooled allocator to avoid hitting the system allocator
class Allocator:
    def __init__(self, device):
        self.device = device

    def alloc(self, size_in_bytes, pinned=False):
        if self.device.is_cuda:
            if self.device.is_capturing:
                raise RuntimeError(f"Cannot allocate memory on device {self} while graph capture is active")
            return wp.alloc_device(self.device.context, size_in_bytes)
        elif self.device.is_cpu:
            if pinned:
                return wp.alloc_pinned(size_in_bytes)
            else:
                return wp.alloc_host(size_in_bytes)

    def free(self, ptr, size_in_bytes, pinned=False):
        if self.device.is_cuda:
            if self.device.is_capturing:
                raise RuntimeError(f"Cannot free memory on device {self} while graph capture is active")
            return wp.free_device(self.device.context, ptr)
        elif self.device.is_cpu:
            if pinned:
                return wp.free_pinned(ptr)
            else:
                return wp.free_host(ptr)
