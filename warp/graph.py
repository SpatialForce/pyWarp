import ctypes

from warp.device import Device
import warp_runtime_py as wp


class Graph:
    def __init__(self, device: Device, exec: ctypes.c_void_p):
        self.device = device
        self.exec = exec

    def __del__(self):
        # use CUDA context guard to avoid side effects during garbage collection
        with self.device.context_guard:
            wp.cuda_graph_destroy(self.device.context, self.exec)
