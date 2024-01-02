import platform
from typing import Union

from warp.allocator import Allocator
from warp.context_guard import ContextGuard
from warp.stream import Stream
from warp.utils.logging import warn
import warp_runtime_py as wp


class Device:
    def __init__(self, runtime, alias, ordinal=-1, is_primary=False, context=None):
        self.runtime = runtime
        self.alias = alias
        self.ordinal = ordinal
        self.is_primary = is_primary

        # context can be None to avoid acquiring primary contexts until the device is used
        self._context = context

        # if the device context is not primary, it cannot be None
        if ordinal != -1 and not is_primary:
            assert context is not None

        # streams will be created when context is acquired
        self._stream = None
        self.null_stream = None

        # indicates whether CUDA graph capture is active for this device
        self.is_capturing = False

        self.allocator = Allocator(self)
        self.context_guard = ContextGuard(self)

        if self.ordinal == -1:
            # CPU device
            self.name = platform.processor() or "CPU"
            self.arch = 0
            self.is_uva = False
            self.is_cubin_supported = False
            self.is_mempool_supported = False

            # TODO: add more device-specific dispatch functions
            self.memset = wp.memset_host
            self.memtile = wp.memtile_host

        elif ordinal >= 0 and ordinal < wp.cuda_device_get_count():
            # CUDA device
            self.name = wp.cuda_device_get_name(ordinal).decode()
            self.arch = wp.cuda_device_get_arch(ordinal)
            self.is_uva = wp.cuda_device_is_uva(ordinal)
            # check whether our NVRTC can generate CUBINs for this architecture
            self.is_cubin_supported = self.arch in runtime.nvrtc_supported_archs
            self.is_mempool_supported = wp.cuda_device_is_memory_pool_supported(ordinal)

            # Warn the user of a possible misconfiguration of their system
            if not self.is_mempool_supported:
                warn(
                    f"Support for stream ordered memory allocators was not detected on device {ordinal}. "
                    "This can prevent the use of graphs and/or result in poor performance. "
                    "Is the UVM driver enabled?"
                )

            # initialize streams unless context acquisition is postponed
            if self._context is not None:
                self.init_streams()

            # TODO: add more device-specific dispatch functions
            self.memset = lambda ptr, value, size: wp.memset_device(self.context, ptr, value, size)
            self.memtile = lambda ptr, src, srcsize, reps: wp.memtile_device(
                self.context, ptr, src, srcsize, reps
            )

        else:
            raise RuntimeError(f"Invalid device ordinal ({ordinal})'")

    def init_streams(self):
        # create a stream for asynchronous work
        self.stream = Stream(self)

        # CUDA default stream for some synchronous operations
        self.null_stream = Stream(self, cuda_stream=None)

    @property
    def is_cpu(self):
        return self.ordinal < 0

    @property
    def is_cuda(self):
        return self.ordinal >= 0

    @property
    def context(self):
        if self._context is not None:
            return self._context
        elif self.is_primary:
            # acquire primary context on demand
            self._context = wp.cuda_device_primary_context_retain(self.ordinal)
            if self._context is None:
                raise RuntimeError(f"Failed to acquire primary context for device {self}")
            self.runtime.context_map[self._context] = self
            # initialize streams
            self.init_streams()
        return self._context

    @property
    def has_context(self):
        return self._context is not None

    @property
    def stream(self):
        if self.context:
            return self._stream
        else:
            raise RuntimeError(f"Device {self} is not a CUDA device")

    @stream.setter
    def stream(self, s):
        if self.is_cuda:
            if s.device != self:
                raise RuntimeError(f"Stream from device {s.device} cannot be used on device {self}")
            self._stream = s
            wp.cuda_context_set_stream(self.context, s.cuda_stream)
        else:
            raise RuntimeError(f"Device {self} is not a CUDA device")

    @property
    def has_stream(self):
        return self._stream is not None

    def __str__(self):
        return self.alias

    def __repr__(self):
        return f"'{self.alias}'"

    def __eq__(self, other):
        if self is other:
            return True
        elif isinstance(other, Device):
            return self.context == other.context
        elif isinstance(other, str):
            if other == "cuda":
                return self == self.runtime.get_current_cuda_device()
            else:
                return other == self.alias
        else:
            return False

    def make_current(self):
        if self.context is not None:
            wp.cuda_context_set_current(self.context)

    def can_access(self, other):
        other = self.runtime.get_device(other)
        if self.context == other.context:
            return True
        elif self.context is not None and other.context is not None:
            return bool(wp.cuda_context_can_access_peer(self.context, other.context))
        else:
            return False


""" Meta-type for arguments that can be resolved to a concrete Device.
"""
Devicelike = Union[Device, str, None]
