import ctypes
from typing import List, Union

from warp.device import Devicelike, Device
from warp.runtime import Runtime
import warp_runtime_py as wp

# initialize global runtime
runtime = None


def init():
    """Initialize the Warp runtime. This function must be called before any other API call. If an error occurs an
    exception will be raised."""
    global runtime

    if runtime is None:
        runtime = Runtime()


def assert_initialized():
    assert runtime is not None, "Warp not initialized, call wp.init() before use"


# global entry points
def is_cpu_available():
    return runtime.llvm


def is_cuda_available():
    return get_cuda_device_count() > 0


def is_cuda_driver_initialized() -> bool:
    """Returns ``True`` if the CUDA driver is initialized.

    This is a stricter test than ``is_cuda_available()`` since a CUDA driver
    call to ``cuCtxGetCurrent`` is made, and the result is compared to
    `CUDA_SUCCESS`. Note that `CUDA_SUCCESS` is returned by ``cuCtxGetCurrent``
    even if there is no context bound to the calling CPU thread.

    This can be helpful in cases in which ``cuInit()`` was called before a fork.
    """
    assert_initialized()

    return wp.cuda_driver_is_initialized()


def get_devices() -> List[Device]:
    """Returns a list of devices supported in this environment."""

    assert_initialized()

    devices = []
    if is_cpu_available():
        devices.append(runtime.cpu_device)
    for cuda_device in runtime.cuda_devices:
        devices.append(cuda_device)
    return devices


def get_cuda_device_count() -> int:
    """Returns the number of CUDA devices supported in this environment."""

    assert_initialized()

    return len(runtime.cuda_devices)


def get_cuda_device(ordinal: Union[int, None] = None) -> Device:
    """Returns the CUDA device with the given ordinal or the current CUDA device if ordinal is None."""

    assert_initialized()

    if ordinal is None:
        return runtime.get_current_cuda_device()
    else:
        return runtime.cuda_devices[ordinal]


def get_cuda_devices() -> List[Device]:
    """Returns a list of CUDA devices supported in this environment."""

    assert_initialized()

    return runtime.cuda_devices


def get_preferred_device() -> Device:
    """Returns the preferred compute device, CUDA if available and CPU otherwise."""

    assert_initialized()

    if is_cuda_available():
        return runtime.cuda_devices[0]
    elif is_cpu_available():
        return runtime.cpu_device
    else:
        return None


def get_device(ident: Devicelike = None) -> Device:
    """Returns the device identified by the argument."""

    assert_initialized()

    return runtime.get_device(ident)


def set_device(ident: Devicelike):
    """Sets the target device identified by the argument."""

    assert_initialized()

    device = runtime.get_device(ident)
    runtime.set_default_device(device)
    device.make_current()


def map_cuda_device(alias: str, context: ctypes.c_void_p = None) -> Device:
    """Assign a device alias to a CUDA context.

    This function can be used to create a wp.Device for an external CUDA context.
    If a wp.Device already exists for the given context, it's alias will change to the given value.

    Args:
        alias: A unique string to identify the device.
        context: A CUDA context pointer (CUcontext).  If None, the currently bound CUDA context will be used.

    Returns:
        The associated wp.Device.
    """

    assert_initialized()

    return runtime.map_cuda_device(alias, context)


def unmap_cuda_device(alias: str):
    """Remove a CUDA device with the given alias."""

    assert_initialized()

    runtime.unmap_cuda_device(alias)
