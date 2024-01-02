from warp.context import get_device
import warp_runtime_py as wp


class Event:
    # event creation flags
    class Flags:
        DEFAULT = 0x0
        BLOCKING_SYNC = 0x1
        DISABLE_TIMING = 0x2

    def __init__(self, device=None, cuda_event=None, enable_timing=False):
        self.owner = False

        device = get_device(device)
        if not device.is_cuda:
            raise RuntimeError(f"Device {device} is not a CUDA device")

        self.device = device

        if cuda_event is not None:
            self.cuda_event = cuda_event
        else:
            flags = Event.Flags.DEFAULT
            if not enable_timing:
                flags |= Event.Flags.DISABLE_TIMING
            self.cuda_event = wp.cuda_event_create(device.context, flags)
            if not self.cuda_event:
                raise RuntimeError(f"Failed to create event on device {device}")
            self.owner = True

    def __del__(self):
        if self.owner:
            wp.cuda_event_destroy(self.device.context, self.cuda_event)
