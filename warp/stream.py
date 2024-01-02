from warp.context import runtime
from warp.device import Device
from warp.event import Event
import warp_runtime_py as wp


class Stream:
    def __init__(self, device=None, **kwargs):
        self.owner = False

        # we can't use get_device() if called during init, but we can use an explicit Device arg
        if runtime is not None:
            device = runtime.get_device(device)
        elif not isinstance(device, Device):
            raise RuntimeError(
                "A device object is required when creating a stream before or during Warp initialization"
            )

        if not device.is_cuda:
            raise RuntimeError(f"Device {device} is not a CUDA device")

        # we pass cuda_stream through kwargs because cuda_stream=None is actually a valid value (CUDA default stream)
        if "cuda_stream" in kwargs:
            self.cuda_stream = kwargs["cuda_stream"]
        else:
            self.cuda_stream = wp.cuda_stream_create(device.context)
            if not self.cuda_stream:
                raise RuntimeError(f"Failed to create stream on device {device}")
            self.owner = True

        self.device = device

    def __del__(self):
        if self.owner:
            wp.cuda_stream_destroy(self.device.context, self.cuda_stream)

    def record_event(self, event: Event = None):
        if event is None:
            event = Event(self.device)
        elif event.device != self.device:
            raise RuntimeError(
                f"Event from device {event.device} cannot be recorded on stream from device {self.device}"
            )

        wp.cuda_event_record(self.device.context, event.cuda_event, self.cuda_stream)

        return event

    def wait_event(self, event):
        wp.cuda_stream_wait_event(self.device.context, self.cuda_stream, event.cuda_event)

    def wait_stream(self, other_stream, event=None):
        if event is None:
            event = Event(other_stream.device)

        wp.cuda_stream_wait_stream(
            self.device.context, self.cuda_stream, other_stream.cuda_stream, event.cuda_event
        )
