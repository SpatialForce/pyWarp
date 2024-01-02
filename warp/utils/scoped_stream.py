from warp.utils.scoped_device import ScopedDevice


class ScopedStream:
    def __init__(self, stream):
        self.stream = stream
        if stream is not None:
            self.device = stream.device
            self.device_scope = ScopedDevice(self.device)

    def __enter__(self):
        if self.stream is not None:
            self.device_scope.__enter__()
            self.saved_stream = self.device.stream
            self.device.stream = self.stream

        return self.stream

    def __exit__(self, exc_type, exc_value, traceback):
        if self.stream is not None:
            self.device.stream = self.saved_stream
            self.device_scope.__exit__(exc_type, exc_value, traceback)
