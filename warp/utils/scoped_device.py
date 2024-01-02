from warp.context import get_device


class ScopedDevice:
    def __init__(self, device):
        self.device = get_device(device)

    def __enter__(self):
        # save the previous default device
        self.saved_device = self.device.runtime.default_device

        # make this the default device
        self.device.runtime.default_device = self.device

        # make it the current CUDA device so that device alias "cuda" will evaluate to this device
        self.device.context_guard.__enter__()

        return self.device

    def __exit__(self, exc_type, exc_value, traceback):
        # restore original CUDA context
        self.device.context_guard.__exit__(exc_type, exc_value, traceback)

        # restore original target device
        self.device.runtime.default_device = self.saved_device
