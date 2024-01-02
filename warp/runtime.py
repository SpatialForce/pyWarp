import ctypes
import platform

from warp import config, build
from warp.context import runtime
from warp.device import Device, Devicelike
import warp_runtime_py as wp


class Runtime:
    def __init__(self):
        error = wp.init()

        if error != 0:
            raise Exception("Warp initialization failed")

        self.device_map = {}  # device lookup by alias
        self.context_map = {}  # device lookup by context

        # register CPU device
        cpu_name = platform.processor()
        if not cpu_name:
            cpu_name = "CPU"
        self.cpu_device = Device(self, "cpu")
        self.device_map["cpu"] = self.cpu_device
        self.context_map[None] = self.cpu_device

        cuda_device_count = wp.cuda_device_get_count()

        if cuda_device_count > 0:
            # get CUDA Toolkit and driver versions
            self.toolkit_version = wp.cuda_toolkit_version()
            self.driver_version = wp.cuda_driver_version()

            # get all architectures supported by NVRTC
            num_archs = wp.nvrtc_supported_arch_count()
            if num_archs > 0:
                archs = (ctypes.c_int * num_archs)()
                wp.nvrtc_supported_archs(archs)
                self.nvrtc_supported_archs = list(archs)
            else:
                self.nvrtc_supported_archs = []

        # register CUDA devices
        self.cuda_devices = []
        self.cuda_primary_devices = []
        for i in range(cuda_device_count):
            alias = f"cuda:{i}"
            device = Device(self, alias, ordinal=i, is_primary=True)
            self.cuda_devices.append(device)
            self.cuda_primary_devices.append(device)
            self.device_map[alias] = device

        # set default device
        if cuda_device_count > 0:
            if wp.cuda_context_get_current() is not None:
                self.set_default_device("cuda")
            else:
                self.set_default_device("cuda:0")
        else:
            # CUDA not available
            self.set_default_device("cpu")

        # initialize kernel cache
        build.init_kernel_cache(config.kernel_cache_dir)

        # print device and version information
        if not config.quiet:
            print(f"Warp {config.version} initialized:")
            if cuda_device_count > 0:
                toolkit_version = (self.toolkit_version // 1000, (self.toolkit_version % 1000) // 10)
                driver_version = (self.driver_version // 1000, (self.driver_version % 1000) // 10)
                print(
                    f"   CUDA Toolkit: {toolkit_version[0]}.{toolkit_version[1]}, Driver: {driver_version[0]}.{driver_version[1]}"
                )
            else:
                if wp.is_cuda_enabled():
                    # Warp was compiled with CUDA support, but no devices are available
                    print("   CUDA devices not available")
                else:
                    # Warp was compiled without CUDA support
                    print("   CUDA support not enabled in this build")
            print("   Devices:")
            print(f'     "{self.cpu_device.alias}"    | {self.cpu_device.name}')
            for cuda_device in self.cuda_devices:
                print(f'     "{cuda_device.alias}" | {cuda_device.name} (sm_{cuda_device.arch})')
            print(f"   Kernel cache: {config.kernel_cache_dir}")

        # CUDA compatibility check
        if cuda_device_count > 0 and not wp.is_cuda_compatibility_enabled():
            if self.driver_version < self.toolkit_version:
                print("******************************************************************")
                print("* WARNING:                                                       *")
                print("*   Warp was compiled without CUDA compatibility support         *")
                print("*   (quick build).  The CUDA Toolkit version used to build       *")
                print("*   Warp is not fully supported by the current driver.           *")
                print("*   Some CUDA functionality may not work correctly!              *")
                print("*   Update the driver or rebuild Warp without the --quick flag.  *")
                print("******************************************************************")

        # global tape
        self.tape = None

    def get_device(self, ident: Devicelike = None) -> Device:
        if isinstance(ident, Device):
            return ident
        elif ident is None:
            return self.default_device
        elif isinstance(ident, str):
            if ident == "cuda":
                return self.get_current_cuda_device()
            else:
                return self.device_map[ident]
        else:
            raise RuntimeError(f"Unable to resolve device from argument of type {type(ident)}")

    def set_default_device(self, ident: Devicelike):
        self.default_device = self.get_device(ident)

    def get_current_cuda_device(self):
        current_context = wp.cuda_context_get_current()
        if current_context is not None:
            current_device = self.context_map.get(current_context)
            if current_device is not None:
                # this is a known device
                return current_device
            elif wp.cuda_context_is_primary(current_context):
                # this is a primary context that we haven't used yet
                ordinal = wp.cuda_context_get_device_ordinal(current_context)
                device = self.cuda_devices[ordinal]
                self.context_map[current_context] = device
                return device
            else:
                # this is an unseen non-primary context, register it as a new device with a unique alias
                alias = f"cuda!{current_context:x}"
                return self.map_cuda_device(alias, current_context)
        elif self.default_device.is_cuda:
            return self.default_device
        elif self.cuda_devices:
            return self.cuda_devices[0]
        else:
            raise RuntimeError("CUDA is not available")

    def rename_device(self, device, alias):
        del self.device_map[device.alias]
        device.alias = alias
        self.device_map[alias] = device
        return device

    def map_cuda_device(self, alias, context=None) -> Device:
        if context is None:
            context = wp.cuda_context_get_current()
            if context is None:
                raise RuntimeError(f"Unable to determine CUDA context for device alias '{alias}'")

        # check if this alias already exists
        if alias in self.device_map:
            device = self.device_map[alias]
            if context == device.context:
                # device already exists with the same alias, that's fine
                return device
            else:
                raise RuntimeError(f"Device alias '{alias}' already exists")

        # check if this context already has an associated Warp device
        if context in self.context_map:
            # rename the device
            device = self.context_map[context]
            return self.rename_device(device, alias)
        else:
            # it's an unmapped context

            # get the device ordinal
            ordinal = wp.cuda_context_get_device_ordinal(context)

            # check if this is a primary context (we could get here if it's a device that hasn't been used yet)
            if wp.cuda_context_is_primary(context):
                # rename the device
                device = self.cuda_primary_devices[ordinal]
                return self.rename_device(device, alias)
            else:
                # create a new Warp device for this context
                device = Device(self, alias, ordinal=ordinal, is_primary=False, context=context)

                self.device_map[alias] = device
                self.context_map[context] = device
                self.cuda_devices.append(device)

                return device

    def unmap_cuda_device(self, alias):
        device = self.device_map.get(alias)

        # make sure the alias refers to a CUDA device
        if device is None or not device.is_cuda:
            raise RuntimeError(f"Invalid CUDA device alias '{alias}'")

        del self.device_map[alias]
        del self.context_map[device.context]
        self.cuda_devices.remove(device)

    def verify_cuda_device(self, device: Devicelike = None):
        if config.verify_cuda:
            device = runtime.get_device(device)
            if not device.is_cuda:
                return

            err = wp.cuda_context_check(device.context)
            if err != 0:
                raise RuntimeError(f"CUDA error detected: {err}")
