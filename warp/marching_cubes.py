import ctypes
import warp_runtime_py as wp

from warp.context import runtime
from warp.dsl.types import vec3, array


class MarchingCubes:
    def __init__(self, nx: int, ny: int, nz: int, max_verts: int, max_tris: int, device=None):
        self.device = runtime.get_device(device)

        if not self.device.is_cuda:
            raise RuntimeError("Only CUDA devices are supported for marching cubes")

        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.max_verts = max_verts
        self.max_tris = max_tris

        # bindings to warp.so
        self.alloc = wp.marching_cubes_create_device
        self.alloc.argtypes = [ctypes.c_void_p]
        self.alloc.restype = ctypes.c_uint64
        self.free = wp.marching_cubes_destroy_device

        from warp.context import zeros

        self.verts = zeros(max_verts, dtype=vec3, device=self.device)
        self.indices = zeros(max_tris * 3, dtype=int, device=self.device)

        # alloc surfacer
        self.id = ctypes.c_uint64(self.alloc(self.device.context))

    def __del__(self):
        # use CUDA context guard to avoid side effects during garbage collection
        with self.device.context_guard:
            # destroy surfacer
            self.free(self.id)

    def resize(self, nx: int, ny: int, nz: int, max_verts: int, max_tris: int):
        # actual allocations will be resized on next call to surface()
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.max_verts = max_verts
        self.max_tris = max_tris

    def surface(self, field: array(dtype=float), threshold: float):
        # WP_API int marching_cubes_surface_host(const float* field, int nx, int ny, int nz, float threshold, wp::vec3* verts, int* triangles, int max_verts, int max_tris, int* out_num_verts, int* out_num_tris);
        num_verts = ctypes.c_int(0)
        num_tris = ctypes.c_int(0)

        error = wp.marching_cubes_surface_device(
            self.id,
            ctypes.cast(field.ptr, ctypes.c_void_p),
            self.nx,
            self.ny,
            self.nz,
            ctypes.c_float(threshold),
            ctypes.cast(self.verts.ptr, ctypes.c_void_p),
            ctypes.cast(self.indices.ptr, ctypes.c_void_p),
            self.max_verts,
            self.max_tris,
            ctypes.c_void_p(ctypes.addressof(num_verts)),
            ctypes.c_void_p(ctypes.addressof(num_tris)),
        )

        if error:
            raise RuntimeError(
                "Buffers may not be large enough, marching cubes required at least {num_verts} vertices, and {num_tris} triangles."
            )

        # resize the geometry arrays
        self.verts.shape = (num_verts.value,)
        self.indices.shape = (num_tris.value * 3,)

        self.verts.size = num_verts.value
        self.indices.size = num_tris.value * 3
