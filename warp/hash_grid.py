import ctypes
import warp_runtime_py as wp


class HashGrid:
    def __init__(self, dim_x, dim_y, dim_z, device=None):
        """Class representing a hash grid object for accelerated point queries.

        Attributes:
            id: Unique identifier for this mesh object, can be passed to kernels.
            device: Device this object lives on, all buffers must live on the same device.

        Args:
            dim_x (int): Number of cells in x-axis
            dim_y (int): Number of cells in y-axis
            dim_z (int): Number of cells in z-axis
        """

        from warp.context import runtime
        self.device = runtime.get_device(device)

        if self.device.is_cpu:
            self.id = wp.hash_grid_create_host(dim_x, dim_y, dim_z)
        else:
            self.id = wp.hash_grid_create_device(self.device.context, dim_x, dim_y, dim_z)

        # indicates whether the grid data has been reserved for use by a kernel
        self.reserved = False

    def build(self, points, radius):
        """Updates the hash grid data structure.

        This method rebuilds the underlying datastructure and should be called any time the set
        of points changes.

        Args:
            points (:class:`warp.array`): Array of points of type :class:`warp.vec3`
            radius (float): The cell size to use for bucketing points, cells are cubes with edges of this width.
                            For best performance the radius used to construct the grid should match closely to
                            the radius used when performing queries.
        """

        if self.device.is_cpu:
            wp.hash_grid_update_host(self.id, radius, ctypes.cast(points.ptr, ctypes.c_void_p), len(points))
        else:
            wp.hash_grid_update_device(self.id, radius, ctypes.cast(points.ptr, ctypes.c_void_p), len(points))
        self.reserved = True

    def reserve(self, num_points):
        if self.device.is_cpu:
            wp.hash_grid_reserve_host(self.id, num_points)
        else:
            wp.hash_grid_reserve_device(self.id, num_points)
        self.reserved = True

    def __del__(self):
        try:
            if self.device.is_cpu:
                wp.hash_grid_destroy_host(self.id)
            else:
                # use CUDA context guard to avoid side effects during garbage collection
                with self.device.context_guard:
                    wp.hash_grid_destroy_device(self.id)

        except Exception:
            pass
