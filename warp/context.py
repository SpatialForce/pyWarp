import ast
import ctypes
import gc
import inspect
import io
import os
from types import FunctionType, ModuleType
from typing import List, Union, Tuple, Any, Optional, Sequence, Dict, Callable

import numpy as np

from warp import config
from warp.codegen.adjoint import Adjoint
from warp.codegen.codegen import make_full_qualified_name
from warp.codegen.struct import Struct, StructInstance
from warp.device import Devicelike, Device
from warp.dsl.types import type_is_generic, get_signature, types_equal, vector_types, type_is_generic_scalar, \
    float_types, scalar_types, int_types, array, int32, float32, np_dtype_to_warp_type, vector, matrix, is_array, \
    float16, float_to_half_bits, launch_bounds_t, type_size_in_bytes, array_type_id, indexedarray
from warp.event import Event
from warp.function import Function, get_function_args
from warp.graph import Graph
from warp.kernel import Kernel
from warp.module import get_module, user_modules, Module
from warp.runtime import Runtime
import warp_runtime_py as wp

from warp.stream import Stream
from warp.utils.scoped_stream import ScopedStream

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


# decorator to register function, @func
def func(f):
    name = make_full_qualified_name(f)

    m = get_module(f.__module__)
    Function(
        func=f, key=name, namespace="", module=m, value_func=None
    )  # value_type not known yet, will be inferred during Adjoint.build()

    # return the top of the list of overloads for this key
    return m.functions[name]


def func_native(snippet, adj_snippet=None):
    """
    Decorator to register native code snippet, @func_native
    """

    def snippet_func(f):
        name = make_full_qualified_name(f)

        m = get_module(f.__module__)
        func = Function(
            func=f, key=name, namespace="", module=m, native_snippet=snippet, adj_native_snippet=adj_snippet
        )  # cuda snippets do not have a return value_type

        return m.functions[name]

    return snippet_func


def func_grad(forward_fn):
    """
    Decorator to register a custom gradient function for a given forward function.
    The function signature must correspond to one of the function overloads in the following way:
    the first part of the input arguments are the original input variables with the same types as their
    corresponding arguments in the original function, and the second part of the input arguments are the
    adjoint variables of the output variables (if available) of the original function with the same types as the
    output variables. The function must not return anything.
    """

    def wrapper(grad_fn):
        generic = any(type_is_generic(x) for x in forward_fn.input_types.values())
        if generic:
            raise RuntimeError(
                f"Cannot define custom grad definition for {forward_fn.key} since functions with generic input arguments are not yet supported."
            )

        reverse_args = {}
        reverse_args.update(forward_fn.input_types)

        # create temporary Adjoint instance to analyze the function signature
        adj = Adjoint(
            grad_fn, skip_forward_codegen=True, skip_reverse_codegen=False, transformers=forward_fn.adj.transformers
        )

        grad_args = adj.args
        grad_sig = get_signature([arg.type for arg in grad_args], func_name=forward_fn.key)

        generic = any(type_is_generic(x.type) for x in grad_args)
        if generic:
            raise RuntimeError(
                f"Cannot define custom grad definition for {forward_fn.key} since the provided grad function has generic input arguments."
            )

        def match_function(f):
            # check whether the function overload f matches the signature of the provided gradient function
            if not hasattr(f.adj, "return_var"):
                f.adj.build(None)
            expected_args = list(f.input_types.items())
            if f.adj.return_var is not None:
                expected_args += [(f"adj_ret_{var.label}", var.type) for var in f.adj.return_var]
            if len(grad_args) != len(expected_args):
                return False
            if any(not types_equal(a.type, exp_type) for a, (_, exp_type) in zip(grad_args, expected_args)):
                return False
            return True

        def add_custom_grad(f: Function):
            # register custom gradient function
            f.custom_grad_func = Function(
                grad_fn,
                key=f.key,
                namespace=f.namespace,
                input_types=reverse_args,
                value_func=None,
                module=f.module,
                template_func=f.template_func,
                skip_forward_codegen=True,
                custom_reverse_mode=True,
                custom_reverse_num_input_args=len(f.input_types),
                skip_adding_overload=False,
                code_transformers=f.adj.transformers,
            )
            f.adj.skip_reverse_codegen = True

        if hasattr(forward_fn, "user_overloads") and len(forward_fn.user_overloads):
            # find matching overload for which this grad function is defined
            for sig, f in forward_fn.user_overloads.items():
                if not grad_sig.startswith(sig):
                    continue
                if match_function(f):
                    add_custom_grad(f)
                    return
            raise RuntimeError(
                f"No function overload found for gradient function {grad_fn.__qualname__} for function {forward_fn.key}"
            )
        else:
            # resolve return variables
            forward_fn.adj.build(None)

            expected_args = list(forward_fn.input_types.items())
            if forward_fn.adj.return_var is not None:
                expected_args += [(f"adj_ret_{var.label}", var.type) for var in forward_fn.adj.return_var]

            # check if the signature matches this function
            if match_function(forward_fn):
                add_custom_grad(forward_fn)
            else:
                raise RuntimeError(
                    f"Gradient function {grad_fn.__qualname__} for function {forward_fn.key} has an incorrect signature. The arguments must match the "
                    "forward function arguments plus the adjoint variables corresponding to the return variables:"
                    f"\n{', '.join(map(lambda nt: f'{nt[0]}: {nt[1].__name__}', expected_args))}"
                )

    return wrapper


def func_replay(forward_fn):
    """
    Decorator to register a custom replay function for a given forward function.
    The replay function is the function version that is called in the forward phase of the backward pass (replay mode) and corresponds to the forward function by default.
    The provided function has to match the signature of one of the original forward function overloads.
    """

    def wrapper(replay_fn):
        generic = any(type_is_generic(x) for x in forward_fn.input_types.values())
        if generic:
            raise RuntimeError(
                f"Cannot define custom replay definition for {forward_fn.key} since functions with generic input arguments are not yet supported."
            )

        args = get_function_args(replay_fn)
        arg_types = list(args.values())
        generic = any(type_is_generic(x) for x in arg_types)
        if generic:
            raise RuntimeError(
                f"Cannot define custom replay definition for {forward_fn.key} since the provided replay function has generic input arguments."
            )

        f = forward_fn.get_overload(arg_types)
        if f is None:
            inputs_str = ", ".join([f"{k}: {v.__name__}" for k, v in args.items()])
            raise RuntimeError(
                f"Could not find forward definition of function {forward_fn.key} that matches custom replay definition with arguments:\n{inputs_str}"
            )
        f.custom_replay_func = Function(
            replay_fn,
            key=f"replay_{f.key}",
            namespace=f.namespace,
            input_types=f.input_types,
            value_func=f.value_func,
            module=f.module,
            template_func=f.template_func,
            skip_reverse_codegen=True,
            skip_adding_overload=True,
            code_transformers=f.adj.transformers,
        )

    return wrapper


# decorator to register kernel, @kernel, custom_name may be a string
# that creates a kernel with a different name from the actual function
def kernel(f=None, *, enable_backward=None):
    def wrapper(f, *args, **kwargs):
        options = {}

        if enable_backward is not None:
            options["enable_backward"] = enable_backward

        m = get_module(f.__module__)
        k = Kernel(
            func=f,
            key=make_full_qualified_name(f),
            module=m,
            options=options,
        )
        return k

    if f is None:
        # Arguments were passed to the decorator.
        return wrapper

    return wrapper(f)


# decorator to register struct, @struct
def struct(c):
    m = get_module(c.__module__)
    s = Struct(cls=c, key=make_full_qualified_name(c), module=m)

    return s


# overload a kernel with the given argument types
def overload(kernel, arg_types=None):
    if isinstance(kernel, Kernel):
        # handle cases where user calls us directly, e.g. wp.overload(kernel, [args...])

        if not kernel.is_generic:
            raise RuntimeError(f"Only generic kernels can be overloaded.  Kernel {kernel.key} is not generic")

        if isinstance(arg_types, list):
            arg_list = arg_types
        elif isinstance(arg_types, dict):
            # substitute named args
            arg_list = [a.type for a in kernel.adj.args]
            for arg_name, arg_type in arg_types.items():
                idx = kernel.arg_indices.get(arg_name)
                if idx is None:
                    raise RuntimeError(f"Invalid argument name '{arg_name}' in overload of kernel {kernel.key}")
                arg_list[idx] = arg_type
        elif arg_types is None:
            arg_list = []
        else:
            raise TypeError("Kernel overload types must be given in a list or dict")

        # return new kernel overload
        return kernel.add_overload(arg_list)

    elif isinstance(kernel, FunctionType):
        # handle cases where user calls us as a function decorator (@wp.overload)

        # ensure this function name corresponds to a kernel
        fn = kernel
        module = get_module(fn.__module__)
        kernel = module.kernels.get(fn.__name__)
        if kernel is None:
            raise RuntimeError(f"Failed to find a kernel named '{fn.__name__}' in module {fn.__module__}")

        if not kernel.is_generic:
            raise RuntimeError(f"Only generic kernels can be overloaded.  Kernel {kernel.key} is not generic")

        # ensure the function is defined without a body, only ellipsis (...), pass, or a string expression
        # TODO: show we allow defining a new body for kernel overloads?
        source = inspect.getsource(fn)
        tree = ast.parse(source)
        assert isinstance(tree, ast.Module)
        assert isinstance(tree.body[0], ast.FunctionDef)
        func_body = tree.body[0].body
        for node in func_body:
            if isinstance(node, ast.Pass):
                continue
            elif isinstance(node, ast.Expr) and isinstance(node.value, (ast.Str, ast.Ellipsis)):
                continue
            raise RuntimeError(
                "Illegal statement in kernel overload definition.  Only pass, ellipsis (...), comments, or docstrings are allowed"
            )

        # ensure all arguments are annotated
        argspec = inspect.getfullargspec(fn)
        if len(argspec.annotations) < len(argspec.args):
            raise RuntimeError(f"Incomplete argument annotations on kernel overload {fn.__name__}")

        # get type annotation list
        arg_list = []
        for arg_name, arg_type in argspec.annotations.items():
            if arg_name != "return":
                arg_list.append(arg_type)

        # add new overload, but we must return the original kernel from @wp.overload decorator!
        kernel.add_overload(arg_list)
        return kernel

    else:
        raise RuntimeError("wp.overload() called with invalid argument!")


builtin_functions = {}


def add_builtin(
        key,
        input_types={},
        value_type=None,
        value_func=None,
        template_func=None,
        doc="",
        namespace="wp::",
        variadic=False,
        initializer_list_func=None,
        export=True,
        group="Other",
        hidden=False,
        skip_replay=False,
        missing_grad=False,
        native_func=None,
        defaults=None,
):
    # wrap simple single-type functions with a value_func()
    if value_func is None:
        def value_func(args, kwds, templates):
            return value_type

    if initializer_list_func is None:
        def initializer_list_func(args, templates):
            return False

    if defaults is None:
        defaults = {}

    # Add specialized versions of this builtin if it's generic by matching arguments against
    # hard coded types. We do this so you can use hard coded warp types outside kernels:
    generic = any(type_is_generic(x) for x in input_types.values())
    if generic and export:
        # get a list of existing generic vector types (includes matrices and stuff)
        # so we can match arguments against them:
        generic_vtypes = [x for x in vector_types if hasattr(x, "_wp_generic_type_str_")]

        # deduplicate identical types:
        def typekey(t):
            return f"{t._wp_generic_type_str_}_{t._wp_type_params_}"

        typedict = {typekey(t): t for t in generic_vtypes}
        generic_vtypes = [typedict[k] for k in sorted(typedict.keys())]

        # collect the parent type names of all the generic arguments:
        def generic_names(l):
            for t in l:
                if hasattr(t, "_wp_generic_type_str_"):
                    yield t._wp_generic_type_str_
                elif type_is_generic_scalar(t):
                    yield t.__name__

        genericset = set(generic_names(input_types.values()))

        # for each of those type names, get a list of all hard coded types derived
        # from them:
        def derived(name):
            if name == "Float":
                return float_types
            elif name == "Scalar":
                return scalar_types
            elif name == "Int":
                return int_types
            return [x for x in generic_vtypes if x._wp_generic_type_str_ == name]

        gtypes = {k: derived(k) for k in genericset}

        # find the scalar data types supported by all the arguments by intersecting
        # sets:
        def scalar_type(t):
            if t in scalar_types:
                return t
            return [p for p in t._wp_type_params_ if p in scalar_types][0]

        scalartypes = [{scalar_type(x) for x in gtypes[k]} for k in gtypes.keys()]
        if scalartypes:
            scalartypes = scalartypes.pop().intersection(*scalartypes)

        scalartypes = list(scalartypes)
        scalartypes.sort(key=str)

        # generate function calls for each of these scalar types:
        for stype in scalartypes:
            # find concrete types for this scalar type (eg if the scalar type is float32
            # this dict will look something like this:
            # {"vec":[wp.vec2,wp.vec3,wp.vec4], "mat":[wp.mat22,wp.mat33,wp.mat44]})
            consistenttypes = {k: [x for x in v if scalar_type(x) == stype] for k, v in gtypes.items()}

            def typelist(param):
                if type_is_generic_scalar(param):
                    return [stype]
                if hasattr(param, "_wp_generic_type_str_"):
                    l = consistenttypes[param._wp_generic_type_str_]
                    return [x for x in l if types_equal(param, x, match_generic=True)]
                return [param]

            # gotta try generating function calls for all combinations of these argument types
            # now.
            import itertools

            typelists = [typelist(param) for param in input_types.values()]
            for argtypes in itertools.product(*typelists):
                # Some of these argument lists won't work, eg if the function is mul(), we won't be
                # able to do a matrix vector multiplication for a mat22 and a vec3, so we call value_func
                # on the generated argument list and skip generation if it fails.
                # This also gives us the return type, which we keep for later:
                try:
                    return_type = value_func(argtypes, {}, [])
                except Exception:
                    continue

                # The return_type might just be vector_t(length=3,dtype=wp.float32), so we've got to match that
                # in the list of hard coded types so it knows it's returning one of them:
                if hasattr(return_type, "_wp_generic_type_str_"):
                    return_type_match = [
                        x
                        for x in generic_vtypes
                        if x._wp_generic_type_str_ == return_type._wp_generic_type_str_
                           and x._wp_type_params_ == return_type._wp_type_params_
                    ]
                    if not return_type_match:
                        continue
                    return_type = return_type_match[0]

                # finally we can generate a function call for these concrete types:
                add_builtin(
                    key,
                    input_types=dict(zip(input_types.keys(), argtypes)),
                    value_type=return_type,
                    doc=doc,
                    namespace=namespace,
                    variadic=variadic,
                    initializer_list_func=initializer_list_func,
                    export=export,
                    group=group,
                    hidden=True,
                    skip_replay=skip_replay,
                    missing_grad=missing_grad,
                )

    func = Function(
        func=None,
        key=key,
        namespace=namespace,
        input_types=input_types,
        value_func=value_func,
        template_func=template_func,
        variadic=variadic,
        initializer_list_func=initializer_list_func,
        export=export,
        doc=doc,
        group=group,
        hidden=hidden,
        skip_replay=skip_replay,
        missing_grad=missing_grad,
        generic=generic,
        native_func=native_func,
        defaults=defaults,
    )

    if key in builtin_functions:
        builtin_functions[key].add_overload(func)
    else:
        builtin_functions[key] = func


def get_stream(device: Devicelike = None) -> Stream:
    """Return the stream currently used by the given device"""

    return get_device(device).stream


def set_stream(stream, device: Devicelike = None):
    """Set the stream to be used by the given device.

    If this is an external stream, caller is responsible for guaranteeing the lifetime of the stream.
    Consider using wp.ScopedStream instead.
    """

    get_device(device).stream = stream


def record_event(event: Event = None):
    """Record a CUDA event on the current stream.

    Args:
        event: Event to record. If None, a new Event will be created.

    Returns:
        The recorded event.
    """

    return get_stream().record_event(event)


def wait_event(event: Event):
    """Make the current stream wait for a CUDA event.

    Args:
        event: Event to wait for.
    """

    get_stream().wait_event(event)


def wait_stream(stream: Stream, event: Event = None):
    """Make the current stream wait for another CUDA stream to complete its work.

    Args:
        event: Event to be used.  If None, a new Event will be created.
    """

    get_stream().wait_stream(stream, event=event)


def zeros(
        shape: Tuple = None,
        dtype=float,
        device: Devicelike = None,
        requires_grad: bool = False,
        pinned: bool = False,
        **kwargs,
) -> array:
    """Return a zero-initialized array

    Args:
        shape: Array dimensions
        dtype: Type of each element, e.g.: warp.vec3, warp.mat33, etc
        device: Device that array will live on
        requires_grad: Whether the array will be tracked for back propagation
        pinned: Whether the array uses pinned host memory (only applicable to CPU arrays)

    Returns:
        A warp.array object representing the allocation
    """

    arr = empty(shape=shape, dtype=dtype, device=device, requires_grad=requires_grad, pinned=pinned, **kwargs)

    # use the CUDA default stream for synchronous behaviour with other streams
    with ScopedStream(arr.device.null_stream):
        arr.zero_()

    return arr


def zeros_like(
        src: array, device: Devicelike = None, requires_grad: bool = None, pinned: bool = None
) -> array:
    """Return a zero-initialized array with the same type and dimension of another array

    Args:
        src: The template array to use for shape, data type, and device
        device: The device where the new array will be created (defaults to src.device)
        requires_grad: Whether the array will be tracked for back propagation
        pinned: Whether the array uses pinned host memory (only applicable to CPU arrays)

    Returns:
        A warp.array object representing the allocation
    """

    arr = empty_like(src, device=device, requires_grad=requires_grad, pinned=pinned)

    arr.zero_()

    return arr


def full(
        shape: Tuple = None,
        value=0,
        dtype=Any,
        device: Devicelike = None,
        requires_grad: bool = False,
        pinned: bool = False,
        **kwargs,
) -> array:
    """Return an array with all elements initialized to the given value

    Args:
        shape: Array dimensions
        value: Element value
        dtype: Type of each element, e.g.: float, warp.vec3, warp.mat33, etc
        device: Device that array will live on
        requires_grad: Whether the array will be tracked for back propagation
        pinned: Whether the array uses pinned host memory (only applicable to CPU arrays)

    Returns:
        A warp.array object representing the allocation
    """

    if dtype == Any:
        # determine dtype from value
        value_type = type(value)
        if value_type == int:
            dtype = int32
        elif value_type == float:
            dtype = float32
        elif value_type in scalar_types or hasattr(value_type, "_wp_scalar_type_"):
            dtype = value_type
        elif isinstance(value, StructInstance):
            dtype = value._cls
        elif hasattr(value, "__len__"):
            # a sequence, assume it's a vector or matrix value
            try:
                # try to convert to a numpy array first
                na = np.array(value, copy=False)
            except Exception as e:
                raise ValueError(f"Failed to interpret the value as a vector or matrix: {e}")

            # determine the scalar type
            scalar_type = np_dtype_to_warp_type.get(na.dtype)
            if scalar_type is None:
                raise ValueError(f"Failed to convert {na.dtype} to a Warp data type")

            # determine if vector or matrix
            if na.ndim == 1:
                dtype = vector(na.size, scalar_type)
            elif na.ndim == 2:
                dtype = matrix(na.shape, scalar_type)
            else:
                raise ValueError("Values with more than two dimensions are not supported")
        else:
            raise ValueError(f"Invalid value type for Warp array: {value_type}")

    arr = empty(shape=shape, dtype=dtype, device=device, requires_grad=requires_grad, pinned=pinned, **kwargs)

    # use the CUDA default stream for synchronous behaviour with other streams
    with ScopedStream(arr.device.null_stream):
        arr.fill_(value)

    return arr


def full_like(
        src: array, value: Any, device: Devicelike = None, requires_grad: bool = None, pinned: bool = None
) -> array:
    """Return an array with all elements initialized to the given value with the same type and dimension of another array

    Args:
        src: The template array to use for shape, data type, and device
        value: Element value
        device: The device where the new array will be created (defaults to src.device)
        requires_grad: Whether the array will be tracked for back propagation
        pinned: Whether the array uses pinned host memory (only applicable to CPU arrays)

    Returns:
        A warp.array object representing the allocation
    """

    arr = empty_like(src, device=device, requires_grad=requires_grad, pinned=pinned)

    arr.fill_(value)

    return arr


def clone(src: array, device: Devicelike = None, requires_grad: bool = None, pinned: bool = None) -> array:
    """Clone an existing array, allocates a copy of the src memory

    Args:
        src: The source array to copy
        device: The device where the new array will be created (defaults to src.device)
        requires_grad: Whether the array will be tracked for back propagation
        pinned: Whether the array uses pinned host memory (only applicable to CPU arrays)

    Returns:
        A warp.array object representing the allocation
    """

    arr = empty_like(src, device=device, requires_grad=requires_grad, pinned=pinned)

    copy(arr, src)

    return arr


def empty(
        shape: Tuple = None,
        dtype=float,
        device: Devicelike = None,
        requires_grad: bool = False,
        pinned: bool = False,
        **kwargs,
) -> array:
    """Returns an uninitialized array

    Args:
        shape: Array dimensions
        dtype: Type of each element, e.g.: `warp.vec3`, `warp.mat33`, etc
        device: Device that array will live on
        requires_grad: Whether the array will be tracked for back propagation
        pinned: Whether the array uses pinned host memory (only applicable to CPU arrays)

    Returns:
        A warp.array object representing the allocation
    """

    # backwards compatibility for case where users called wp.empty(n=length, ...)
    if "n" in kwargs:
        shape = (kwargs["n"],)
        del kwargs["n"]

    # ensure shape is specified, even if creating a zero-sized array
    if shape is None:
        shape = 0

    return array(shape=shape, dtype=dtype, device=device, requires_grad=requires_grad, pinned=pinned, **kwargs)


def empty_like(
        src: array, device: Devicelike = None, requires_grad: bool = None, pinned: bool = None
) -> array:
    """Return an uninitialized array with the same type and dimension of another array

    Args:
        src: The template array to use for shape, data type, and device
        device: The device where the new array will be created (defaults to src.device)
        requires_grad: Whether the array will be tracked for back propagation
        pinned: Whether the array uses pinned host memory (only applicable to CPU arrays)

    Returns:
        A warp.array object representing the allocation
    """

    if device is None:
        device = src.device

    if requires_grad is None:
        if hasattr(src, "requires_grad"):
            requires_grad = src.requires_grad
        else:
            requires_grad = False

    if pinned is None:
        if hasattr(src, "pinned"):
            pinned = src.pinned
        else:
            pinned = False

    arr = empty(shape=src.shape, dtype=src.dtype, device=device, requires_grad=requires_grad, pinned=pinned)
    return arr


def from_numpy(
        arr: np.ndarray,
        dtype: Optional[type] = None,
        shape: Optional[Sequence[int]] = None,
        device: Optional[Devicelike] = None,
        requires_grad: bool = False,
) -> array:
    if dtype is None:
        base_type = np_dtype_to_warp_type.get(arr.dtype)
        if base_type is None:
            raise RuntimeError("Unsupported NumPy data type '{}'.".format(arr.dtype))

        dim_count = len(arr.shape)
        if dim_count == 2:
            dtype = vector(length=arr.shape[1], dtype=base_type)
        elif dim_count == 3:
            dtype = matrix(shape=(arr.shape[1], arr.shape[2]), dtype=base_type)
        else:
            dtype = base_type

    return array(
        data=arr,
        dtype=dtype,
        shape=shape,
        owner=False,
        device=device,
        requires_grad=requires_grad,
    )


# given a kernel destination argument type and a value convert
#  to a c-type that can be passed to a kernel
def pack_arg(kernel, arg_type, arg_name, value, device, adjoint=False):
    if is_array(arg_type):
        if value is None:
            # allow for NULL arrays
            return arg_type.__ctype__()

        else:
            # check for array type
            # - in forward passes, array types have to match
            # - in backward passes, indexed array gradients are regular arrays
            if adjoint:
                array_matches = isinstance(value, array)
            else:
                array_matches = type(value) is type(arg_type)

            if not array_matches:
                adj = "adjoint " if adjoint else ""
                raise RuntimeError(
                    f"Error launching kernel '{kernel.key}', {adj}argument '{arg_name}' expects an array of type {type(arg_type)}, but passed value has type {type(value)}."
                )

            # check subtype
            if not types_equal(value.dtype, arg_type.dtype):
                adj = "adjoint " if adjoint else ""
                raise RuntimeError(
                    f"Error launching kernel '{kernel.key}', {adj}argument '{arg_name}' expects an array with dtype={arg_type.dtype} but passed array has dtype={value.dtype}."
                )

            # check dimensions
            if value.ndim != arg_type.ndim:
                adj = "adjoint " if adjoint else ""
                raise RuntimeError(
                    f"Error launching kernel '{kernel.key}', {adj}argument '{arg_name}' expects an array with {arg_type.ndim} dimension(s) but the passed array has {value.ndim} dimension(s)."
                )

            # check device
            # if a.device != device and not device.can_access(a.device):
            if value.device != device:
                raise RuntimeError(
                    f"Error launching kernel '{kernel.key}', trying to launch on device='{device}', but input array for argument '{arg_name}' is on device={value.device}."
                )

            return value.__ctype__()

    elif isinstance(arg_type, Struct):
        assert value is not None
        return value.__ctype__()

    # try to convert to a value type (vec3, mat33, etc)
    elif issubclass(arg_type, ctypes.Array):
        if types_equal(type(value), arg_type):
            return value
        else:
            # try constructing the required value from the argument (handles tuple / list, Gf.Vec3 case)
            try:
                return arg_type(value)
            except Exception:
                raise ValueError(f"Failed to convert argument for param {arg_name} to {type_str(arg_type)}")

    elif isinstance(value, bool):
        return ctypes.c_bool(value)

    elif isinstance(value, arg_type):
        try:
            # try to pack as a scalar type
            if arg_type is float16:
                return arg_type._type_(float_to_half_bits(value.value))
            else:
                return arg_type._type_(value.value)
        except Exception:
            raise RuntimeError(
                "Error launching kernel, unable to pack kernel parameter type "
                f"{type(value)} for param {arg_name}, expected {arg_type}"
            )

    else:
        try:
            # try to pack as a scalar type
            if arg_type is float16:
                return arg_type._type_(float_to_half_bits(value))
            else:
                return arg_type._type_(value)
        except Exception as e:
            print(e)
            raise RuntimeError(
                "Error launching kernel, unable to pack kernel parameter type "
                f"{type(value)} for param {arg_name}, expected {arg_type}"
            )


# represents all data required for a kernel launch
# so that launches can be replayed quickly, use `wp.launch(..., record_cmd=True)`
class Launch:
    def __init__(self, kernel, device, hooks=None, params=None, params_addr=None, bounds=None, max_blocks=0):
        # if not specified look up hooks
        if not hooks:
            module = kernel.module
            if not module.load(device):
                return

            hooks = module.get_kernel_hooks(kernel, device)

        # if not specified set a zero bound
        if not bounds:
            bounds = launch_bounds_t(0)

        # if not specified then build a list of default value params for args
        if not params:
            params = []
            params.append(bounds)

            for a in kernel.adj.args:
                if isinstance(a.type, array):
                    params.append(a.type.__ctype__())
                elif isinstance(a.type, Struct):
                    params.append(a.type().__ctype__())
                else:
                    params.append(pack_arg(kernel, a.type, a.label, 0, device, False))

            kernel_args = [ctypes.c_void_p(ctypes.addressof(x)) for x in params]
            kernel_params = (ctypes.c_void_p * len(kernel_args))(*kernel_args)

            params_addr = kernel_params

        self.kernel = kernel
        self.hooks = hooks
        self.params = params
        self.params_addr = params_addr
        self.device = device
        self.bounds = bounds
        self.max_blocks = max_blocks

    def set_dim(self, dim):
        self.bounds = launch_bounds_t(dim)

        # launch bounds always at index 0
        self.params[0] = self.bounds

        # for CUDA kernels we need to update the address to each arg
        if self.params_addr:
            self.params_addr[0] = ctypes.c_void_p(ctypes.addressof(self.bounds))

    # set kernel param at an index, will convert to ctype as necessary
    def set_param_at_index(self, index, value):
        arg_type = self.kernel.adj.args[index].type
        arg_name = self.kernel.adj.args[index].label

        carg = pack_arg(self.kernel, arg_type, arg_name, value, self.device, False)

        self.params[index + 1] = carg

        # for CUDA kernels we need to update the address to each arg
        if self.params_addr:
            self.params_addr[index + 1] = ctypes.c_void_p(ctypes.addressof(carg))

    # set kernel param at an index without any type conversion
    # args must be passed as ctypes or basic int / float types
    def set_param_at_index_from_ctype(self, index, value):
        if isinstance(value, ctypes.Structure):
            # not sure how to directly assign struct->struct without reallocating using ctypes
            self.params[index + 1] = value

            # for CUDA kernels we need to update the address to each arg
            if self.params_addr:
                self.params_addr[index + 1] = ctypes.c_void_p(ctypes.addressof(value))

        else:
            self.params[index + 1].__init__(value)

    # set kernel param by argument name
    def set_param_by_name(self, name, value):
        for i, arg in enumerate(self.kernel.adj.args):
            if arg.label == name:
                self.set_param_at_index(i, value)

    # set kernel param by argument name with no type conversions
    def set_param_by_name_from_ctype(self, name, value):
        # lookup argument index
        for i, arg in enumerate(self.kernel.adj.args):
            if arg.label == name:
                self.set_param_at_index_from_ctype(i, value)

    # set all params
    def set_params(self, values):
        for i, v in enumerate(values):
            self.set_param_at_index(i, v)

    # set all params without performing type-conversions
    def set_params_from_ctypes(self, values):
        for i, v in enumerate(values):
            self.set_param_at_index_from_ctype(i, v)

    def launch(self) -> Any:
        if self.device.is_cpu:
            self.hooks.forward(*self.params)
        else:
            wp.cuda_launch_kernel(
                self.device.context, self.hooks.forward, self.bounds.size, self.max_blocks, self.params_addr
            )


def launch(
        kernel,
        dim: Tuple[int],
        inputs: List,
        outputs: List = [],
        adj_inputs: List = [],
        adj_outputs: List = [],
        device: Devicelike = None,
        stream: Stream = None,
        adjoint=False,
        record_tape=True,
        record_cmd=False,
        max_blocks=0,
):
    """Launch a Warp kernel on the target device

    Kernel launches are asynchronous with respect to the calling Python thread.

    Args:
        kernel: The name of a Warp kernel function, decorated with the ``@wp.kernel`` decorator
        dim: The number of threads to launch the kernel, can be an integer, or a Tuple of ints with max of 4 dimensions
        inputs: The input parameters to the kernel
        outputs: The output parameters (optional)
        adj_inputs: The adjoint inputs (optional)
        adj_outputs: The adjoint outputs (optional)
        device: The device to launch on (optional)
        stream: The stream to launch on (optional)
        adjoint: Whether to run forward or backward pass (typically use False)
        record_tape: When true the launch will be recorded the global wp.Tape() object when present
        record_cmd: When True the launch will be returned as a ``Launch`` command object, the launch will not occur until the user calls ``cmd.launch()``
        max_blocks: The maximum number of CUDA thread blocks to use. Only has an effect for CUDA kernel launches.
            If negative or zero, the maximum hardware value will be used.
    """

    assert_initialized()

    # if stream is specified, use the associated device
    if stream is not None:
        device = stream.device
    else:
        device = runtime.get_device(device)

    # check function is a Kernel
    if isinstance(kernel, Kernel) is False:
        raise RuntimeError("Error launching kernel, can only launch functions decorated with @wp.kernel.")

    # debugging aid
    if config.print_launches:
        print(f"kernel: {kernel.key} dim: {dim} inputs: {inputs} outputs: {outputs} device: {device}")

    # construct launch bounds
    bounds = launch_bounds_t(dim)

    if bounds.size > 0:
        # first param is the number of threads
        params = []
        params.append(bounds)

        # converts arguments to kernel's expected ctypes and packs into params
        def pack_args(args, params, adjoint=False):
            for i, a in enumerate(args):
                arg_type = kernel.adj.args[i].type
                arg_name = kernel.adj.args[i].label

                params.append(pack_arg(kernel, arg_type, arg_name, a, device, adjoint))

        fwd_args = inputs + outputs
        adj_args = adj_inputs + adj_outputs

        if (len(fwd_args)) != (len(kernel.adj.args)):
            raise RuntimeError(
                f"Error launching kernel '{kernel.key}', passed {len(fwd_args)} arguments but kernel requires {len(kernel.adj.args)}."
            )

        # if it's a generic kernel, infer the required overload from the arguments
        if kernel.is_generic:
            fwd_types = kernel.infer_argument_types(fwd_args)
            kernel = kernel.get_overload(fwd_types)

        # delay load modules, including new overload if needed
        module = kernel.module
        if not module.load(device):
            return

        # late bind
        hooks = module.get_kernel_hooks(kernel, device)

        pack_args(fwd_args, params)
        pack_args(adj_args, params, adjoint=True)

        # run kernel
        if device.is_cpu:
            if adjoint:
                if hooks.backward is None:
                    raise RuntimeError(
                        f"Failed to find backward kernel '{kernel.key}' from module '{kernel.module.name}' for device '{device}'"
                    )

                hooks.backward(*params)

            else:
                if hooks.forward is None:
                    raise RuntimeError(
                        f"Failed to find forward kernel '{kernel.key}' from module '{kernel.module.name}' for device '{device}'"
                    )

                if record_cmd:
                    launch = Launch(
                        kernel=kernel, hooks=hooks, params=params, params_addr=None, bounds=bounds, device=device
                    )
                    return launch
                else:
                    hooks.forward(*params)

        else:
            kernel_args = [ctypes.c_void_p(ctypes.addressof(x)) for x in params]
            kernel_params = (ctypes.c_void_p * len(kernel_args))(*kernel_args)

            with ScopedStream(stream):
                if adjoint:
                    if hooks.backward is None:
                        raise RuntimeError(
                            f"Failed to find backward kernel '{kernel.key}' from module '{kernel.module.name}' for device '{device}'"
                        )

                    wp.cuda_launch_kernel(
                        device.context, hooks.backward, bounds.size, max_blocks, kernel_params
                    )

                else:
                    if hooks.forward is None:
                        raise RuntimeError(
                            f"Failed to find forward kernel '{kernel.key}' from module '{kernel.module.name}' for device '{device}'"
                        )

                    if record_cmd:
                        launch = Launch(
                            kernel=kernel,
                            hooks=hooks,
                            params=params,
                            params_addr=kernel_params,
                            bounds=bounds,
                            device=device,
                        )
                        return launch

                    else:
                        # launch
                        wp.cuda_launch_kernel(
                            device.context, hooks.forward, bounds.size, max_blocks, kernel_params
                        )

                try:
                    runtime.verify_cuda_device(device)
                except Exception as e:
                    print(f"Error launching kernel: {kernel.key} on device {device}")
                    raise e

    # record on tape if one is active
    if runtime.tape and record_tape:
        runtime.tape.record_launch(kernel, dim, max_blocks, inputs, outputs, device)


def synchronize():
    """Manually synchronize the calling CPU thread with any outstanding CUDA work on all devices

    This method allows the host application code to ensure that any kernel launches
    or memory copies have completed.
    """

    if is_cuda_driver_initialized():
        # save the original context to avoid side effects
        saved_context = wp.cuda_context_get_current()

        # TODO: only synchronize devices that have outstanding work
        for device in runtime.cuda_devices:
            # avoid creating primary context if the device has not been used yet
            if device.has_context:
                if device.is_capturing:
                    raise RuntimeError(f"Cannot synchronize device {device} while graph capture is active")

                wp.cuda_context_synchronize(device.context)

        # restore the original context to avoid side effects
        wp.cuda_context_set_current(saved_context)


def synchronize_device(device: Devicelike = None):
    """Manually synchronize the calling CPU thread with any outstanding CUDA work on the specified device

    This method allows the host application code to ensure that any kernel launches
    or memory copies have completed.

    Args:
        device: Device to synchronize.  If None, synchronize the current CUDA device.
    """

    device = runtime.get_device(device)
    if device.is_cuda:
        if device.is_capturing:
            raise RuntimeError(f"Cannot synchronize device {device} while graph capture is active")

        wp.cuda_context_synchronize(device.context)


def synchronize_stream(stream_or_device=None):
    """Manually synchronize the calling CPU thread with any outstanding CUDA work on the specified stream.

    Args:
        stream_or_device: `wp.Stream` or a device.  If the argument is a device, synchronize the device's current stream.
    """

    if isinstance(stream_or_device, Stream):
        stream = stream_or_device
    else:
        stream = runtime.get_device(stream_or_device).stream

    wp.cuda_stream_synchronize(stream.device.context, stream.cuda_stream)


def force_load(device: Union[Device, str, List[Device], List[str]] = None, modules: List[Module] = None):
    """Force user-defined kernels to be compiled and loaded

    Args:
        device: The device or list of devices to load the modules on.  If None, load on all devices.
        modules: List of modules to load.  If None, load all imported modules.
    """

    if is_cuda_driver_initialized():
        # save original context to avoid side effects
        saved_context = wp.cuda_context_get_current()

    if device is None:
        devices = get_devices()
    elif isinstance(device, list):
        devices = [get_device(device_item) for device_item in device]
    else:
        devices = [get_device(device)]

    if modules is None:
        modules = user_modules.values()

    for d in devices:
        for m in modules:
            m.load(d)

    if is_cuda_available():
        # restore original context to avoid side effects
        wp.cuda_context_set_current(saved_context)


def load_module(
        module: Union[Module, ModuleType, str] = None, device: Union[Device, str] = None, recursive: bool = False
):
    """Force user-defined module to be compiled and loaded

    Args:
        module: The module to load.  If None, load the current module.
        device: The device to load the modules on.  If None, load on all devices.
        recursive: Whether to load submodules.  E.g., if the given module is `warp.sim`, this will also load `warp.sim.model`, `warp.sim.articulation`, etc.

    Note: A module must be imported before it can be loaded by this function.
    """

    if module is None:
        # if module not specified, use the module that called us
        module = inspect.getmodule(inspect.stack()[1][0])
        module_name = module.__name__
    elif isinstance(module, Module):
        module_name = module.name
    elif isinstance(module, ModuleType):
        module_name = module.__name__
    elif isinstance(module, str):
        module_name = module
    else:
        raise TypeError(f"Argument must be a module, got {type(module)}")

    modules = []

    # add the given module, if found
    m = user_modules.get(module_name)
    if m is not None:
        modules.append(m)

    # add submodules, if recursive
    if recursive:
        prefix = module_name + "."
        for name, mod in user_modules.items():
            if name.startswith(prefix):
                modules.append(mod)

    force_load(device=device, modules=modules)


def set_module_options(options: Dict[str, Any], module: Optional[Any] = None):
    """Set options for the current module.

    Options can be used to control runtime compilation and code-generation
    for the current module individually. Available options are listed below.

    * **mode**: The compilation mode to use, can be "debug", or "release", defaults to the value of ``warp.config.mode``.
    * **max_unroll**: The maximum fixed-size loop to unroll (default 16)

    Args:

        options: Set of key-value option pairs
    """

    if module is None:
        m = inspect.getmodule(inspect.stack()[1][0])
    else:
        m = module

    get_module(m.__name__).options.update(options)
    get_module(m.__name__).unload()


def get_module_options(module: Optional[Any] = None) -> Dict[str, Any]:
    """Returns a list of options for the current module."""
    if module is None:
        m = inspect.getmodule(inspect.stack()[1][0])
    else:
        m = module

    return get_module(m.__name__).options


def capture_begin(device: Devicelike = None, stream=None, force_module_load=True):
    """Begin capture of a CUDA graph

    Captures all subsequent kernel launches and memory operations on CUDA devices.
    This can be used to record large numbers of kernels and replay them with low-overhead.

    Args:

        device: The device to capture on, if None the current CUDA device will be used
        stream: The CUDA stream to capture on
        force_module_load: Whether or not to force loading of all kernels before capture, in general it is better to use :func:`~warp.load_module()` to selectively load kernels.

    """

    if config.verify_cuda is True:
        raise RuntimeError("Cannot use CUDA error verification during graph capture")

    if stream is not None:
        device = stream.device
    else:
        device = runtime.get_device(device)
        if not device.is_cuda:
            raise RuntimeError("Must be a CUDA device")

    if force_module_load:
        force_load(device)

    device.is_capturing = True

    # disable garbage collection to avoid older allocations getting collected during graph capture
    gc.disable()

    with ScopedStream(stream):
        wp.cuda_graph_begin_capture(device.context)


def capture_end(device: Devicelike = None, stream=None) -> Graph:
    """Ends the capture of a CUDA graph

    Returns:
        A handle to a CUDA graph object that can be launched with :func:`~warp.capture_launch()`
    """

    if stream is not None:
        device = stream.device
    else:
        device = runtime.get_device(device)
        if not device.is_cuda:
            raise RuntimeError("Must be a CUDA device")

    with ScopedStream(stream):
        graph = wp.cuda_graph_end_capture(device.context)

    device.is_capturing = False

    # re-enable GC
    gc.enable()

    if graph is None:
        raise RuntimeError(
            "Error occurred during CUDA graph capture. This could be due to an unintended allocation or CPU/GPU synchronization event."
        )
    else:
        return Graph(device, graph)


def capture_launch(graph: Graph, stream: Stream = None):
    """Launch a previously captured CUDA graph

    Args:
        graph: A Graph as returned by :func:`~warp.capture_end()`
        stream: A Stream to launch the graph on (optional)
    """

    if stream is not None:
        if stream.device != graph.device:
            raise RuntimeError(f"Cannot launch graph from device {graph.device} on stream from device {stream.device}")
        device = stream.device
    else:
        device = graph.device

    with ScopedStream(stream):
        wp.cuda_graph_launch(device.context, graph.exec)


def copy(
        dest: array, src: array, dest_offset: int = 0, src_offset: int = 0, count: int = 0,
        stream: Stream = None
):
    """Copy array contents from src to dest

    Args:
        dest: Destination array, must be at least as big as source buffer
        src: Source array
        dest_offset: Element offset in the destination array
        src_offset: Element offset in the source array
        count: Number of array elements to copy (will copy all elements if set to 0)
        stream: The stream on which to perform the copy (optional)

    """

    if not is_array(src) or not is_array(dest):
        raise RuntimeError("Copy source and destination must be arrays")

    # backwards compatibility, if count is zero then copy entire src array
    if count <= 0:
        count = src.size

    if count == 0:
        return

    # copying non-contiguous arrays requires that they are on the same device
    if not (src.is_contiguous and dest.is_contiguous) and src.device != dest.device:
        if dest.is_contiguous:
            # make a contiguous copy of the source array
            src = src.contiguous()
        else:
            # make a copy of the source array on the destination device
            src = src.to(dest.device)

    if src.is_contiguous and dest.is_contiguous:
        bytes_to_copy = count * type_size_in_bytes(src.dtype)

        src_size_in_bytes = src.size * type_size_in_bytes(src.dtype)
        dst_size_in_bytes = dest.size * type_size_in_bytes(dest.dtype)

        src_offset_in_bytes = src_offset * type_size_in_bytes(src.dtype)
        dst_offset_in_bytes = dest_offset * type_size_in_bytes(dest.dtype)

        src_ptr = src.ptr + src_offset_in_bytes
        dst_ptr = dest.ptr + dst_offset_in_bytes

        if src_offset_in_bytes + bytes_to_copy > src_size_in_bytes:
            raise RuntimeError(
                f"Trying to copy source buffer with size ({bytes_to_copy}) from offset ({src_offset_in_bytes}) is larger than source size ({src_size_in_bytes})"
            )

        if dst_offset_in_bytes + bytes_to_copy > dst_size_in_bytes:
            raise RuntimeError(
                f"Trying to copy source buffer with size ({bytes_to_copy}) to offset ({dst_offset_in_bytes}) is larger than destination size ({dst_size_in_bytes})"
            )

        if src.device.is_cpu and dest.device.is_cpu:
            wp.memcpy_h2h(dst_ptr, src_ptr, bytes_to_copy)
        else:
            # figure out the CUDA context/stream for the copy
            if stream is not None:
                copy_device = stream.device
            elif dest.device.is_cuda:
                copy_device = dest.device
            else:
                copy_device = src.device

            with ScopedStream(stream):
                if src.device.is_cpu and dest.device.is_cuda:
                    wp.memcpy_h2d(copy_device.context, dst_ptr, src_ptr, bytes_to_copy)
                elif src.device.is_cuda and dest.device.is_cpu:
                    wp.memcpy_d2h(copy_device.context, dst_ptr, src_ptr, bytes_to_copy)
                elif src.device.is_cuda and dest.device.is_cuda:
                    if src.device == dest.device:
                        wp.memcpy_d2d(copy_device.context, dst_ptr, src_ptr, bytes_to_copy)
                    else:
                        wp.memcpy_peer(copy_device.context, dst_ptr, src_ptr, bytes_to_copy)
                else:
                    raise RuntimeError("Unexpected source and destination combination")

    else:
        # handle non-contiguous and indexed arrays

        if src.shape != dest.shape:
            raise RuntimeError("Incompatible array shapes")

        src_elem_size = type_size_in_bytes(src.dtype)
        dst_elem_size = type_size_in_bytes(dest.dtype)

        if src_elem_size != dst_elem_size:
            raise RuntimeError("Incompatible array data types")

        # can't copy to/from fabric arrays of arrays, because they are jagged arrays of arbitrary lengths
        # TODO?
        if (
                isinstance(src, (fabricarray, indexedfabricarray))
                and src.ndim > 1
                or isinstance(dest, (fabricarray, indexedfabricarray))
                and dest.ndim > 1
        ):
            raise RuntimeError("Copying to/from Fabric arrays of arrays is not supported")

        src_desc = src.__ctype__()
        dst_desc = dest.__ctype__()
        src_ptr = ctypes.pointer(src_desc)
        dst_ptr = ctypes.pointer(dst_desc)
        src_type = array_type_id(src)
        dst_type = array_type_id(dest)

        if src.device.is_cuda:
            with ScopedStream(stream):
                wp.array_copy_device(src.device.context, dst_ptr, src_ptr, dst_type, src_type, src_elem_size)
        else:
            wp.array_copy_host(dst_ptr, src_ptr, dst_type, src_type, src_elem_size)

    # copy gradient, if needed
    if hasattr(src, "grad") and src.grad is not None and hasattr(dest, "grad") and dest.grad is not None:
        copy(dest.grad, src.grad, stream=stream)


def type_str(t):
    if t is None:
        return "None"
    elif t == Any:
        return "Any"
    elif t == Callable:
        return "Callable"
    elif t == Tuple[int, int]:
        return "Tuple[int, int]"
    elif isinstance(t, int):
        return str(t)
    elif isinstance(t, List):
        return "Tuple[" + ", ".join(map(type_str, t)) + "]"
    elif isinstance(t, array):
        return f"Array[{type_str(t.dtype)}]"
    elif isinstance(t, indexedarray):
        return f"IndexedArray[{type_str(t.dtype)}]"
    elif isinstance(t, fabricarray):
        return f"FabricArray[{type_str(t.dtype)}]"
    elif isinstance(t, indexedfabricarray):
        return f"IndexedFabricArray[{type_str(t.dtype)}]"
    elif hasattr(t, "_wp_generic_type_str_"):
        generic_type = t._wp_generic_type_str_

        # for concrete vec/mat types use the short name
        if t in vector_types:
            return t.__name__

        # for generic vector / matrix type use a Generic type hint
        if generic_type == "vec_t":
            # return f"Vector"
            return f"Vector[{type_str(t._wp_type_params_[0])},{type_str(t._wp_scalar_type_)}]"
        elif generic_type == "quat_t":
            # return f"Quaternion"
            return f"Quaternion[{type_str(t._wp_scalar_type_)}]"
        elif generic_type == "mat_t":
            # return f"Matrix"
            return f"Matrix[{type_str(t._wp_type_params_[0])},{type_str(t._wp_type_params_[1])},{type_str(t._wp_scalar_type_)}]"
        elif generic_type == "transform_t":
            # return f"Transformation"
            return f"Transformation[{type_str(t._wp_scalar_type_)}]"
        else:
            raise TypeError("Invalid vector or matrix dimensions")
    else:
        return t.__name__


def print_function(f, file, noentry=False):  # pragma: no cover
    """Writes a function definition to a file for use in reST documentation

    Args:
        f: The function being written
        file: The file object for output
        noentry: If True, then the :noindex: and :nocontentsentry: directive
          options will be added

    Returns:
        A bool indicating True if f was written to file
    """

    if f.hidden:
        return False

    args = ", ".join(f"{k}: {type_str(v)}" for k, v in f.input_types.items())

    return_type = ""

    try:
        # todo: construct a default value for each of the functions args
        # so we can generate the return type for overloaded functions
        return_type = " -> " + type_str(f.value_func(None, None, None))
    except Exception:
        pass

    print(f".. function:: {f.key}({args}){return_type}", file=file)
    if noentry:
        print("   :noindex:", file=file)
        print("   :nocontentsentry:", file=file)
    print("", file=file)

    if f.doc != "":
        if not f.missing_grad:
            print(f"   {f.doc}", file=file)
        else:
            print(f"   {f.doc} [1]_", file=file)
        print("", file=file)

    print(file=file)

    return True


def print_builtins(file):  # pragma: no cover
    header = (
        "..\n"
        "   Autogenerated File - Do not edit. Run build_docs.py to generate.\n"
        "\n"
        ".. functions:\n"
        ".. currentmodule:: warp\n"
        "\n"
        "Kernel Reference\n"
        "================"
    )

    print(header, file=file)

    # type definitions of all functions by group
    print("\nScalar Types", file=file)
    print("------------", file=file)

    for t in scalar_types:
        print(f".. class:: {t.__name__}", file=file)
    # Manually add wp.bool since it's inconvenient to add to wp.types.scalar_types:
    print(f".. class:: {bool.__name__}", file=file)

    print("\n\nVector Types", file=file)
    print("------------", file=file)

    for t in vector_types:
        print(f".. class:: {t.__name__}", file=file)

    print("\nGeneric Types", file=file)
    print("-------------", file=file)

    print(".. class:: Int", file=file)
    print(".. class:: Float", file=file)
    print(".. class:: Scalar", file=file)
    print(".. class:: Vector", file=file)
    print(".. class:: Matrix", file=file)
    print(".. class:: Quaternion", file=file)
    print(".. class:: Transformation", file=file)
    print(".. class:: Array", file=file)

    # build dictionary of all functions by group
    groups = {}

    for k, f in builtin_functions.items():
        # build dict of groups
        if f.group not in groups:
            groups[f.group] = []

        # append all overloads to the group
        for o in f.overloads:
            groups[f.group].append(o)

    # Keep track of what function names have been written
    written_functions = {}

    for k, g in groups.items():
        print("\n", file=file)
        print(k, file=file)
        print("---------------", file=file)

        for f in g:
            if f.key in written_functions:
                # Add :noindex: + :nocontentsentry: since Sphinx gets confused
                print_function(f, file=file, noentry=True)
            else:
                if print_function(f, file=file):
                    written_functions[f.key] = []

    # footnotes
    print(".. rubric:: Footnotes", file=file)
    print(".. [1] Note: function gradients not implemented for backpropagation.", file=file)


def export_stubs(file):  # pragma: no cover
    """Generates stub file for auto-complete of builtin functions"""

    import textwrap

    print(
        "# Autogenerated file, do not edit, this file provides stubs for builtins autocomplete in VSCode, PyCharm, etc",
        file=file,
    )
    print("", file=file)
    print("from typing import Any", file=file)
    print("from typing import Tuple", file=file)
    print("from typing import Callable", file=file)
    print("from typing import TypeVar", file=file)
    print("from typing import Generic", file=file)
    print("from typing import overload as over", file=file)
    print(file=file)

    # type hints, these need to be mirrored into the stubs file
    print('Length = TypeVar("Length", bound=int)', file=file)
    print('Rows = TypeVar("Rows", bound=int)', file=file)
    print('Cols = TypeVar("Cols", bound=int)', file=file)
    print('DType = TypeVar("DType")', file=file)

    print('Int = TypeVar("Int")', file=file)
    print('Float = TypeVar("Float")', file=file)
    print('Scalar = TypeVar("Scalar")', file=file)
    print("Vector = Generic[Length, Scalar]", file=file)
    print("Matrix = Generic[Rows, Cols, Scalar]", file=file)
    print("Quaternion = Generic[Float]", file=file)
    print("Transformation = Generic[Float]", file=file)
    print("Array = Generic[DType]", file=file)
    print("FabricArray = Generic[DType]", file=file)
    print("IndexedFabricArray = Generic[DType]", file=file)

    # prepend __init__.py
    with open(os.path.join(os.path.dirname(file.name), "__init__.py")) as header_file:
        # strip comment lines
        lines = [line for line in header_file if not line.startswith("#")]
        header = "".join(lines)

    print(header, file=file)
    print(file=file)

    for k, g in builtin_functions.items():
        for f in g.overloads:
            args = ", ".join(f"{k}: {type_str(v)}" for k, v in f.input_types.items())

            return_str = ""

            if f.export is False or f.hidden is True:  # or f.generic:
                continue

            try:
                # todo: construct a default value for each of the functions args
                # so we can generate the return type for overloaded functions
                return_type = f.value_func(None, None, None)
                if return_type:
                    return_str = " -> " + type_str(return_type)

            except Exception:
                pass

            print("@over", file=file)
            print(f"def {f.key}({args}){return_str}:", file=file)
            print('    """', file=file)
            print(textwrap.indent(text=f.doc, prefix="    "), file=file)
            print('    """', file=file)
            print("    ...\n\n", file=file)


def export_builtins(file: io.TextIOBase):  # pragma: no cover
    def ctype_str(t):
        if isinstance(t, int):
            return "int"
        elif isinstance(t, float):
            return "float"
        else:
            return t.__name__

    file.write("namespace wp {\n\n")
    file.write('extern "C" {\n\n')

    for k, g in builtin_functions.items():
        for f in g.overloads:
            if f.export is False or f.generic:
                continue

            simple = True
            for k, v in f.input_types.items():
                if isinstance(v, array) or v == Any or v == Callable or v == Tuple:
                    simple = False
                    break

            # only export simple types that don't use arrays
            # or templated types
            if not simple or f.variadic:
                continue

            args = ", ".join(f"{ctype_str(v)} {k}" for k, v in f.input_types.items())
            params = ", ".join(f.input_types.keys())

            return_type = ""

            try:
                # todo: construct a default value for each of the functions args
                # so we can generate the return type for overloaded functions
                return_type = ctype_str(f.value_func(None, None, None))
            except Exception:
                continue

            if return_type.startswith("Tuple"):
                continue

            if args == "":
                file.write(f"WP_API void {f.mangled_name}({return_type}* ret) {{ *ret = wp::{f.key}({params}); }}\n")
            elif return_type == "None":
                file.write(f"WP_API void {f.mangled_name}({args}) {{ wp::{f.key}({params}); }}\n")
            else:
                file.write(
                    f"WP_API void {f.mangled_name}({args}, {return_type}* ret) {{ *ret = wp::{f.key}({params}); }}\n"
                )

    file.write('\n}  // extern "C"\n\n')
    file.write("}  // namespace wp\n")
