"""Native array execution; forest preparation belongs to the shared core."""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score._native import register, workspace_sizes


def inverse_coordinates(coords, forest_arrays):
    """Initialize DOFs from coordinates and validated, device-resident indices.

    Arrays are parent, frame_x, frame_y, frame_z, doftype, in kinematic order.
    Like Torch's inverseKin initialization, this operation does not propagate
    coordinate derivatives. Subsequent DOF optimization uses forward kinematics.
    """
    if coords.dtype not in (jnp.float32, jnp.float64):
        raise TypeError("kinematics requires float32 or float64 coordinates")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("kinematic coordinates must have shape (nodes, 3)")
    count = coords.shape[0]
    if len(forest_arrays) != 5 or any(
        value.shape != (count,) or value.dtype != jnp.int32 for value in forest_arrays
    ):
        raise ValueError("kinematics requires five int32 arrays of length nodes")
    target = "tmol_inverse_kinematics"
    register(target)
    size = workspace_sizes(target, (count,), coords.dtype.itemsize * 8)[0]
    outputs = (
        jax.ShapeDtypeStruct((count, 9), coords.dtype),
        jax.ShapeDtypeStruct((size,), np.uint8),
    )

    @jax.custom_jvp
    def initialize(x):
        return jax.ffi.ffi_call(target, outputs, vmap_method="sequential")(
            x, *forest_arrays
        )[0]

    @initialize.defjvp
    def no_coordinate_derivative(primals, tangents):
        raise NotImplementedError(
            "inverse kinematics initializes independent DOFs; coordinate "
            "derivatives through initialization are not implemented"
        )

    return initialize(coords)


def forward_coordinates(dofs, tree, forward, backward):
    """Run validated device scan plans with a native first-order reverse rule.

    A plan is (device nodes, device scans, host int32 generation pairs). Forward
    mode and higher coordinate derivatives need native rules and are unsupported.
    """
    if dofs.dtype not in (jnp.float32, jnp.float64):
        raise TypeError("kinematics requires float32 or float64 DOFs")
    if dofs.ndim != 2 or dofs.shape[1] != 9:
        raise ValueError("kinematic DOFs must have shape (nodes, 9)")
    count = dofs.shape[0]
    if tree.shape != (count, 6) or tree.dtype != jnp.int32:
        raise ValueError("kinematic tree must have shape (nodes, 6) and dtype int32")
    register("tmol_forward_kinematics")
    register("tmol_kinematic_vjp")

    def execute(x, plan, derivative, *residual):
        nodes, scans, generations = plan
        generations = np.asarray(generations)
        if (
            generations.dtype != np.int32
            or generations.ndim != 2
            or generations.shape[1] != 2
        ):
            raise ValueError("kinematic generations must be int32 pairs")
        if (
            nodes.dtype != jnp.int32
            or scans.dtype != jnp.int32
            or nodes.ndim != 1
            or scans.ndim != 1
        ):
            raise ValueError("kinematic nodes and scans must be one-dimensional int32")
        target = "tmol_kinematic_vjp" if derivative else "tmol_forward_kinematics"
        outputs = (jax.ShapeDtypeStruct((count, 9 if derivative else 3), x.dtype),)
        if not derivative:
            outputs += (jax.ShapeDtypeStruct((count, 16), x.dtype),)
        branches = {}
        from tmol_jax.score._native import libraries

        for platform in libraries():
            size = _scan_workspace_size(
                platform,
                count,
                tuple(generations.ravel()),
                x.dtype.itemsize * 8,
                derivative,
            )

            def call(*args, size=size):
                return jax.ffi.ffi_call(
                    target,
                    outputs + (jax.ShapeDtypeStruct((size,), np.uint8),),
                    vmap_method="sequential",
                )(*args, generations=generations.ravel())[:-1]

            branches[platform] = call
        return jax.lax.platform_dependent(*residual, x, tree, nodes, scans, **branches)

    @jax.custom_vjp
    def transform(x):
        return execute(x, forward, False)[0]

    def transform_forward(x):
        coords, hts = execute(x, forward, False)
        return coords, (x, hts)

    def transform_backward(residual, dcoords):
        x, hts = residual
        return (execute(x, backward, True, dcoords, hts)[0],)

    transform.defvjp(transform_forward, transform_backward)
    return transform(dofs)


@lru_cache(maxsize=None)
def _scan_workspace_size(platform, count, generations, bits, derivative):
    import ctypes
    from tmol_jax.score._native import libraries

    library = libraries()[platform]
    try:
        query = library.tmol_kinematic_scan_workspace_bytes
    except AttributeError as error:
        raise RuntimeError(
            "native kinematic scans are missing; rebuild tmol-jax-native"
        ) from error
    query.argtypes = [
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_int,
    ]
    query.restype = ctypes.c_uint64
    gens = (ctypes.c_int32 * len(generations))(*generations)
    size = query(count, gens, len(generations) // 2, bits, derivative)
    if size == (1 << 64) - 1:
        raise ValueError("invalid native kinematic workspace layout")
    return size
