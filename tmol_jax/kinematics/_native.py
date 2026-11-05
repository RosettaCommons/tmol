"""Native array execution; forest preparation belongs to the shared core."""

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
