"""Static FFI transport inside the packing operations' existing JIT boundaries."""

import jax
import numpy as np

from tmol_jax.score._native import register


def stage(arguments, target, outputs, operation, extent=0, width=0, **attributes):
    register(target)
    return jax.ffi.ffi_call(
        target,
        outputs,
        input_output_aliases=attributes.pop("aliases", None),
        vmap_method="sequential",
    )(
        *arguments,
        operation=np.int64(operation),
        extent=np.int64(extent),
        width=np.int64(width),
        **attributes,
    )
