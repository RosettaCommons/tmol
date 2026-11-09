"""Shared ligand preparation with JAX refinement and parameter context.

CIF entry points use the common framework-free atom-array reader.
Preparation is host orchestration; its numerical refinement executes through JAX.
"""

import inspect

from tmol_core import ligand as _core

__all__ = _core.__all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    value = getattr(_core, name)
    if inspect.isfunction(value):
        from ._execution import with_jax_execution

        value = with_jax_execution(value)
    globals()[name] = value
    return value
