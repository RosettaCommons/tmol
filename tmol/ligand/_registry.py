"""Torch compatibility for shared ligand registration."""

from tmol_core.ligand import _registry as _core
from ._execution import _with_torch_execution


def __getattr__(name):
    return getattr(_core, name)


_rebuild_canonical_ordering = _with_torch_execution(_core._rebuild_canonical_ordering)
