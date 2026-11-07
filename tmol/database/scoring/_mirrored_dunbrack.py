"""Torch storage adapter for common D-residue library preparation."""

import attr
import numpy as np
import torch

from tmol_core.database import mirrored_dunbrack as _core
from tmol_core.database.mirrored_dunbrack import (  # noqa: F401
    D_PREFIX,
    _d_table_name,
    _reflected_grid_start,
)


def _prepare(record, operation):
    """Expose host views, run scientific preparation once, restore storage."""
    originals = {}
    device = None

    def host(value):
        nonlocal device
        if isinstance(value, torch.Tensor):
            device = value.device if device is None else device
            array = value.detach().cpu().numpy()
            originals[id(array)] = value
            return array
        if attr.has(type(value)):
            return attr.evolve(
                value,
                **{
                    field.name: host(getattr(value, field.name))
                    for field in attr.fields(type(value))
                },
            )
        return value

    def restore(value):
        if isinstance(value, np.ndarray):
            original = originals.get(id(value))
            if original is not None:
                return original
            return torch.from_numpy(np.ascontiguousarray(value)).to(device=device)
        if attr.has(type(value)):
            return attr.evolve(
                value,
                **{
                    field.name: restore(getattr(value, field.name))
                    for field in attr.fields(type(value))
                },
            )
        return value

    return restore(operation(host(record)))


def _mirror_rotameric_data(data):
    return _prepare(data, _core._mirror_rotameric_data)


def _mirror_rotameric_library(library):
    return _prepare(library, _core._mirror_rotameric_library)


def _mirror_semi_rotameric_library(library):
    return _prepare(library, _core._mirror_semi_rotameric_library)


def _with_mirrored_libraries(library, d_residue_names):
    return _core._with_mirrored_libraries(
        library,
        d_residue_names,
        mirror_rotameric=_mirror_rotameric_library,
        mirror_semi=_mirror_semi_rotameric_library,
    )
