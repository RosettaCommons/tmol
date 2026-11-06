"""Torch defaults and result conversion for the shared ligand pipeline."""

from functools import wraps
from inspect import signature

import attr

from tmol.database import ParameterDatabase
from tmol.io._canonical_ordering import CanonicalOrdering
from tmol_core.io.canonical_ordering import CanonicalOrdering as CoreCanonicalOrdering


def _torch_ordering(result):
    if isinstance(result, CoreCanonicalOrdering) and not isinstance(
        result, CanonicalOrdering
    ):
        return CanonicalOrdering(
            **{
                field.name: getattr(result, field.name)
                for field in attr.fields(CoreCanonicalOrdering)
                if field.init
            }
        )
    if isinstance(result, tuple):
        return tuple(_torch_ordering(value) for value in result)
    return result


def _with_torch_execution(function):
    """Supply frontend services without altering shared chemistry or algorithms."""
    sig = signature(function)

    @wraps(function)
    def call(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        if "param_db" in sig.parameters and bound.arguments.get("param_db") is None:
            bound.arguments["param_db"] = ParameterDatabase.get_default()
        if (
            "conformer_execution" in sig.parameters
            and bound.arguments.get("conformer_execution") is None
        ):
            from . import _conformer_generation

            bound.arguments["conformer_execution"] = _conformer_generation
        if (
            "atom_array_reader" in sig.parameters
            and bound.arguments.get("atom_array_reader") is None
        ):
            from tmol.io._structure_io import atom_array_from_file

            bound.arguments["atom_array_reader"] = atom_array_from_file
        return _torch_ordering(function(*bound.args, **bound.kwargs))

    return call
