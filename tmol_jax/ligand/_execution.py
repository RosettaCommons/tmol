"""Frontend parameter and execution services for common ligand preparation."""

from functools import wraps
from inspect import signature

import attr

from tmol_core.io.canonical_ordering import CanonicalOrdering as CoreCanonicalOrdering
from tmol_jax.database import ParameterDatabase
from tmol_jax.io.canonical_ordering import CanonicalOrdering


def _jax_ordering(result):
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
        return tuple(_jax_ordering(value) for value in result)
    return result


def with_jax_execution(function):
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
        return _jax_ordering(function(*bound.args, **bound.kwargs))

    return call
