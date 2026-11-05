"""Chemical and scoring parameter databases used by TMol."""

from __future__ import annotations

import os

from .chemical import (  # noqa: F401
    AtomType,
    ChemicalDatabase,
    RawResidueType,
    _l_base_name,
)
from ._patched_chemdb import PatchedChemicalDatabase  # noqa: F401
from .scoring import ScoringDatabase  # noqa: F401
from .scoring._elec import PartialCharges  # noqa: F401
from .scoring._cartbonded import CartRes, ConnectionCartRes  # noqa: F401
from .scoring._mirrored_dunbrack import _with_mirrored_libraries


from tmol_core.database.context import (  # noqa: F401
    ParameterDatabase as _ParameterDatabase,
    _inject_residue_params,  # noqa: F401
)


class ParameterDatabase(_ParameterDatabase):
    """Immutable parameter context with Torch scoring-table storage."""

    __default = None
    _load_scoring = staticmethod(ScoringDatabase.from_file)
    _with_mirrored_libraries = staticmethod(_with_mirrored_libraries)

    @classmethod
    def get_default(cls) -> "ParameterDatabase":
        """Return the process-global cached parameter database (read-only)."""
        if cls.__dict__.get("_ParameterDatabase__default") is None:
            cls.__default = cls.from_file(
                os.path.join(os.path.dirname(__file__), "default")
            )
        return cls.__default


__all__ = []
