"""JAX scoring with the complete declared beta2016 preset composition."""

import yaml

from .energy_term import EnergyTerm
from .score_function import ScoreFunction
from .score_types import ScoreType


def beta2016_score_function(param_db=None):
    """Load every shared beta2016 weight using the supplied parameter context.

    A term with a provably empty interaction set retains its declared zero
    channel. Missing required parameters fail during rendering before tracing.
    """
    from tmol_jax.database import ParameterDatabase, default_directory

    if param_db is None:
        param_db = ParameterDatabase.get_default()
    path = default_directory() / "score_functions/beta2016.sfxn"
    with path.open() as source:
        weights = yaml.safe_load(source)["weights"]
    score = ScoreFunction(param_db)
    for name, weight in weights.items():
        score.set_weight(ScoreType[name], weight)
    return score


__all__ = [
    "ScoreFunction",
    "ScoreType",
    "EnergyTerm",
    "beta2016_score_function",
]
