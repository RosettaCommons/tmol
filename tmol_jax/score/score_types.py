"""JAX integer score identifiers over the common stable channel ABI."""

from enum import IntEnum
from tmol_core.scoring.channels import SCORE_TYPE_NAMES

ScoreType = IntEnum(
    "ScoreType",
    {name: index for index, name in enumerate(SCORE_TYPE_NAMES)},
    module=__name__,
)
ScoreType.__doc__ = "Shared score indices with the original JAX integer-enum contract."

__all__ = ["ScoreType"]
