"""Torch-compatible enumeration of the common scientific score channels."""

from tmol.utility._auto_number import AutoNumber
from tmol_core.scoring.channels import SCORE_TYPE_NAMES, TERM_CHANNELS

# Preserve the public Enum base, name and module for existing serialized values.
ScoreType = AutoNumber(
    "ScoreType", {name: () for name in SCORE_TYPE_NAMES}, module=__name__
)
ScoreType.__doc__ = "Stable indices for energy terms in score-function weight tensors."


def _score_types_for_term(name):
    return [ScoreType[channel] for channel in TERM_CHANNELS[name]]
