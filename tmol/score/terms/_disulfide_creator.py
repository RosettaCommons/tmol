from tmol.score.terms._term_creator import TermCreator, _score_term_creator
from tmol.score._score_types import _score_types_for_term
from tmol.database import ParameterDatabase
import torch


@_score_term_creator
class DisulfideTermCreator(TermCreator):
    """Create the disulfide-geometry energy term."""

    _score_types = _score_types_for_term("disulfide")

    @classmethod
    def create_term(cls, param_db: ParameterDatabase, device: torch.device):
        import tmol.score.disulfide._disulfide_energy_term

        return tmol.score.disulfide._disulfide_energy_term.DisulfideEnergyTerm(
            param_db, device
        )

    @classmethod
    def score_types(cls):
        return cls._score_types
