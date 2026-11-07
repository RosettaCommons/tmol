from tmol.score.terms._term_creator import TermCreator, _score_term_creator
from tmol.score._score_types import _score_types_for_term
from tmol.database import ParameterDatabase
import torch


@_score_term_creator
class DunbrackTermCreator(TermCreator):
    """Create the Dunbrack rotamer energy term."""

    _score_types = _score_types_for_term("dunbrack")

    @classmethod
    def create_term(cls, param_db: ParameterDatabase, device: torch.device):
        import tmol.score.dunbrack._dunbrack_energy_term

        return tmol.score.dunbrack._dunbrack_energy_term.DunbrackEnergyTerm(
            param_db, device
        )

    @classmethod
    def score_types(cls):
        return cls._score_types
