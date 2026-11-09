from tmol.score.terms._term_creator import TermCreator, _score_term_creator
from tmol.score._score_types import _score_types_for_term
from tmol.database import ParameterDatabase
import torch


@_score_term_creator
class NaTorsionTermCreator(TermCreator):
    """Create the nucleic-acid torsion energy term."""

    _score_types = _score_types_for_term("na_torsion")

    @classmethod
    def create_term(cls, param_db: ParameterDatabase, device: torch.device):
        import tmol.score.na_torsion._na_torsion_energy_term

        return tmol.score.na_torsion._na_torsion_energy_term.NaTorsionEnergyTerm(
            param_db, device
        )

    @classmethod
    def score_types(cls):
        return cls._score_types
