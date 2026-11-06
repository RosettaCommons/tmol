from tmol.score.terms import *  # noqa: F401, F403
from tmol.score import ScoreTermFactory

import pytest

from tmol import ScoreType


@pytest.mark.parametrize("score_type", [ScoreType.fa_ljrep, ScoreType.omega])
def test_score_term_factory_smoke(default_database, torch_device, score_type):
    term = ScoreTermFactory.create_term_for_score_type(
        score_type, param_db=default_database, device=torch_device
    )
    assert term


def test_score_type_preserves_legacy_enum_contract():
    import pickle
    from tmol.utility._auto_number import AutoNumber

    # Protocol-0 values written by the original public enum must still load.
    for index, name in (
        (0, "fa_ljatr"),
        (21, "ref"),
        (23, "na_torsion"),
        (25, "metal_coordination"),
        (26, "n_score_types"),
    ):
        old = f"ctmol.score._score_types\nScoreType\n(I{index}\ntR.".encode()
        assert pickle.loads(old) is ScoreType[name]
    assert issubclass(ScoreType, AutoNumber)
    assert ScoreType.fa_ljatr != 0
    assert [value.value for value in ScoreType] == list(range(27))
    for value in ScoreType:
        assert pickle.loads(pickle.dumps(value)) is value
