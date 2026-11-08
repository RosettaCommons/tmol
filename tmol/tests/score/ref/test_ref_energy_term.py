import torch

from tmol.score import RefEnergyTerm

from tmol.tests.score.common import UbiquitinEnergyTermTestBase


def test_smoke(default_database, torch_device: torch.device):
    ref_energy = RefEnergyTerm(param_db=default_database, device=torch_device)

    assert ref_energy.device == torch_device


def test_annotate_block_types(
    fresh_default_packed_block_types, default_database, torch_device: torch.device
):
    ref_energy = RefEnergyTerm(param_db=default_database, device=torch_device)

    pbt = fresh_default_packed_block_types
    bt_list = pbt.active_block_types

    for bt in bt_list:
        ref_energy.setup_block_type(bt)
        assert hasattr(bt, "ref_weight")
    ref_energy.setup_packed_block_types(pbt)
    assert hasattr(pbt, "ref_weights")
    ref_weights = pbt.ref_weights
    ref_energy.setup_packed_block_types(pbt)

    assert pbt.ref_weights.device == torch_device
    assert (
        pbt.ref_weights is ref_weights
    )  # Test to make sure the parameters remain the same instance


class TestRefEnergyTerm(UbiquitinEnergyTermTestBase):
    energy_term_class = RefEnergyTerm

    @classmethod
    def test_whole_pose_scoring_gradcheck(cls, ubq_pdb, default_database, torch_device):
        return super().test_whole_pose_scoring_gradcheck(
            ubq_pdb, default_database, torch_device
        )

    @classmethod
    def test_block_scoring(cls, ubq_pdb, default_database, torch_device):
        resnums = [(0, 4)]
        return super().test_block_scoring(
            ubq_pdb,
            default_database,
            torch_device,
            resnums=resnums,
            update_baseline=False,
        )

    @classmethod
    def test_block_scoring_reweighted_gradcheck(
        cls, ubq_pdb, default_database, torch_device
    ):
        resnums = [(0, 4)]
        return super().test_block_scoring_reweighted_gradcheck(
            ubq_pdb, default_database, torch_device, resnums
        )


def test_reference_accumulation_preserves_small_weights_and_padding():
    import math
    import numpy as np
    from tmol_core.scoring.ref import reference_energies

    # A small real reference survives cancellation of large custom weights;
    # padding and residue order must not alter this coordinate-independent sum.
    weights = np.array([2**24, 0.75, -(2**24), -1.25], dtype=np.float32)
    indices = np.array([[0, 1, 2, 3, -1], [2, 0, -1, 3, 1], [-1] * 5])
    block, pose = reference_energies(indices, weights)
    expected = np.array(
        [math.fsum(float(weights[i]) for i in row if i >= 0) for row in indices],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(pose, expected)
    np.testing.assert_array_equal(block[indices < 0], 0)
    assert pose.dtype == block.dtype == weights.dtype
    assert pose[0] == pose[1] == -0.5
    assert pose[2] == 0


def test_reference_accumulation_empty_and_double_precision():
    import numpy as np
    from tmol_core.scoring.ref import reference_energies

    block, pose = reference_energies(np.full((2, 3), -1), np.array([], np.float32))
    np.testing.assert_array_equal(block, np.zeros((2, 3), np.float32))
    np.testing.assert_array_equal(pose, np.zeros(2, np.float32))
    weights = np.array([1, 2**-40], dtype=np.float64)
    _, pose = reference_energies(np.array([[0, 1, -1]]), weights)
    assert pose.dtype == np.float64
    assert pose[0] == 1 + 2**-40
