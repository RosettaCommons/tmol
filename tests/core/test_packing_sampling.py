"""Common sampling chemistry without a tensor framework."""

from types import SimpleNamespace
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tmol_core.pack.dunbrack import (
    DunbrackSamplingMetadata,
    pack_dunbrack_sampler_metadata,
    prepare_dunbrack_sampler_residue,
)


def test_own_d_library_precedes_reference_and_nonalpha_borrowing_is_neutral():
    indices = {"ALA": 0, "DALA": 1}
    metadata = DunbrackSamplingMetadata(indices, (0, 0), (1, 1))
    indices["DALA"] = 0
    uaids = np.arange(12, dtype=np.int32).reshape(4, 3)

    def residue(name, backbone):
        return SimpleNamespace(
            base_name=name,
            dunbrack_reference="ALA",
            chi_samples=(),
            torsion_to_uaids={"phi": uaids, "psi": uaids[::-1]},
            properties=SimpleNamespace(
                polymer=SimpleNamespace(
                    is_polymer=True, polymer_type="amino_acid", backbone_type=backbone
                )
            ),
        )

    def unused_kinematics(_):
        raise AssertionError("No sampled chi need kinematic budget preparation")

    native_d = residue("DALA", "alpha_aa")
    assert metadata.library_for_type(native_d) == 1
    own = prepare_dunbrack_sampler_residue(
        native_d,
        metadata,
        expanded_limit=8,
        limit=8,
        prepare_kinematics=unused_kinematics,
    )
    assert own.rottable_set_for_bt == 1
    np.testing.assert_array_equal(own.bbdihe_uaids[0], uaids)
    borrowed = prepare_dunbrack_sampler_residue(
        residue("BETA_CUSTOM", "beta_aa"),
        metadata,
        expanded_limit=8,
        limit=8,
        prepare_kinematics=unused_kinematics,
    )
    assert borrowed.rottable_set_for_bt == 0
    np.testing.assert_array_equal(borrowed.bbdihe_uaids, -1)


def test_sampling_batch_axis_and_framework_free_import():
    from tmol_core.pack.na_sampling import pack_na_sampler_params

    na = pack_na_sampler_params([])
    assert na["chi_atom"].shape == (0, 1)
    assert na["ring"].shape == (0, 5)
    assert na["proton"].shape == (0, 1, 0)
    assert na["n_combos"].shape == (0,)
    assert na["n_chi"] == 1
    zero_samples = pack_na_sampler_params(
        [dict(ring=np.arange(5), base=0, chi1_atom=1, proton_chi=[(1, 2, ())])]
    )
    assert zero_samples["proton"].shape == (1, 0, 1)
    np.testing.assert_array_equal(zero_samples["n_combos"], [0])
    packed = pack_dunbrack_sampler_metadata([], [], [])
    assert packed.bbdihe_uaids.shape == (0, 2, 4, 3)
    assert packed.non_dunbrack_samples.shape == (0, 0, 2, 0)
    assert packed.non_dunbrack_samples.dtype == np.float32
    assert packed.defines_rotamers_for_bts.dtype == np.bool_
    with pytest.raises(ValueError, match="residue type axis"):
        pack_dunbrack_sampler_metadata([object()], [], [])
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import tmol_core.pack.dunbrack; "
            "import tmol_core.pack.sampling_request; "
            "import tmol_core.pack.na_sampling; "
            "import tmol_core.pack.pose_imposition; "
            "import tmol_core.pack.opth; "
            "assert not {'torch', 'jax', 'tmol'}.intersection(sys.modules)",
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        check=True,
    )


def test_sample_count_policy_checks_capacity_before_wrapped_totals():
    from tmol_core.pack.chi import checked_sample_count
    import pytest

    capacity = 2**31 - 1
    assert checked_sample_count(0, 0, 0, 0, 2, 3) == 0
    assert checked_sample_count(3, 6, 1, 3, 2, 3) == 6
    assert (
        checked_sample_count(1, capacity, capacity, capacity, capacity, capacity)
        == capacity
    )
    for n, total, smallest, largest in (
        (capacity + 1, 0, 0, 0),
        (2, -2, 2**63 - 1, 2**63 - 1),
        (2, -1, -1, 0),
        (2, capacity + 1, 1, capacity),
    ):
        with pytest.raises(ValueError, match="index capacity"):
            checked_sample_count(n, total, smallest, largest, 2**63, 2**63)
    with pytest.raises(ValueError, match="Sampling budget 2"):
        checked_sample_count(3, 6, 1, 3, 2, 2)
