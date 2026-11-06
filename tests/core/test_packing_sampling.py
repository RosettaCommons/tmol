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
            "assert not {'torch', 'jax', 'tmol'}.intersection(sys.modules)",
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        check=True,
    )
