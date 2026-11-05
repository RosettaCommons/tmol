"""Framework-free packing boundaries and invalid retained-atom identities."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tmol_core.pack.fingerprint import (
    AtomFingerprint,
    MCFingerprint,
    pack_fingerprint_rows,
    prepare_packed_fingerprints,
)


def test_conflicting_retained_atom_identity_is_rejected():
    token = AtomFingerprint(0, 1, 2, 6)

    def fingerprint(atom):
        return MCFingerprint(
            np.array([atom], dtype=np.int32), (token,), (token,), {token: atom}
        )

    with pytest.raises(ValueError, match="disagree on an atom identity"):
        pack_fingerprint_rows(
            [[fingerprint(2), fingerprint(3)]], {"first": 0, "second": 1}
        )


def test_empty_copy_plan_and_framework_free_imports():
    packed = prepare_packed_fingerprints([])
    assert packed.atom_mapping.shape == (0, 0, 0, 0)
    assert packed.source_atom_mapping.shape == (0, 0)
    assert packed.source_fingerprint.shape == (0,)
    assert packed.atom_mapping.dtype == np.int32
    assert packed.sampler_mapping == {}
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import tmol_core.pack.fingerprint; "
            "assert not {'torch', 'jax', 'tmol'}.intersection(sys.modules)",
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        check=True,
    )
