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


@pytest.mark.parametrize("name", ["ILE", "DILE", "DA"])
def test_descriptor_basis_preserves_mutable_sampler_and_chemical_sources(name):
    import attr
    import gc
    import weakref

    from tmol_core.chemical import ChemicalDatabase
    from tmol_core.chemical.restypes import refined_from_raw
    from tmol_core.pack import fingerprint as fp
    from tmol_core.pack.kinematics import annotate_residue_rotamer_kinematics

    database = ChemicalDatabase.get_default()
    residue = refined_from_raw(next(r for r in database.residues if r.name == name))

    # Fingerprints consume the chemical tree and ideal Cartesian geometry,
    # not inverse-kinematic DOFs; keep that execution dependency out of this test.
    annotate_residue_rotamer_kinematics(
        residue, lambda coords, *args: np.zeros((len(coords), 9), np.float32)
    )

    class MutableSampler:
        active = True
        roots = ()

        def sampler_name(self):
            return "mutable"

        def defines_rotamers_for_rt(self, rt):
            return self.active

        def first_sc_atoms_for_rt(self, rt):
            return self.roots

    sampler = MutableSampler()
    # Alternate retained regions, as input completion and packing do. Direct
    # uncached descriptors provide the identity/ordering/chirality reference.
    mainchain = set(residue.properties.polymer.mainchain_atoms)
    root = next(a.name for a in residue.atoms if a.name not in mainchain)
    basis = None
    for roots in ((), (root,), (), (root,)):
        sampler.roots = roots
        fp.annotate_residue_type_with_sampler_fingerprints(residue, [sampler], database)
        observed = residue.mc_fingerprints["mutable"]
        atoms, descriptors, mapping = fp.create_mainchain_fingerprint(
            residue, roots, database
        )
        np.testing.assert_array_equal(observed.mc_ats, atoms)
        assert observed.mc_at_fingerprints == descriptors
        assert observed.at_for_fingerprint == mapping
        if basis is not None:
            assert residue._mc_atom_fingerprint_basis[1] is basis
        basis = residue._mc_atom_fingerprint_basis[1]

    sampler.active = False
    fp.annotate_residue_type_with_sampler_fingerprints(residue, [sampler], database)
    assert residue.mc_fingerprints == {}
    assert residue._mc_sampler_aliases == {}
    sampler.active = True
    sampler.roots = ()
    # Distinct custom chemistry must replace the basis, even for the same rt.
    atom_type = residue.atoms[0].atom_type
    custom = attr.evolve(
        database,
        atom_types=tuple(
            attr.evolve(at, element="O") if at.name == atom_type else at
            for at in database.atom_types
        ),
    )
    fp.annotate_residue_type_with_sampler_fingerprints(residue, [sampler], custom)
    assert residue._mc_atom_fingerprint_basis[1] is not basis
    assert residue.mc_fingerprints["mutable"].mc_at_fingerprints[0].element == 8
    reference = weakref.ref(custom)
    del custom
    gc.collect()
    assert reference() is None
    fp.annotate_residue_type_with_sampler_fingerprints(residue, [sampler], database)
    assert residue._mc_atom_fingerprint_basis[1] == basis
