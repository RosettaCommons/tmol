"""Sampling metadata preserves library handedness and retained atom identities."""

import numpy as np
import pytest

from .dunbrack_molecules import DUNBRACK_CASES
from .molecules import DOMAIN_CASES


def _prepare(backend, *, limit=1000):
    pbt = backend.pose.packed_block_types
    if backend.framework == "jax":
        from tmol_jax.pack import DunbrackChiSampler

        sampler = DunbrackChiSampler.from_database(
            backend.parameter_database, expanded_limit=limit, limit=limit
        )
        return (
            sampler,
            sampler.prepare_packed_block_types(pbt),
            _copy_plan(backend, (sampler,)),
        )
    from tmol.score.dunbrack._params import DunbrackParamResolver
    from tmol.pack.rotamer.dunbrack._dunbrack_chi_sampler import DunbrackChiSampler

    resolver = DunbrackParamResolver.from_database(
        backend.parameter_database.scoring.dun, backend.device
    )
    sampler = DunbrackChiSampler(resolver, limit, limit)
    packed = sampler.annotate_packed_block_types(pbt)
    return sampler, packed, _copy_plan(backend, (sampler,))


def _copy_plan(backend, samplers):
    pbt = backend.pose.packed_block_types
    if backend.framework == "jax":
        from tmol_jax.pack import prepare_atom_copy_plan

        return prepare_atom_copy_plan(pbt, samplers)
    from tmol.pack.rotamer._single_residue_kinforest import (
        _construct_single_residue_kinforest,
    )
    from tmol.pack.rotamer._mainchain_fingerprint import (
        _annotate_residue_type_with_sampler_fingerprints,
        _find_unique_fingerprints,
    )

    for block in pbt.active_block_types:
        _construct_single_residue_kinforest(block)
        _annotate_residue_type_with_sampler_fingerprints(block, samplers, pbt.chem_db)
    return _find_unique_fingerprints(pbt)


@pytest.mark.parametrize("case_name", (*DUNBRACK_CASES, "connected_noncanonical"))
def test_sampling_preparation_preserves_retained_atoms_and_chemical_ownership(
    backend, case_name
):
    case = {**DOMAIN_CASES, **DUNBRACK_CASES}[case_name]
    backend.molecule(case, terms=("ref",))
    sampler, packed, plan = _prepare(backend)
    pbt = backend.pose.packed_block_types
    source = backend.numpy(plan.source_atom_mapping)
    source_ids = backend.numpy(plan.source_fingerprint)
    mapping = backend.numpy(plan.atom_mapping)
    builds = backend.numpy(packed.defines_rotamers_for_bts)
    for index, block in enumerate(pbt.active_block_types):
        if not builds[index]:
            assert source_ids[index] == -1
            continue
        retained = source[index][source[index] >= 0]
        assert len(retained) > 0
        assert len(set(retained)) == len(retained)
        assert np.all(retained < block.n_atoms)
        # A copy from a residue to itself must retain exactly the same atoms,
        # including chirality and duplicate-atom identity, in source order.
        row = plan.sampler_mapping[id(sampler)]
        np.testing.assert_array_equal(
            mapping[row, source_ids[index], index], source[index]
        )
        assert not set(sampler.first_sc_atoms_for_rt(block)) & {
            block.atoms[atom].name for atom in retained
        }
        # Backbone atoms are never rebuilt by the Dunbrack sidechain sampler.
        assert set(block.properties.polymer.mainchain_atoms) <= {
            block.atoms[atom].name for atom in retained
        }
    for name in ("bbdihe_uaids", "non_dunbrack_samples", "rottable_set_for_bt"):
        value = getattr(packed, name)
        assert value.device == backend.device
    assert np.isfinite(backend.numpy(packed.non_dunbrack_samples)).all()


def test_d_libraries_are_distinct_from_l_and_ligands_remain_unassigned(backend):
    from .molecules import MolecularCase, CASES

    case = MolecularCase(
        CASES["ligand"].parameter_files,
        (("LEU", "DLEU", "PHE", "DPHE", "LG1"),),
        ((1, 1, 1, 1, 1),),
    )
    backend.molecule(case, terms=("ref",))
    _, packed, plan = _prepare(backend)
    indices = backend.numpy(packed.rottable_set_for_bt)
    assert np.all(indices[:4] >= 0)
    assert indices[0] != indices[1]
    assert indices[2] != indices[3]
    assert indices[4] == -1
    assert not backend.numpy(packed.defines_rotamers_for_bts)[4]
    assert backend.numpy(plan.source_fingerprint)[4] == -1


def test_modified_nucleotide_copy_plan_preserves_sugar_ring_and_sampler_union(backend):
    class Sampler:
        def __init__(self, rebuild_base):
            self.rebuild_base = rebuild_base

        @staticmethod
        def sampler_name():
            return "CustomNucleotide"

        @staticmethod
        def defines_rotamers_for_rt(block):
            return True

        def first_sc_atoms_for_rt(self, block):
            if self.rebuild_base:
                return ("N9" if "N9" in block.atom_to_idx else "N1",)
            return ()

    backend.molecule(DOMAIN_CASES["modified_dna"], terms=("ref",))
    samplers = Sampler(False), Sampler(True)
    plan = _copy_plan(backend, samplers)
    source = backend.numpy(plan.source_atom_mapping)
    source_ids = backend.numpy(plan.source_fingerprint)
    mapping = backend.numpy(plan.atom_mapping)
    for index, block in enumerate(backend.pose.packed_block_types.active_block_types):
        np.testing.assert_array_equal(
            np.sort(source[index][source[index] >= 0]), np.arange(block.n_atoms)
        )
        for sampler in samplers:
            row = mapping[plan.sampler_mapping[id(sampler)], source_ids[index], index]
            if sampler.rebuild_base:
                retained = {block.atoms[atom].name for atom in row if atom >= 0}
                assert {"C1'", "C2'", "C3'", "C4'", "O4'"} <= retained
                assert not set(sampler.first_sc_atoms_for_rt(block)) & retained
            else:
                np.testing.assert_array_equal(row, source[index])
