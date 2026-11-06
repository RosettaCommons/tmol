"""Fixed amino-acid samples preserve Gly hydrogens and mixed-system identity."""

import numpy as np

from .molecules import CASES, MolecularCase
from .test_molecular_sampling import block_mask
from .test_current_conformers import ensure_fallback_sampler


def test_fixed_amino_acid_ideal_samples_and_masks(backend):
    case = MolecularCase(
        CASES["mixed_batch"].parameter_files,
        (("ALA", "DALA", "GLY", "HYP", "LG1"),),
        ((1, 1, 1, 1, 1),),
    )
    b = backend.molecule(case, terms=("ref",))
    if b.framework == "jax":
        from tmol_jax.pack import (
            FixedAAChiSampler,
            PackerTask,
            PackerPalette,
            SetPackerTask,
            build_rotamers,
        )
    else:
        from tmol import PackerTask, PackerPalette
        from tmol.pack import FixedAAChiSampler, SetPackerTask, build_rotamers
    sampler = FixedAAChiSampler()
    task = PackerTask(b.pose, PackerPalette())
    task.add_conformer_sampler(sampler)
    ensure_fallback_sampler(b, task)
    task.restrict_to_repacking()
    fixed = SetPackerTask.from_packer_task(task)
    before = b.coords().copy()
    _, rotamers = build_rotamers(b.pose, fixed, b.parameter_database.chemical)
    counts, choices, atoms, chi = map(
        b.numpy, sampler.sample_chi_for_poses(b.pose, fixed)
    )
    blocks = b.pose.packed_block_types.active_block_types
    considered = b.numpy(fixed.cons_bt_block_type)
    allowed = b.numpy(fixed.is_cons_bt_allowed)
    for row, (type_index, enabled) in enumerate(zip(considered, allowed)):
        name = blocks[type_index].name.split(":")[0]
        assert counts[row] == int(enabled and name in ("ALA", "DALA", "GLY"))
    assert len(choices) == 3
    np.testing.assert_array_equal(choices, np.flatnonzero(counts))
    np.testing.assert_array_equal(atoms, -1)
    np.testing.assert_array_equal(chi, 0)
    # Gly's two alpha hydrogens must remain distinct after ideal reconstruction.
    xyz = b.numpy(rotamers.coords)
    for row, bt_ind in enumerate(b.numpy(rotamers.block_type_ind_for_rot)):
        bt = blocks[bt_ind]
        if bt.name.split(":")[0] == "GLY":
            offset = int(b.numpy(rotamers.coord_offset_for_rot)[row])
            hydrogens = [
                i
                for i in bt.bond_indices[
                    bt.bond_indices[:, 0] == bt.atom_to_idx["CA"], 1
                ]
                if bt.atoms[i].name not in bt.properties.polymer.mainchain_atoms
            ]
            assert len(hydrogens) == 2
            assert (
                np.linalg.norm(xyz[offset + hydrogens[0]] - xyz[offset + hydrogens[1]])
                > 1.0
            )
    np.testing.assert_array_equal(b.coords(), before)
    task.disable_packing_by_block_mask(
        block_mask(b, np.ones(task.is_real_block.shape, bool))
    )
    empty = sampler.sample_chi_for_poses(b.pose, SetPackerTask.from_packer_task(task))
    assert not b.numpy(empty[0]).any() and b.numpy(empty[1]).size == 0
