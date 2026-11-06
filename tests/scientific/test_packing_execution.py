"""Complete molecular repacking selects real conformers without changing input chemistry."""

import numpy as np
import pytest

from .test_complete_preset import CASES
from .test_molecular_sampling import sampling_task
from .test_na_sampling import na_sampling_task
from .test_current_conformers import current_samplers, ensure_fallback_sampler


@pytest.mark.parametrize(
    "name", ("ligand", "connected_noncanonical", "d_peptide", "mixed_batch", "dna")
)
def test_complete_packing_selects_conformers_and_preserves_chemistry(backend, name):
    backend.molecule(CASES[name], terms=("ref",))
    task, _, concrete = (
        na_sampling_task(backend, level=1) if name == "dna" else sampling_task(backend)
    )
    current, _ = current_samplers(backend)
    task.add_conformer_sampler(current)
    ensure_fallback_sampler(backend, task)
    if backend.framework == "jax":
        from tmol_jax.pack import build_rotamers, pack_rotamers
    else:
        from tmol.pack import build_rotamers
        from tmol import pack_rotamers
    _, rotamers = build_rotamers(
        backend.pose, concrete(task), backend.parameter_database.chemical
    )
    backend.use_beta2016()
    before = backend.coords().copy()
    energy_before = backend.scores(before)
    packed = pack_rotamers(backend.pose, backend.score_function, task, seed=17)
    np.testing.assert_array_equal(backend.coords(), before)
    np.testing.assert_array_equal(
        backend.numpy(packed.block_type_ind), backend.numpy(backend.pose.block_type_ind)
    )
    np.testing.assert_array_equal(
        backend.numpy(packed.inter_residue_connections),
        backend.numpy(backend.pose.inter_residue_connections),
    )
    assert packed.packed_block_types is backend.pose.packed_block_types
    source, offsets, counts, starts, rot_types = map(
        backend.numpy,
        (
            rotamers.coords,
            rotamers.coord_offset_for_rot,
            rotamers.n_rots_for_block,
            rotamers.rot_offset_for_block,
            rotamers.block_type_ind_for_rot,
        ),
    )
    xyz = backend.numpy(packed.coords)
    out_offsets = backend.numpy(packed.block_coord_offset)
    n_atoms = backend.numpy(packed.packed_block_types.n_atoms)
    for p, block in np.argwhere(counts > 0):
        candidates = range(starts[p, block], starts[p, block] + counts[p, block])
        n = n_atoms[rot_types[starts[p, block]]]
        selected = xyz[p, out_offsets[p, block] : out_offsets[p, block] + n]
        assert any(
            np.array_equal(selected, source[offsets[r] : offsets[r] + n])
            for r in candidates
        )
    backend.pose = packed
    if backend.framework == "torch":
        backend._render_torch()
    energy_after, gradient = backend.value_and_grad(xyz)
    assert np.isfinite(energy_after).all() and np.isfinite(gradient).all()
    # IncludeCurrent makes the original conformation an explicit candidate.
    assert np.all(energy_after <= energy_before + 3e-4 + 3e-6 * np.abs(energy_before))
