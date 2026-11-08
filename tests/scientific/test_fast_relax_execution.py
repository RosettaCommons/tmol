"""The shared protocol runs actual molecular packing and minimization stages."""

import numpy as np
import pytest

from .test_complete_preset import CASES


@pytest.mark.parametrize(
    "name", ("ligand", "connected_noncanonical", "d_peptide", "mixed_batch")
)
def test_fast_relax_molecular_stages_and_per_pose_acceptance(backend, name):
    check_fast_relax_molecular_stages(backend, name)


def check_fast_relax_molecular_stages(backend, name, *, seed=73):
    backend.molecule(CASES[name], terms=())
    backend.use_beta2016()
    b, pose, score = backend, backend.pose, backend.score_function
    if b.framework == "jax":
        from tmol_jax import fast_relax, PackerPalette, ScoreType
        from tmol_jax.optimization import run_cart_min
    else:
        from tmol import fast_relax, PackerPalette, ScoreType, run_cart_min

    original = b.coords().copy()
    initial = b.scores(original)
    repulsive = float(score.get_weight(ScoreType.fa_ljrep))

    class Minimizer:
        def __init__(self):
            self.weights, self.releases = [], 0

        def __call__(self, current, sfxn, *, fold_forest, move_map, verbose):
            self.weights.append(float(sfxn.get_weight(ScoreType.fa_ljrep)))
            if b.framework == "jax":
                return run_cart_min(current, sfxn, max_iter=2)
            return run_cart_min(current, sfxn, optimizer_kwargs={"max_iter": 2})

        def release_retained_state(self):
            self.releases += 1

    minimizer = Minimizer()
    options = {"seed": seed} if b.framework == "jax" else {}
    if b.framework == "torch":
        b.module.manual_seed(seed)
    result = fast_relax(
        pose,
        score,
        PackerPalette(),
        None,
        None,
        schedule=[0.25, 1.0],
        num_repeats=1,
        min_fn=minimizer,
        **options,
    )
    np.testing.assert_allclose(minimizer.weights, [0.25 * repulsive, repulsive])
    assert minimizer.releases == 2
    np.testing.assert_array_equal(b.coords(), original)
    np.testing.assert_array_equal(
        b.numpy(result.block_type_ind), b.numpy(pose.block_type_ind)
    )
    np.testing.assert_array_equal(
        b.numpy(result.inter_residue_connections),
        b.numpy(pose.inter_residue_connections),
    )
    assert result.packed_block_types is pose.packed_block_types
    final = b.numpy(score.render_whole_pose_scoring_module(result)(result.coords))
    assert np.isfinite(final).all()
    assert np.all(final <= initial + 1e-5 * np.maximum(1, np.abs(initial)))
    assert np.any(final < initial - 1e-4)
    assert float(score.get_weight(ScoreType.fa_ljrep)) == repulsive


@pytest.mark.parametrize("grow", (False, True))
def test_relax_acceptance_preserves_each_selected_molecular_layout(backend, grow):
    from types import SimpleNamespace
    from tmol_core.relax import accept_best
    from .test_pose_imposition import imposition_fixture, impose

    rotamers, graph, choices, _ = imposition_fixture(backend, "design", design=True)
    original = backend.pose
    selected = backend.numpy(choices).copy()
    counts, starts, types = map(
        backend.numpy,
        (
            rotamers.n_rots_for_block,
            rotamers.rot_offset_for_block,
            rotamers.block_type_ind_for_rot,
        ),
    )
    natoms = backend.numpy(original.packed_block_types.n_atoms)
    for pose, row in enumerate(counts):
        molten = 0
        for block, count in enumerate(row):
            if count > 1:
                start = starts[pose, block]
                selected[pose, 0, molten] = np.argmin(
                    natoms[types[start : start + count]]
                )
                molten += 1
    choices = (
        backend.module.device_put(selected, backend.device)
        if backend.framework == "jax"
        else backend.module.as_tensor(selected, device=backend.device)
    )
    changed = impose(backend, rotamers, graph, choices)
    assert original.coords.shape[1] != changed.coords.shape[1]
    best, candidate = (original, changed) if grow else (changed, original)
    if backend.framework == "jax":
        from tmol_jax.relax.fast_relax import _JaxRelaxExecution

        execution = _JaxRelaxExecution(0)

        def array(a):
            return backend.module.device_put(np.asarray(a, np.float32), backend.device)

    else:
        from tmol.relax._fast_relax import _TorchRelaxExecution

        execution = _TorchRelaxExecution()

        def array(a):
            return backend.module.tensor(a, device=backend.device)

    score = SimpleNamespace(
        render_whole_pose_scoring_module=lambda pose: lambda xyz: array([1.0, 3.0])
    )
    snapshots = [backend.numpy(p.coords).copy() for p in (best, candidate)]
    result, values = accept_best(execution, score, best, array([2.0, 2.0]), candidate)
    np.testing.assert_array_equal(backend.numpy(values), [1, 2])
    actual = backend.numpy(result.coords)
    for row, expected in enumerate((candidate, best)):
        for field in (
            "block_type_ind",
            "block_coord_offset",
            "inter_residue_connections",
        ):
            np.testing.assert_array_equal(
                backend.numpy(getattr(result, field))[row],
                backend.numpy(getattr(expected, field))[row],
            )
        width = expected.coords.shape[1]
        np.testing.assert_array_equal(
            actual[row, :width], backend.numpy(expected.coords)[row]
        )
        np.testing.assert_array_equal(actual[row, width:], 0)
        if expected.pdb_info is not None:
            if backend.framework == "jax":
                assert result.pdb_info.res_name[row] == expected.pdb_info.res_name[row]
            else:
                for field in ("atom_occupancy", "atom_b_factor"):
                    np.testing.assert_array_equal(
                        getattr(result.pdb_info, field)[row, :width],
                        getattr(expected.pdb_info, field)[row],
                    )
    for pose, snapshot in zip((best, candidate), snapshots):
        np.testing.assert_array_equal(backend.numpy(pose.coords), snapshot)
