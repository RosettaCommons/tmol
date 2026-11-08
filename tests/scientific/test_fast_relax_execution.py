"""The shared protocol runs actual molecular packing and minimization stages."""

import numpy as np
import pytest

from .test_complete_preset import CASES


def test_relax_minimizers_preserve_fixed_molecular_coordinates(backend, monkeypatch):
    from importlib import import_module

    b = backend
    b.molecule(CASES["connected_noncanonical"], terms=())
    b.use_beta2016()
    api = import_module("tmol_jax" if b.framework == "jax" else "tmol")
    pose, original = b.pose, b.coords().copy()
    mask = np.zeros(original.shape[:2], dtype=bool)
    mask = (
        b.module.device_put(mask, b.device)
        if b.framework == "jax"
        else b.module.as_tensor(mask, device=b.device)
    )
    cartesian = api.CartesianMoveMap(mask)
    kinematic = api.MoveMap.from_pose_stack(pose)
    for constructor, move_map in (
        (api.CartesianRelaxMinimizer, cartesian),
        (api.KinematicRelaxMinimizer, kinematic),
    ):
        result = constructor(move_map, cuda_graph=False)(pose, b.score_function)
        np.testing.assert_array_equal(b.numpy(result.coords), original)
        assert result.coords.dtype == pose.coords.dtype
        assert result.coords.device == pose.coords.device
        assert result.packed_block_types is pose.packed_block_types
    np.testing.assert_array_equal(b.coords(), original)
    with pytest.raises(TypeError, match="needs a CartesianMoveMap"):
        api.CartesianRelaxMinimizer(kinematic)
    with pytest.raises(TypeError, match="needs a MoveMap"):
        api.KinematicRelaxMinimizer(cartesian)
    if b.framework == "jax":
        for constructor in (api.CartesianRelaxMinimizer, api.KinematicRelaxMinimizer):
            with pytest.raises(NotImplementedError, match="graph capture"):
                constructor(cuda_graph=True)

    if b.framework == "jax":
        from tmol_jax.optimization import run_cart_relax
        from tmol_jax.score import ScoreFunction
        from .test_optimizer_contract import analytic_cartesian_fixture

        with b.module.default_device(b.device):
            pose, score_function = analytic_cartesian_fixture(
                b, monkeypatch, repulsive=True
            )
            original_weights = score_function.weights
            relaxed, result = run_cart_relax(
                pose,
                score_function,
                ramp_schedule=(0.5, 1.0),
                max_iter=3,
                return_result=True,
            )

            assert score_function.weights == original_weights
            assert len(result["stages"]) == 2
            assert result["fun"] <= result["initial_fun"] + 1e-6
            np.testing.assert_array_equal(relaxed.coords[:, 2], pose.coords[:, 2])

            score_function = ScoreFunction(None)
            for schedule in (
                (),
                (0.5,),
                (0.5, 0.25, 1.0),
                (0.0, 1.0),
                (float("nan"), 1.0),
            ):
                with pytest.raises(ValueError, match="ramp"):
                    run_cart_relax(pose, score_function, ramp_schedule=schedule)


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
            self.weights, self.rtols, self.releases = [], [], 0

        def __call__(self, current, sfxn, *, verbose=False, rtol=None):
            self.weights.append(float(sfxn.get_weight(ScoreType.fa_ljrep)))
            self.rtols.append(rtol)
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
        packer_palette=PackerPalette(),
        schedule=[
            0.25,
            {"fa_rep_pack_frac": 1.0, "fa_rep_min_frac": 1.0, "min_rtol": 5e-4},
        ],
        num_repeats=1,
        minimizer=minimizer,
        **options,
    )
    np.testing.assert_allclose(minimizer.weights, [0.25 * repulsive, repulsive])
    assert minimizer.rtols == [None, 5e-4]
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


def acceptance_poses(backend, source):
    """Keep the design workflow and the original unequal-width AGA/AKA pair."""
    from importlib import import_module
    import attrs
    from .test_pose_imposition import imposition_fixture, impose

    if source == "legacy_aga_aka":
        api = import_module("tmol_jax" if backend.framework == "jax" else "tmol")
        small, large = (
            api.extended_pose_stack_from_sequences([seq, seq], device=backend.device)
            for seq in ("AGA", "AKA")
        )
        width = small.coords.shape[1]
        combined = api.PoseStackBuilder.from_poses([small, large], backend.device)
        small, large = (
            api.PoseStackBuilder.from_poses(
                [combined.split(i), combined.split(i + 1)], backend.device
            )
            for i in (0, 2)
        )
        fields = dict(
            coords=small.coords[:, :width],
            pdb_info=attrs.evolve(
                small.pdb_info,
                **{
                    name: getattr(small.pdb_info, name)[:, :width]
                    for name in ("atom_occupancy", "atom_b_factor")
                },
            ),
        )
        small = (
            small.replace(**fields)
            if backend.framework == "jax"
            else attrs.evolve(small, **fields)
        )
        assert small.coords.shape[1] < large.coords.shape[1]
        assert small.packed_block_types is large.packed_block_types
        return small, large

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
    return original, changed


@pytest.mark.parametrize("source", ("design", "legacy_aga_aka"))
@pytest.mark.parametrize("grow", (False, True))
def test_relax_acceptance_preserves_each_selected_molecular_layout(
    backend, source, grow
):
    from functools import partial
    from types import SimpleNamespace
    import attrs

    original, changed = acceptance_poses(backend, source)
    assert original.coords.shape[1] != changed.coords.shape[1]
    best, candidate = (original, changed) if grow else (changed, original)
    # Distinct host metadata identifies the winning source for each pose row.
    alternatives = []
    for pose, b_factor in ((best, 1.0), (candidate, 2.0)):
        info = attrs.evolve(
            pose.pdb_info,
            atom_b_factor=np.full_like(pose.pdb_info.atom_b_factor, b_factor),
        )
        alternatives.append(
            pose.replace(pdb_info=info)
            if backend.framework == "jax"
            else attrs.evolve(pose, pdb_info=info)
        )
    best, candidate = alternatives
    if backend.framework == "jax":
        from tmol_core.relax import accept_best
        from tmol_jax.relax.fast_relax import _JaxRelaxExecution

        accept = partial(accept_best, _JaxRelaxExecution(0))

        def array(a):
            return backend.module.device_put(np.asarray(a, np.float32), backend.device)

    else:
        from tmol.relax._fast_relax import _accept_best

        accept = _accept_best

        def array(a):
            return backend.module.tensor(a, device=backend.device)

    def equal(actual, expected):
        assert actual.dtype == expected.dtype and actual.device == expected.device
        np.testing.assert_array_equal(backend.numpy(actual), backend.numpy(expected))

    score = SimpleNamespace(
        render_whole_pose_scoring_module=lambda pose: lambda xyz: array([1.0, 3.0])
    )
    snapshots = [backend.numpy(p.coords).copy() for p in (best, candidate)]
    result, values = accept(score, best, array([2.0, 2.0]), candidate)
    equal(values, array([1.0, 2.0]))
    fields = ["block_type_ind", "block_coord_offset", "inter_residue_connections"]
    if backend.framework == "torch":
        fields += ["block_type_ind64", "block_coord_offset64"]
    actual = backend.numpy(result.coords)
    for row, expected in enumerate((candidate, best)):
        for field in fields:
            equal(getattr(result, field)[row], getattr(expected, field)[row])
        width = expected.coords.shape[1]
        equal(result.coords[row, :width], expected.coords[row])
        np.testing.assert_array_equal(actual[row, width:], 0)
        if expected.pdb_info is not None:
            if backend.framework == "jax":
                assert result.pdb_info.res_name[row] == expected.pdb_info.res_name[row]
            for field in ("atom_occupancy", "atom_b_factor"):
                merged = getattr(result.pdb_info, field)
                wanted = getattr(expected.pdb_info, field)[row]
                assert merged.shape[1] == result.coords.shape[1]
                np.testing.assert_array_equal(merged[row, : len(wanted)], wanted)
                np.testing.assert_array_equal(merged[row, len(wanted) :], 0)
    for pose, snapshot in zip((best, candidate), snapshots):
        assert pose.coords.device == backend.device
        assert backend.numpy(pose.coords).dtype == snapshot.dtype
        np.testing.assert_array_equal(backend.numpy(pose.coords), snapshot)
