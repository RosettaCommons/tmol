"""Rendered scorers must not silently retain stale weights or scoring state."""

from pathlib import Path
import subprocess
from types import SimpleNamespace
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tmol_jax.score import score_function as scoring
from tmol_jax.score import ScoreFunction, ScoreType
from tmol_jax.score.energy_term import EnergyTerm


class QuadraticTerm(EnergyTerm):
    def __init__(self, database):
        self.scale = 1.0

    def score_types_provided(self):
        return (ScoreType.fa_ljatr,)

    def setup_for_pose_stack(self, pose, block_types):
        return 1.0

    def __call__(self, pose, state, coords_override=None):
        return {
            ScoreType.fa_ljatr: self.scale
            * state
            * (coords_override**2).sum(axis=(1, 2))
        }


@pytest.fixture
def scorer(monkeypatch):
    original = scoring.importlib.import_module
    monkeypatch.setattr(
        scoring.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(QuadraticTerm=QuadraticTerm)
            if name == "tmol_jax.score.fake.fake_energy_term"
            else original(name)
        ),
    )
    monkeypatch.setitem(
        scoring._TERM_FOR_TYPE, ScoreType.fa_ljatr, ("fake", "QuadraticTerm")
    )
    score = ScoreFunction(None)
    score.set_weight(ScoreType.fa_ljatr, 2.0)
    return score


def test_render_freezes_configuration(scorer):
    pose = SimpleNamespace(coords=jnp.ones((2, 3, 3)), packed_block_types=None)
    old = jax.jit(scorer.render_whole_pose_scoring_module(pose))
    scorer.set_weight(ScoreType.fa_ljatr, 3.0)
    scorer.terms[0].scale = 2.0
    new = scorer.jitted(pose)
    # Even first tracing after mutation preserves the original render snapshot.
    np.testing.assert_allclose(old(pose.coords), [18, 18])
    np.testing.assert_allclose(new(pose.coords), [54, 54])
    per_term = scorer.render_whole_pose_scoring_module(
        pose, sum_terms=False, apply_weights=False
    )
    np.testing.assert_allclose(per_term(pose.coords), [[18, 18]])
    energy, grad = scorer.jitted_value_and_grad(pose)(pose.coords)
    np.testing.assert_allclose(energy, [54, 54])
    np.testing.assert_allclose(grad, np.full((2, 3, 3), 12.0))


def test_zero_weight_and_shape_validation(scorer):
    pose = SimpleNamespace(coords=jnp.ones((1, 2, 3)), packed_block_types=None)
    scorer.set_weight(ScoreType.fa_ljatr, 0)
    fn = scorer.jitted(pose)
    np.testing.assert_array_equal(fn(pose.coords), [0])
    np.testing.assert_array_equal(jax.grad(lambda c: fn(c).sum())(pose.coords), 0)
    with pytest.raises(ValueError, match="shape"):
        fn(jnp.ones((1, 3, 3)))
    with pytest.raises(ValueError, match="finite"):
        scorer.set_weight(ScoreType.fa_ljatr, float("nan"))


def test_import_does_not_load_torch_or_enable_global_x64():
    import tmol_jax

    # Other shared tests legitimately import Torch in the parent process.
    code = """
import sys
import jax
initial_x64 = jax.config.x64_enabled
sys.path.insert(0, sys.argv[1])
import tmol_jax
from tmol_jax.score import ScoreFunction
assert "torch" not in sys.modules
assert jax.config.x64_enabled == initial_x64
"""
    subprocess.run(
        [sys.executable, "-c", code, str(Path(tmol_jax.__file__).parents[1])],
        check=True,
        capture_output=True,
        text=True,
    )


def test_reference_options_snapshot_clone_dtype_and_transform_contract():
    from tests.scientific.execution import Execution
    from tests.scientific.dunbrack_molecules import DUNBRACK_CASES

    backend = Execution("jax-cpu").molecule(DUNBRACK_CASES["d_peptide"], terms=("ref",))
    sf, pose = backend.score_function, backend.pose
    weights = {"ALA": 1.0, "LEU": 2.0, "ARG": 3.0, "PHE": 4.0}
    sf.set_options({"ref_weights": weights})
    old = sf.render_whole_pose_scoring_module(pose)
    clone = sf.clone()
    weights["ALA"] = 99.0
    sf.set_option("ref_weights", {"DALA": 8.0})
    new = sf.render_whole_pose_scoring_module(pose)
    with jax.enable_x64():
        coords = pose.coords.astype(jnp.float64)
        assert jax.jit(old)(coords).dtype == coords.dtype
        np.testing.assert_array_equal(jax.jit(old)(coords), [10.0])
        np.testing.assert_array_equal(clone.jitted(pose)(coords), [10.0])
        np.testing.assert_array_equal(jax.jit(new)(coords), [8.0])
        values, tangent = jax.jit(lambda x: jax.jvp(old, (x,), (jnp.ones_like(x),)))(
            coords
        )
        np.testing.assert_array_equal(tangent, [0.0])
        np.testing.assert_array_equal(
            jax.jit(jax.vmap(old))(jnp.stack([coords, coords + 3])), [[10.0], [10.0]]
        )
    with pytest.raises(NotImplementedError, match="unknown_option"):
        sf.set_option("unknown_option", True)


def test_rerender_keeps_topology_parameters_and_callback_identity_fresh():
    from tmol_jax.pose.constraint_set import ConstraintSet

    from tests.scientific.execution import Execution
    from tests.scientific.molecules import MolecularCase

    pose = (
        Execution("jax-cpu")
        .molecule(MolecularCase((), (("GLY",), ("GLY",)), ((1,), (1,))), terms=())
        .pose
    )
    sf = ScoreFunction(None)
    sf.set_weight(ScoreType.constraint, 1.0)

    def squared_xyz(atoms, params):
        return params[:, 0] * (atoms[:, 0] ** 2).sum(axis=-1)

    def linear_x(atoms, params):
        return params[:, 0] * atoms[:, 0, 0]

    old = []
    for atom, scale, callback in (
        (0, 2.0, squared_xyz),
        (2, 3.0, squared_xyz),
        (2, 5.0, linear_x),
    ):
        cs = ConstraintSet.create_empty(device=pose.coords.device, n_poses=2).add_constraints(
            callback,
            jnp.array([[[0, 0, atom]]], jnp.int32),
            jnp.array([[scale]], jnp.float32),
        )
        import attrs

        constrained = attrs.evolve(pose, constraint_set=cs)
        fn = sf.jitted_value_and_grad(constrained)
        expected_gradient = np.zeros(pose.coords.shape, np.float32)
        if callback is squared_xyz:
            expected_energy = scale * np.square(np.asarray(pose.coords)[0, atom]).sum()
            expected_gradient[0, atom] = 2 * scale * np.asarray(pose.coords)[0, atom]
        else:
            expected_energy = scale * float(pose.coords[0, atom, 0])
            expected_gradient[0, atom, 0] = scale
        old.append((fn, [expected_energy, 0.0], expected_gradient))
    # First execution after every mutation must retain each rendered snapshot.
    for fn, energy, gradient in old:
        actual = fn(pose.coords)
        np.testing.assert_allclose(actual[0], energy)
        np.testing.assert_allclose(actual[1], gradient)
        batched = jax.jit(jax.vmap(fn))(jnp.stack([pose.coords, pose.coords]))
        np.testing.assert_allclose(batched[0], [energy, energy])
        np.testing.assert_allclose(batched[1], [gradient, gradient])


def test_explicit_double_weight_storage_and_clone(scorer):
    score = ScoreFunction(None, weight_dtype=np.float64)
    score.set_weight(ScoreType.fa_ljatr, 0.55)
    cloned = score.clone()
    assert cloned.weight_dtype == np.dtype(np.float64)
    assert cloned.get_weight(ScoreType.fa_ljatr) == 0.55
    with jax.enable_x64():
        pose = SimpleNamespace(
            coords=jnp.ones((1, 2, 3), jnp.float64), packed_block_types=None
        )
        value, gradient = cloned.jitted_value_and_grad(pose)(pose.coords)
        np.testing.assert_allclose(value, [6 * 0.55], rtol=2e-15, atol=2e-15)
        np.testing.assert_allclose(gradient, 2 * 0.55, rtol=2e-15, atol=2e-15)
    with pytest.raises(TypeError, match="float32 or float64"):
        ScoreFunction(None, weight_dtype=np.float16)
    with pytest.raises(ValueError, match="finite"):
        scorer.set_weight(ScoreType.fa_ljatr, 1e100)
