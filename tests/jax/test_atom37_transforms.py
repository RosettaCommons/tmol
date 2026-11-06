"""Coordinate transforms over fixed shared Atom37 chemistry."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.scientific.execution import Execution
from tests.scientific.test_atom37_input import atom37_fixture
from tmol_core.io.atom37 import Atom37MappingError


def execution():
    return Execution(
        "jax-cuda"
        if "jax-cuda" in os.environ.get("TMOL_TEST_BACKENDS", "")
        else "jax-cpu"
    )


@pytest.mark.parametrize("name", ("ligand", "hydroxyproline", "zinc"))
def test_atom37_fixed_topology_vmap_and_native_translation_gradient(name):
    b, builder, source, _, _ = atom37_fixture(execution(), name, count=1)
    x = jax.device_put(source, b.device)
    bind = builder.prepare(x)
    ordinary = bind(x)
    shifts = jax.device_put(np.array([[0, 0, 0], [1, 2, 3]], np.float32), b.device)
    coords = jax.jit(jax.vmap(lambda shift: bind(x + shift).coords))(shifts)
    real = np.asarray(ordinary.atom_is_real)
    np.testing.assert_allclose(
        np.asarray(coords[1] - coords[0])[real],
        np.broadcast_to([1, 2, 3], (real.sum(), 3)),
        rtol=2e-5,
        atol=2e-5,
    )
    assert coords.device == b.device
    derivative = jax.jit(jax.grad(lambda shift: bind(x + shift).coords.sum()))(
        shifts[0]
    )
    np.testing.assert_allclose(derivative, real.sum(), rtol=2e-5, atol=2e-5)
    # Fitted metal virtuals are included here. This first-derivative contract
    # exceeds the mature Torch path, which detaches its inferred site fit.


def test_atom37_rejects_invalid_triplets_and_prepares_missing_leaf_atoms():
    b, builder, source, ordinary, context = atom37_fixture(
        execution(), "d_peptide", count=1
    )
    x = jax.device_put(source, b.device)
    with pytest.raises(Atom37MappingError, match="wholly finite"):
        builder.prepare(x.at[0, 0, 1, 0].set(jnp.nan))
    with pytest.raises(Atom37MappingError, match="wholly finite"):
        builder.prepare(x.at[0, 0, 1, 0].set(jnp.inf))
    bind = builder.prepare(x)
    with pytest.raises(ValueError, match="batch size"):
        bind(jnp.concatenate((x, x)))
    bad = jax.jit(lambda y: bind(y).coords)(x.at[0, 0, 1, 0].set(jnp.nan))
    assert np.isnan(np.asarray(bad)).all()
    from tmol_jax import atom_array_from_pose_stack

    atoms = atom_array_from_pose_stack(ordinary)
    hydrogen = int(np.flatnonzero(atoms.element == "H")[-1])
    missing = x.at[0, hydrogen, 1].set(jnp.nan)
    rebuilt = jax.jit(lambda y: bind(y).coords)(missing)
    assert np.isfinite(np.asarray(rebuilt)).all()
    gradient = jax.jit(jax.grad(lambda y: jnp.sin(bind(y).coords).sum()))(missing)
    assert np.isfinite(np.asarray(gradient)).all()
    np.testing.assert_array_equal(gradient[0, hydrogen], 0)
    assert np.linalg.norm(np.asarray(gradient)) > 1
    # Removing a nonleaf mainchain atom changes the prepared chemistry contract.
    carbon = int(np.flatnonzero(atoms.atom_name == "CA")[0])
    with pytest.raises(Atom37MappingError, match="mainchain"):
        builder.prepare(x.at[0, carbon, 1].set(jnp.nan))
    invalid = jax.jit(lambda y: bind(y).coords)(x.at[0, carbon, 1].set(jnp.nan))
    assert np.isnan(np.asarray(invalid)).all()


@pytest.mark.parametrize("name", ("ligand", "d_peptide", "hydroxyproline"))
def test_atom37_source_energy_gradient_matches_finite_difference(name):
    b, builder, source, _, context = atom37_fixture(execution(), name, count=1)
    x = jax.device_put(source, b.device)
    bind = builder.prepare(x)
    b.pose, b.parameter_database = bind(x), context.parameter_database
    b.use_beta2016()
    with jax.enable_x64():
        scorer = b.score_function.render_whole_pose_scoring_module(b.pose)
        objective = jax.jit(lambda y: scorer(bind(y).coords.astype(jnp.float64)).sum())
        gradient = jax.jit(jax.grad(objective))(x)
        direction = np.where(
            np.isfinite(source), np.random.default_rng(413).normal(size=source.shape), 0
        ).astype(np.float32)
        direction /= np.linalg.norm(direction)
        direction = jax.device_put(direction, b.device)
        h = 3e-3
        numerical = (objective(x + h * direction) - objective(x - h * direction)) / (
            2 * h
        )
        analytic = jnp.sum(gradient * direction)
        np.testing.assert_allclose(numerical, analytic, rtol=2e-3, atol=2e-3)
