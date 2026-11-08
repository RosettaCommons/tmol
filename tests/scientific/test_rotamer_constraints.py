"""Sparse constraints preserve physical atom order and floating parameter AD."""

from contextlib import nullcontext
import numpy as np
import pytest

from .constraint_execution import Constraints
from .dunbrack_molecules import DUNBRACK_CASES
from .molecules import CASES
from .rotamer_execution import RotamerExecution

CONSTRAINT_CASES = {
    "ligand": CASES["ligand"],
    "noncanonical": CASES["connected_noncanonical"],
    "d_peptide": DUNBRACK_CASES["d_peptide"],
    "mixed_batch": DUNBRACK_CASES["mixed_batch"],
}


def constrained_rotamers(backend, name, sampled):
    backend.molecule(CONSTRAINT_CASES[name], terms=("constraint",))
    types = backend.numpy(backend.pose.block_type_ind)
    harmonic, torsions = [], []
    for p in range(len(types)):
        blocks = np.flatnonzero(types[p] >= 0)
        first, second = int(blocks[0]), int(blocks[-1])

        def atom(block, preferred, fallback):
            bt = backend.pose.packed_block_types.active_block_types[types[p, block]]
            return [p, block, bt.atom_to_idx.get(preferred, fallback)]

        harmonic.append([atom(first, "CB", 0), atom(first, "CG", 1)])
        if second != first:
            harmonic.append([atom(first, "CB", 0), atom(second, "CB", 0)])
            # A/B/A/B is a two-block constraint; physical torsion order must
            # never be sorted along with the discrete block inventory.
            torsions.append(
                [
                    atom(first, "CB", 0),
                    atom(second, "CB", 0),
                    atom(first, "CA", 1),
                    atom(second, "C", 2),
                ]
            )
    entries = [
        (
            "harmonic",
            np.asarray(harmonic, np.int32),
            np.tile([1.3, 0.8], (len(harmonic), 1)),
        )
    ]
    if torsions:
        entries.append(
            (
                "circularharmonic",
                np.asarray(torsions, np.int32),
                np.tile([0.2, 0.9, 0.1], (len(torsions), 1)),
            )
        )
    constraints = Constraints(backend).attach(entries)
    return RotamerExecution(backend, family="constraint", sampled=sampled), constraints


@pytest.mark.parametrize("name", CONSTRAINT_CASES)
@pytest.mark.parametrize("sampled", (False, True))
def test_sparse_constraints_values_coordinate_and_parameter_gradients(
    backend, name, sampled
):
    precision = (
        backend.module.enable_x64() if backend.framework == "jax" else nullcontext()
    )
    with precision:
        e, cs = constrained_rotamers(backend, name, sampled)
        xyz = backend.numpy(e.rotamers.coords).astype(np.float64)
        params = backend.numpy(cs.constraint_params).astype(np.float64)
        values, indices = map(backend.numpy, e.scores(xyz))
        assert values.shape[0] == 1 and indices.shape == (3, values.shape[1])
        assert np.any(values > 1e-3) and np.isfinite(values).all()
        if sampled and name in ("d_peptide", "mixed_batch"):
            assert backend.numpy(e.rotamers.n_rots_for_block).max() > 1
        if not sampled:
            total = np.zeros(backend.pose.n_poses)
            np.add.at(total, indices[0], values[0])
            np.testing.assert_allclose(
                total,
                backend.scores(backend.coords().astype(np.float64)),
                rtol=2e-12,
                atol=2e-10,
            )
        weights = np.cos(np.arange(values.size).reshape(values.shape) * 0.21) + 0.4
        x, p = e.array(xyz), e.array(params)
        if backend.framework == "jax":

            def loss(x, p):
                return (e.constraint_values(x, p) * e.array(weights)).sum()

            dx, dp = backend.module.jit(backend.module.grad(loss, argnums=(0, 1)))(x, p)
        else:
            x = x.detach().requires_grad_()
            p = p.detach().requires_grad_()
            energy = (e.constraint_values(x, p) * e.array(weights)).sum()
            dx, dp = backend.module.autograd.grad(energy, (x, p))
        dx, dp = map(backend.numpy, (dx, dp))
        np.testing.assert_allclose(dx.sum(axis=0), 0, atol=2e-10)
        rng = np.random.default_rng(512)
        for coordinate in (True, False):
            direction = rng.normal(size=xyz.shape if coordinate else params.shape)
            direction /= np.linalg.norm(direction)
            analytic = np.sum((dx if coordinate else dp) * direction)
            for step in (1e-5, 2e-6):
                plus = e.constraint_values(
                    e.array(xyz + step * direction if coordinate else xyz),
                    e.array(params if coordinate else params + step * direction),
                )
                minus = e.constraint_values(
                    e.array(xyz - step * direction if coordinate else xyz),
                    e.array(params if coordinate else params - step * direction),
                )
                numeric = np.sum(
                    (backend.numpy(plus) - backend.numpy(minus)) * weights
                ) / (2 * step)
                np.testing.assert_allclose(analytic, numeric, rtol=3e-6, atol=3e-6)

        if backend.framework == "jax" and name == "mixed_batch":
            import jax
            import jax.numpy as jnp
            from tmol_jax.pack.constraint_scoring import (
                prepare_rotamer_constraints,
                sparse_constraint_scores,
            )
            from tmol_jax.pose import ConstraintSet
            from tmol_jax.score.constraint import ConstraintEnergyTerm

            b = backend
            if sampled:
                for dtype in (np.float32, np.float64):
                    with (
                        jax.enable_x64(dtype == np.float64),
                        jax.default_device(b.device),
                    ):
                        x = jnp.asarray(e.rotamers.coords, dtype)
                        p = jnp.asarray(cs.constraint_params, dtype)

                        def loss(x, p):
                            values = sparse_constraint_scores(x, e.state, p)[0]
                            return jnp.sum(
                                values * jnp.cos(jnp.arange(values.size) * 0.21)
                            )

                        value, (dx, dp) = jax.jit(jax.value_and_grad(loss, (0, 1)))(
                            x, p
                        )
                        direction = jnp.sin(jnp.arange(x.size).reshape(x.shape)).astype(
                            dtype
                        )
                        pdirection = jnp.cos(
                            jnp.arange(p.size).reshape(p.shape)
                        ).astype(dtype)
                        _, tangent = jax.jit(
                            lambda x, p: jax.jvp(loss, (x, p), (direction, pdirection))
                        )(x, p)
                        tolerance = 2e-4 if dtype == np.float32 else 2e-11
                        np.testing.assert_allclose(
                            tangent,
                            jnp.sum(dx * direction) + jnp.sum(dp * pdirection),
                            rtol=tolerance,
                            atol=tolerance,
                        )
                        bv, (bx, bp) = jax.jit(
                            jax.vmap(jax.value_and_grad(loss, (0, 1)))
                        )(
                            jnp.stack((x, x + jnp.array([2.0, -3.0, 4.0], dtype))),
                            jnp.stack((p, p)),
                        )
                        for actual, expected in ((bv, value), (bx, dx), (bp, dp)):
                            np.testing.assert_allclose(
                                actual,
                                jnp.stack((expected, expected)),
                                rtol=tolerance,
                                atol=tolerance,
                            )
                            assert actual.dtype == dtype and actual.device == b.device
                        # Native torsion provides its first coordinate derivative. Parameter
                        # curvature remains differentiable without a second native derivative.
                        h = jax.jit(jax.hessian(lambda params: loss(x, params)))(p)
                        np.testing.assert_allclose(
                            h.reshape(p.size, p.size),
                            h.reshape(p.size, p.size).T,
                            rtol=tolerance,
                            atol=tolerance,
                        )
                        assert jnp.isfinite(h).all() and jnp.any(h != 0)
            else:
                cs = ConstraintSet.create_empty(
                    b.device, b.pose.n_poses
                ).add_constraints(
                    ConstraintEnergyTerm.circularharmonic,
                    np.array([[[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 0, 1]]], np.int32),
                    e.array(np.array([[0.0, 1.0, 0.0]], np.float32)),
                )
                with pytest.raises(NotImplementedError, match="more than two blocks"):
                    prepare_rotamer_constraints(
                        b.pose.replace(constraint_set=cs), e.rotamers
                    )


def test_sparse_custom_callback_curvature_legacy_and_empty(backend):
    if backend.framework != "jax":
        pytest.skip("JAX coordinate-constraint compatibility and callback transforms")
    import jax
    import jax.numpy as jnp
    from tmol_jax.pack.constraint_scoring import (
        prepare_rotamer_constraints,
        sparse_constraint_scores,
    )
    from tmol_jax.pose import ConstraintSet

    import attrs
    from tmol_jax.score.constraint import ConstraintEnergyTerm
    from tmol_jax.score.constraint.constraint_energy_term import (
        ConstraintSet as Coordinates,
    )

    with jax.default_device(backend.device):
        b = backend.molecule(CASES["mixed_batch"], terms=("constraint",))
        e = RotamerExecution(b, family="constraint")
        values, indices = sparse_constraint_scores(e.rotamers.coords, e.state)
        assert values.shape == (1, 0) and indices.shape == (3, 0)
        np.testing.assert_array_equal(
            jax.grad(lambda x: sparse_constraint_scores(x, e.state)[0].sum())(
                e.rotamers.coords
            ),
            jnp.zeros_like(e.rotamers.coords),
        )

        def callback(atoms, params):
            return params[:, 0] * jnp.sum(atoms[:, 0] ** 2, axis=-1)

        cs = ConstraintSet.create_empty(b.device, b.pose.n_poses).add_constraints(
            callback, np.array([[[0, 0, 0]]], np.int32), jnp.array([[3.0]])
        )
        state = prepare_rotamer_constraints(
            b.pose.replace(constraint_set=cs), e.rotamers
        )

        def loss(x):
            return sparse_constraint_scores(x, state)[0].sum()

        x = e.rotamers.coords
        _, hvp = jax.jit(lambda x: jax.jvp(jax.grad(loss), (x,), (jnp.ones_like(x),)))(
            x
        )
        expected = np.zeros(x.shape, x.dtype)
        expected[0] = 6
        np.testing.assert_array_equal(hvp, expected)

        legacy = Coordinates(
            jnp.array([[2.0, 3.0, 4.0]]),
            jnp.array([[0, 0]]),
            jnp.array([2.0]),
            jnp.array([0.1]),
        )
        pose = b.pose.replace(constraint_set=legacy)
        state = prepare_rotamer_constraints(pose, e.rotamers)
        term = ConstraintEnergyTerm()
        expected = sum(term(pose, term.setup_for_pose_stack(pose)).values()).sum()
        np.testing.assert_allclose(
            sparse_constraint_scores(x, state)[0].sum(), expected
        )
        with pytest.raises(ValueError, match="prepared shape"):
            sparse_constraint_scores(x, state, jnp.zeros((2, 5)))
        # A flattened padded coordinate is not a molecular atom.
        padded = attrs.evolve(
            legacy, atom_indices=jnp.array([[1, pose.coords.shape[1] - 1]])
        )
        with pytest.raises(ValueError, match="padding"):
            prepare_rotamer_constraints(pose.replace(constraint_set=padded), e.rotamers)
