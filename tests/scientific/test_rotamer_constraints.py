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
