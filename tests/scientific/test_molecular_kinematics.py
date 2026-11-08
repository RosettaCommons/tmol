"""Atom identities and per-term derivatives through real molecular kinematics."""

from contextlib import nullcontext

import numpy as np
import pytest

from .kinematics import MolecularKinematicExecution, MOLECULAR_KINEMATIC_TERMS
from .molecules import BACKBONE_CASES, DOMAIN_CASES
from .dunbrack_molecules import DUNBRACK_CASES
from .na_molecules import NA_CASES
from .metal_molecules import METAL_CASES

KINEMATIC_CASES = {
    name: (case, MOLECULAR_KINEMATIC_TERMS) for name, case in DOMAIN_CASES.items()
}
KINEMATIC_CASES.update(
    backbone_variants=(BACKBONE_CASES["variants"], ("omega", "rama")),
    dunbrack_d_peptide=(
        DUNBRACK_CASES["d_peptide"],
        ("dunbrack_rot", "dunbrack_rotdev", "dunbrack_semirot"),
    ),
    dunbrack_mixed=(
        DUNBRACK_CASES["mixed_batch"],
        ("dunbrack_rot", "dunbrack_rotdev", "dunbrack_semirot"),
    ),
    nucleic_modified=(NA_CASES["methylcytosine"], ("na_torsion", "na_torsion_well")),
    coordination_bridge=(METAL_CASES["bridging_donor"], ("metal_coordination",)),
)


@pytest.mark.parametrize("case_name", KINEMATIC_CASES)
def test_molecular_coordinate_mapping_and_term_gradients(backend, case_name):
    context = (
        backend.module.enable_x64() if backend.framework == "jax" else nullcontext()
    )
    with context:
        case, terms = KINEMATIC_CASES[case_name]
        backend.molecule(case, terms=terms)
        coordinates = backend.coords().astype(np.float64)
        # Padding deliberately contains a recognizable value and must survive.
        pbt = backend.pose.packed_block_types
        kinds = backend.numpy(backend.pose.block_type_ind)
        counts = backend.numpy(pbt.n_atoms)
        offsets = backend.numpy(backend.pose.block_coord_offset)
        physical = np.zeros(coordinates.shape[:2], bool)
        for p, b in zip(*np.nonzero(kinds >= 0)):
            begin = offsets[p, b]
            physical[p, begin : begin + counts[kinds[p, b]]] = True
        coordinates[~physical] = 7.125
        execution = MolecularKinematicExecution(backend, coordinates)
        ids = execution.atom_ids
        assert ids[0] == -1
        np.testing.assert_array_equal(np.sort(ids[1:]), np.flatnonzero(physical))
        initial = execution.initial_dofs
        rebuilt = execution.evaluate(initial)
        np.testing.assert_allclose(
            rebuilt[physical], coordinates[physical], rtol=2e-5, atol=2e-5
        )
        np.testing.assert_array_equal(rebuilt[~physical], coordinates[~physical])
        values, gradient = execution.score(initial, gradient=True)
        assert np.isfinite(values).all() and np.isfinite(gradient).all()
        assert np.any(np.abs(values) > 1e-3) and np.linalg.norm(gradient) > 1e-3
        direction = np.random.default_rng(177).normal(size=initial.shape)
        direction /= np.linalg.norm(direction)
        step = 1e-5
        numeric = (
            execution.score(initial + step * direction)
            - execution.score(initial - step * direction)
        ) / (2 * step)
        analytic = np.sum(gradient * direction, axis=(1, 2))
        np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-5)

    if backend.framework == "jax" and case_name in ("mixed_batch", "ligand"):
        from .execution import Execution
        from tmol_jax import PoseStackKinematics

        jax, jnp = backend.module, backend.module.numpy
        # Explicit CPU construction must work in a CUDA lane too. Subsequent
        # transforms retain the original CPU operands and their device checks.
        with jax.default_device(backend.device):
            cpu = Execution("jax-cpu").molecule(DOMAIN_CASES[case_name], terms=("ref",))
        pose = cpu.pose
        if case_name == "mixed_batch":
            module = PoseStackKinematics(pose)
            mapping = module.bind(np.asarray(pose.coords))
            initial = mapping.initial_dofs
            ids = np.asarray(module.atom_ids)
            parent = module.forest._indices.parent
            kinds = module.forest._indices.doftype
            roots = np.flatnonzero(
                (parent == 0) & (kinds == 1) & (ids // pose.coords.shape[1] == 1)
            )
            assert len(roots) == 1
            root = roots[0]
            shift = jax.device_put(np.array([1.5, -0.25, 0.5], np.float32), pose.device)
            shifted = initial.at[root, :3].add(shift)
            batch = jnp.stack((initial, shifted))
            result = jax.jit(jax.vmap(mapping))(batch)
            assert result.device == initial.device == pose.device
            np.testing.assert_allclose(result[0], pose.coords, rtol=2e-5, atol=2e-5)
            np.testing.assert_allclose(result[1, 0], result[0, 0], atol=2e-5)
            real = np.asarray(pose.atom_is_real[1])
            np.testing.assert_allclose(
                result[1, 1, real] - result[0, 1, real],
                np.broadcast_to(shift, (real.sum(), 3)),
                atol=2e-5,
            )
            np.testing.assert_array_equal(result[1, 1, ~real], result[0, 1, ~real])
            gradient = jax.jit(jax.vmap(jax.grad(lambda x: mapping(x).sum())))(batch)
            np.testing.assert_allclose(gradient[:, root, :3], real.sum(), atol=2e-5)
            with pytest.raises(NotImplementedError, match="initializes independent"):
                jax.grad(lambda x: module.inverse(x).sum())(pose.coords)
            with pytest.raises(ValueError, match="coordinate shape"):
                module.bind(pose.coords[:, :-1])
            with jax.enable_x64(False):
                with pytest.raises(ValueError, match="enable JAX float64"):
                    module.bind(np.asarray(pose.coords, dtype=np.float64))
                with pytest.raises(ValueError, match="enable JAX float64"):
                    module.bind(kin_dtype=jnp.float64)
            with pytest.raises(TypeError, match="kin_dtype"):
                module.bind(kin_dtype=jnp.float16)
            with jax.enable_x64():
                precise = module.bind(kin_dtype=jnp.float64)
                batch = jnp.stack([precise.initial_dofs, precise.initial_dofs])
                rebuilt = jax.jit(jax.vmap(precise))(batch)
                assert rebuilt.dtype == pose.coords.dtype
                derivative = jax.jit(jax.vmap(jax.grad(lambda x: precise(x).sum())))(
                    batch
                )
                assert derivative.dtype == jnp.float64
                np.testing.assert_allclose(
                    derivative[:, root, :3], real.sum(), atol=2e-5
                )
        else:
            with pytest.raises(ValueError, match="at least one block"):
                PoseStackKinematics(pose[:0])
            module = PoseStackKinematics(pose)
            mapping = module.bind()
            before = mapping(mapping.initial_dofs)
            gradient = jax.grad(lambda x: mapping(x).sum())(mapping.initial_dofs)
            assert before.device == gradient.device == pose.device
            batched = jax.vmap(mapping)(jnp.stack((mapping.initial_dofs,) * 2))
            np.testing.assert_allclose(batched[1], before, atol=2e-5)
            # A different pose view requires its own preparation; the old map is stable.
            reordered = pose[[0, 0]]
            other = PoseStackKinematics(reordered).bind()
            np.testing.assert_allclose(
                other(other.initial_dofs)[1], before[0], atol=2e-5
            )
            np.testing.assert_array_equal(mapping(mapping.initial_dofs), before)
