"""Coupled candidate states reproduce molecular whole-pose values and gradients."""

from contextlib import nullcontext
import attr
import numpy as np
import pytest

from tmol_core.pack.correlation import CorrelatedBlockGroup, correlation_indices
from .disulfide_molecules import DISULFIDE_CASES
from .metal_molecules import METAL_CASES
from .rotamer_execution import RotamerExecution
from .test_rotamer_constraints import constrained_rotamers


def joint_states(execution):
    b, original = execution.backend, execution.rotamers
    fields = {}
    for name in (
        "n_rots_for_pose",
        "rot_offset_for_pose",
        "n_rots_for_block",
        "rot_offset_for_block",
    ):
        fields[name] = b.numpy(getattr(original, name)) * 2
    for name in ("pose_for_rot", "block_type_ind_for_rot", "block_ind_for_rot"):
        fields[name] = np.repeat(b.numpy(getattr(original, name)), 2)
    groups = tuple(
        CorrelatedBlockGroup(p, tuple(np.flatnonzero(row)))
        for p, row in enumerate(fields["n_rots_for_block"])
    )
    fields["group_for_block"] = correlation_indices(
        fields["n_rots_for_block"].shape, groups, fields["n_rots_for_block"]
    )
    coords, offsets, atom_poses = [], [], []
    cursor = 0
    for r, (p, block) in enumerate(
        zip(b.numpy(original.pose_for_rot), b.numpy(original.block_ind_for_rot))
    ):
        start = int(original.coord_offset_for_rot[r])
        n = b.pose.block_type(p, block).n_atoms
        xyz = b.numpy(original.coords)[start : start + n]
        for state in range(2):
            offsets.append(cursor)
            cursor += n
            perturb = (
                np.random.default_rng(317 + int(block))
                .normal(0, 0.012, xyz.shape)
                .astype(xyz.dtype)
            )
            coords.append(xyz + state * perturb)
            atom_poses.extend([p] * n)
    fields.update(
        coords=np.concatenate(coords),
        coord_offset_for_rot=np.asarray(offsets, np.int32),
        pose_ind_for_atom=np.asarray(atom_poses),
    )

    def array(name, value):
        dtype = b.numpy(getattr(original, name)).dtype
        return execution.array(value.astype(dtype))

    updates = {name: array(name, value) for name, value in fields.items()}
    updates.update(
        correlated_groups=groups, max_n_rots_per_pose=2 * original.max_n_rots_per_pose
    )
    execution.rotamers = attr.evolve(original, **updates)
    if b.framework == "jax":
        from tmol_jax.pack.connection_scoring import (
            prepare_rotamer_disulfide,
            prepare_rotamer_metal,
        )
        from tmol_jax.pack.constraint_scoring import prepare_rotamer_constraints

        prepare = {
            "disulfide": prepare_rotamer_disulfide,
            "metal": prepare_rotamer_metal,
            "constraint": prepare_rotamer_constraints,
        }[execution.family]
        execution.state = prepare(b.pose, execution.rotamers, b.parameter_database)
    else:
        execution.scorer = execution.term.render_rotamer_scoring_module(
            b.pose, execution.rotamers
        )
    return execution


@pytest.mark.parametrize("family", ("disulfide", "metal", "constraint"))
def test_joint_connection_values_and_gradients_match_whole_pose(backend, family):
    with backend.module.enable_x64() if backend.framework == "jax" else nullcontext():
        if family == "constraint":
            execution, _ = constrained_rotamers(backend, "noncanonical", False)
        else:
            case = (
                DISULFIDE_CASES["ld"]
                if family == "disulfide"
                else METAL_CASES["bridging_donor"]
            )
            backend.molecule(
                case,
                terms=("disulfide" if family == "disulfide" else "metal_coordination",),
            )
            execution = RotamerExecution(backend, family=family)
        execution = joint_states(execution)
        r = execution.rotamers
        xyz = backend.numpy(r.coords).astype(np.float64)
        values, indices = map(backend.numpy, execution.scores(xyz))
        assert np.isfinite(values).all() and np.any(np.abs(values) > 1e-3)
        for state in range(2):
            selected = (indices[1] % 2 == state) & (indices[2] % 2 == state)
            weights = np.broadcast_to(selected, values.shape).astype(np.float64)
            scored, gradient, _ = execution.value_and_grad(xyz, weights)
            whole_xyz = backend.coords().astype(np.float64)
            for rot in range(state, r.n_rotamers_total, 2):
                p, block = int(r.pose_for_rot[rot]), int(r.block_ind_for_rot[rot])
                source = int(r.coord_offset_for_rot[rot])
                target = int(backend.pose.block_coord_offset[p, block])
                n = backend.pose.block_type(p, block).n_atoms
                whole_xyz[p, target : target + n] = xyz[source : source + n]
            whole, whole_grad = backend.value_and_grad(whole_xyz)
            np.testing.assert_allclose(
                scored[:, selected].sum(), whole.sum(), atol=2e-9, rtol=2e-9
            )
            for rot in range(r.n_rotamers_total):
                p, block = int(r.pose_for_rot[rot]), int(r.block_ind_for_rot[rot])
                source = int(r.coord_offset_for_rot[rot])
                target = int(backend.pose.block_coord_offset[p, block])
                n = backend.pose.block_type(p, block).n_atoms
                expected = (
                    whole_grad[p, target : target + n]
                    if rot % 2 == state
                    else np.zeros((n, 3))
                )
                np.testing.assert_allclose(
                    gradient[source : source + n], expected, atol=2e-8, rtol=2e-8
                )
