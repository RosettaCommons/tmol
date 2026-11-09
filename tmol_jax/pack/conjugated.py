"""Joint covalent-group sampling through shared chemistry and native kinematics."""

from collections import OrderedDict
from copy import copy
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.chemical.packed import PackedBlockTypesData
from tmol_core.pack.chi import (
    DEFAULT_CHI_SAMPLE_EXPANDED_LIMIT,
    DEFAULT_CHI_SAMPLE_LIMIT,
)
from tmol_core.pack.conjugated import (
    conjugation_connections,
    find_conjugated_groups,
    group_sampling_plans,
    group_task_rows,
    group_sample_layout,
    group_conformer_grid,
    group_torsion_nodes,
    append_current_conformer,
    anchor_library_mask,
    anchor_library_rows,
    registration_masks,
)
from tmol_core.kinematics.scan_ordering import (
    repeat_local_kinforest,
    repeat_local_scan_plan,
)
from tmol_jax.pose.query import _torsion_requests, _measure_torsions
from .current import IncludeCurrentSampler, FallbackSampler
from .kinematics import prepare_block_group_kinematics
from .conformer_geometry import ConformerGeometry


def conjugated_groups(pose):
    """Find covalent groups from actual residue ports and pose connections."""
    pbt = pose.packed_block_types
    types = pbt.active_block_types
    return find_conjugated_groups(
        np.asarray(pose.block_type_ind),
        np.asarray(pose.inter_residue_connections),
        conjugation_connections(
            types, PackedBlockTypesData.join_kinematic_connections(types)
        ),
        np.asarray(pbt.n_conn),
        np.asarray([bt.properties.polymer.is_polymer for bt in types]),
    )


def _geometry(tree, count, device):
    nodes, scans, gens = repeat_local_scan_plan(tree, count)
    return ConformerGeometry(
        SimpleNamespace(
            total=len(tree.id) * count,
            forest=repeat_local_kinforest(tree, count).T,
            nodes=nodes,
            scans=scans,
            gens=gens,
        ),
        device,
    )


class ConjugatedChiSampler(IncludeCurrentSampler):
    """Sample linked members together, preserving cycles and fixed attachments.

    Selection is eager; group folding executes the common native CPU/CUDA
    kinematic kernels. Ring puckers and ring closure are not generated.
    """

    def __init__(
        self,
        chi_sample_expanded_limit=DEFAULT_CHI_SAMPLE_EXPANDED_LIMIT,
        chi_sample_limit=DEFAULT_CHI_SAMPLE_LIMIT,
        library_sampler=None,
        include_current=True,
    ):
        self.chi_sample_expanded_limit = chi_sample_expanded_limit
        self.chi_sample_limit = chi_sample_limit
        self.library_sampler = library_sampler
        self.include_current = include_current

    def defines_rotamers_for_rt(self, restype):
        return any(
            c.kinematic and c.name not in ("up", "down") for c in restype.connections
        )

    def group_conformers(self, pose, anchor_chi=None, budget=None, *, plans=None):
        if plans is None:
            plans = group_sampling_plans(
                conjugated_groups(pose),
                pose.packed_block_types.active_block_types,
                np.asarray(pose.block_type_ind),
                include_anchor=self.library_sampler is not None,
            )
        expanded, limit = budget or (
            self.chi_sample_expanded_limit,
            self.chi_sample_limit,
        )
        result = []
        for plan in plans:
            group = plan.group
            columns, conformers = group_conformer_grid(
                plan,
                (anchor_chi or {}).get((group.pose, group.anchor)),
                expanded,
                limit,
                include_anchor=self.library_sampler is not None,
                reserve_current=self.include_current,
            )
            if self.include_current and columns:
                conformers = append_current_conformer(
                    conformers,
                    self._angles(pose, group, columns),
                )
            result.append((group, columns, conformers))
        return result

    @staticmethod
    def _angles(pose, group, columns):
        requests = [
            request
            for owner, name, _b, _c in columns
            for request in _torsion_requests(
                pose, group.pose, group.blocks[owner], name
            )
        ]
        return _measure_torsions(pose, requests, degrees=False)

    def anchor_library_chi(self, pose, task, plans):
        if self.library_sampler is None:
            return {}
        index = task.conformer_sampler_index[id(self.library_sampler)]
        allowed = anchor_library_mask(
            plans, task.per_block_conformer_sampler_allowed.shape, index
        )
        if not allowed.any():
            return {}
        library_task = copy(task)
        library_task.chi_sample_budget = None
        library_task.per_block_conformer_sampler_allowed = jax.device_put(
            allowed, pose.device
        )
        _, choices, atoms, chi = self.library_sampler.sample_chi_for_poses(
            pose, library_task
        )
        selected = anchor_library_rows(
            [p.group for p in plans],
            np.asarray(pose.block_type_ind),
            np.asarray(task.cons_bt_pose),
            np.asarray(task.cons_bt_block),
            np.asarray(task.cons_bt_block_type),
            np.asarray(choices),
        )
        atoms, chi = np.asarray(atoms), np.asarray(chi)
        return {key: (atoms[rows], chi[rows]) for key, rows in selected.items()}

    def _group_geometry(self, pose, group, count):
        pbt = pose.packed_block_types
        types = tuple(int(pose.block_type_ind[group.pose, b]) for b in group.blocks)
        cache = getattr(pbt, "_group_geometry", None)
        if cache is None:
            pbt._group_geometry = cache = OrderedDict()
        key = (types, group.links, count)
        if key not in cache:
            tree, offsets = prepare_block_group_kinematics(
                [pbt.active_block_types[i] for i in types], group.links
            )
            cache[key] = (
                tree,
                offsets,
                _geometry(tree, 1, pose.device),
                _geometry(tree, count, pose.device),
            )
            if len(cache) > 32:
                cache.popitem(last=False)
        cache.move_to_end(key)
        return cache[key]

    def group_coordinates(self, pose, group, columns, conformers):
        """Fold the whole group and return original atom order for every state."""
        n_atoms = sum(pose.block_type(group.pose, b).n_atoms for b in group.blocks)
        chunk = max(1, 4096 // n_atoms)
        if len(conformers) > chunk:
            return jnp.concatenate(
                [
                    self.group_coordinates(
                        pose, group, columns, conformers[i : i + chunk]
                    )
                    for i in range(0, len(conformers), chunk)
                ]
            )
        if not len(conformers):
            return jnp.empty((0, n_atoms, 3), pose.coords.dtype, device=pose.device)
        tree, offsets, one, many = self._group_geometry(pose, group, len(conformers))
        coordinates = jnp.concatenate(
            [
                pose.coords[
                    group.pose,
                    int(pose.block_coord_offset[group.pose, b]) : int(
                        pose.block_coord_offset[group.pose, b]
                    )
                    + pose.block_type(group.pose, b).n_atoms,
                ]
                for b in group.blocks
            ]
        )
        current = self._angles(pose, group, columns).astype(pose.coords.dtype)
        nodes = group_torsion_nodes(tree, columns, int(offsets[-1]))
        with jax.default_device(pose.device):
            dofs = one.inverse(coordinates)
            stacked = jnp.broadcast_to(dofs, (len(conformers),) + dofs.shape)
            stacked = stacked.at[:, nodes, 3].set(
                dofs[nodes, 3] + (jnp.asarray(conformers, pose.coords.dtype) - current)
            )
            flat = jnp.concatenate((dofs[:1], stacked[:, 1:].reshape(-1, 9)))
            return many.forward(flat).reshape(len(conformers), n_atoms, 3)

    def create_samples_for_poses(self, pose, task):
        task, _rows, index = self._selection_inputs(pose, task)
        types = np.asarray(pose.block_type_ind)
        allowed = group_task_rows(
            np.asarray(task.cons_bt_pose),
            np.asarray(task.cons_bt_block),
            np.asarray(task.cons_bt_block_type),
            np.asarray(task.is_cons_bt_allowed),
            np.asarray(task.per_block_conformer_sampler_allowed[..., index]),
        )
        plans = group_sampling_plans(
            conjugated_groups(pose),
            pose.packed_block_types.active_block_types,
            types,
            allowed,
            include_anchor=self.library_sampler is not None,
        )
        groups = self.group_conformers(
            pose,
            self.anchor_library_chi(pose, task, plans),
            task.chi_sample_budget,
            plans=plans,
        )
        counts, choices, rows, correlated = group_sample_layout(
            plans,
            [len(states) for _, _, states in groups],
            allowed,
            types,
            len(task.cons_bt_pose),
        )
        folded = [self.group_coordinates(pose, *entry) for entry in groups]
        pieces = []
        for gi, owner, _gbt, _first in rows:
            offsets = plans[gi].topology.offsets
            pieces.append(
                folded[gi][:, offsets[owner] : offsets[owner + 1]].reshape(-1, 3)
            )
        with jax.default_device(pose.device):
            coordinates = (
                jnp.concatenate(pieces)
                if pieces
                else jnp.empty((0, 3), pose.coords.dtype)
            )
        return (
            jax.device_put(counts, pose.device),
            jax.device_put(choices, pose.device),
            dict(conformer_coordinates=coordinates, correlated_gbts=correlated),
        )


def add_conjugated_group_sampler(task, sampler=None, exclude=()):
    """Register joint sampling and suppress conflicting independent choices."""
    pose = task.pose_stack
    groups = conjugated_groups(pose)
    if not groups:
        return None
    if sampler is None:
        from .dunbrack import DunbrackChiSampler

        library = next(
            (s for s in task.conformer_samplers if isinstance(s, DunbrackChiSampler)),
            None,
        )
        sampler = ConjugatedChiSampler(library_sampler=library)
    sampled, members = registration_masks(
        groups, pose.block_type_ind.shape, sampler.library_sampler is not None
    )
    task.add_conformer_sampler_by_block_mask(sampler, sampled)
    for other in task.conformer_samplers:
        if other is not sampler and other not in exclude:
            mask = (
                sampled
                if type(other) in (FallbackSampler, IncludeCurrentSampler)
                else members
            )
            task.disable_sampler_by_block_mask(other, mask)
    return sampler
