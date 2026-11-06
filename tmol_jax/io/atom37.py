"""Atom37 routing over shared chemistry and native coordinate reconstruction."""

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.io.atom37 import (
    Atom37MappingError,
    atom37_mapping,
    required_mainchain_entries,
)
from tmol_core.io import structure_context as common
from tmol_core.io.canonical_atom_mapping import map_atoms_to_canonical


def atom37_slot_map_for_ordering(canonical_ordering, atom_names_by_slot, device=None):
    """Return the common residue-specific slot layout on the requested device."""
    from tmol_core.io.atom37 import atom37_slot_map_for_ordering as plan
    from .structure import _device

    return jax.device_put(
        plan(canonical_ordering, atom_names_by_slot).astype(np.int32), _device(device)
    )


class _CoordinateReplay:
    def __init__(self, pose, canonical_coords, ordering, canonical_types):
        from .pose_stack_construction import _canonical_metadata
        from ._missing import leaf_execution_metadata
        from ._hydrogen import execution_metadata

        self.pose = pose
        self.context_plan = getattr(pose, "_canonical_context_replay", ())
        pbt = pose.packed_block_types
        metadata, leaves = _canonical_metadata(pbt, ordering)
        types, offsets = map(np.asarray, (pose.block_type_ind, pose.block_coord_offset))
        canonical, packed, leaf = [], [], []
        for p in range(pose.n_poses):
            source_rows = np.flatnonzero(canonical_types[p] >= 0)
            for b, r in enumerate(source_rows):
                t = types[p, b]
                for atom in range(len(pbt.active_block_types[t].atoms)):
                    index = metadata.bt_canonical_atom_ind_map[t, atom]
                    if index >= 0:
                        canonical.append((p, r, index))
                        packed.append((p, b, atom, offsets[p, b] + atom))
                        leaf.append(leaves[t, atom])
        metal = getattr(pose, "_canonical_metal_replay", None)
        if metal is not None:
            placed = {(int(p), int(atom)) for p, atom, _ in np.asarray(metal[0])}
            leaf = [
                flag and (int(row[0]), int(row[3])) not in placed
                for flag, row in zip(leaf, packed)
            ]
        self.canonical = jax.device_put(np.asarray(canonical, np.int32).T, pose.device)
        self.packed = jax.device_put(np.asarray(packed, np.int32).T, pose.device)
        self.leaf = jax.device_put(np.asarray(leaf, bool), pose.device)
        observed = np.isfinite(
            np.asarray(canonical_coords)[tuple(np.asarray(self.canonical))]
        ).all(-1)
        self.observed = jax.device_put(observed, pose.device)
        missing = np.zeros((*types.shape, pbt.max_n_atoms), bool)
        flat = np.zeros(pose.coords.shape[:2], bool)
        self.leaf_metadata = jax.device_put(
            leaf_execution_metadata(pose, missing, flat), pose.device
        )
        self.h_distances, self.h_metadata = execution_metadata(pose, missing)

    def __call__(self, canonical_coords):
        from ._native import generate_leaf_atoms
        from ._hydrogen import complete_hydrogens

        source = canonical_coords[tuple(self.canonical)]
        finite = jnp.isfinite(source).all(-1)
        p, b, atom, flat = self.packed
        # Discrete nonleaf presence belongs to preparation. Native leaf masks
        # may vary without changing residue identity or the packed atom layout.
        valid = jnp.all(self.leaf | (finite == self.observed))
        value = jnp.where(finite[:, None], source, self.pose.coords[p, flat])
        coords = self.pose.coords.at[p, flat].set(value)
        missing = self.leaf_metadata[0].at[p, b, atom].set(self.leaf & ~finite)
        flat_missing = self.leaf_metadata[1].at[p, flat].set(self.leaf & ~finite)
        metal = getattr(self.pose, "_canonical_metal_replay", None)
        if metal is not None:
            from ._metal import execute_metal_placement

            coords = execute_metal_placement(coords, *metal)
        if self.context_plan:
            from ._context import execute_context_plan

            coords = execute_context_plan(coords, self.context_plan)
        coords = generate_leaf_atoms(
            coords, (missing, flat_missing, *self.leaf_metadata[2:])
        )
        h_metadata = (*self.h_metadata[:4], missing, *self.h_metadata[5:])
        coords = complete_hydrogens(coords, self.h_distances, h_metadata)
        return self.pose.replace(coords=jnp.where(valid, coords, jnp.nan))


class PreparedAtom37PoseBuilder:
    """Prepare chemistry eagerly; bind coordinates with native first derivatives.

    Call ``prepare(example_coords)`` before differentiating or jitting the
    returned callable. A prepared execution fixes nonleaf presence and chemistry;
    changing those requires another preparation. Missing leaf atoms are rebuilt
    natively. Hydrogen optimization is not yet implemented by this adapter.
    """

    def __init__(self, context, template, routing, fragment_mapping, mainchain):
        self.context = context
        self.template = template
        self.routing = routing
        self.fragment_mapping = fragment_mapping
        self.mainchain = mainchain

    def _canonical_coords(self, coords):
        device = self.context.packed_block_types.n_atoms.device
        if coords.ndim != 4 or coords.shape[-2:] != (37, 3):
            raise ValueError("atom37_coords must have shape [n_poses, n_tokens, 37, 3]")
        if coords.dtype != jnp.float32:
            raise TypeError("atom37_coords must have dtype float32")
        if not isinstance(coords, jax.core.Tracer) and coords.device != device:
            raise ValueError("atom37_coords and context must be on the same device")
        tokens, slots, residues, atoms = self.routing
        if int(tokens.max()) >= coords.shape[1]:
            raise Atom37MappingError("token_id exceeds atom37_coords token count")
        template = self.template["coords"]
        if template.shape[0] not in (1, coords.shape[0]):
            raise ValueError("topology and atom37_coords pose counts differ")
        base = jnp.broadcast_to(
            jax.device_put(template, device), (coords.shape[0], *template.shape[1:])
        )
        source = coords[:, tokens, slots]
        valid = jnp.isfinite(source).all(-1) | jnp.isnan(source).all(-1)
        return jnp.where(
            jnp.all(valid), base.at[:, residues, atoms].set(source), jnp.nan
        )

    def prepare(self, example_coords, *, opt_h=False):
        if opt_h:
            raise NotImplementedError("Atom37 hydrogen optimization is not implemented")
        if isinstance(example_coords, jax.core.Tracer):
            raise ValueError("prepare Atom37 chemistry outside JAX transformations")
        from .pose_stack_construction import pose_stack_from_canonical_form

        canonical = self._canonical_coords(example_coords)
        source = np.asarray(example_coords)[:, self.routing[0], self.routing[1]]
        if not np.all(np.isfinite(source).all(-1) | np.isnan(source).all(-1)):
            raise Atom37MappingError("mapped triplets must be wholly finite or all NaN")
        for residue, atom, name in self.mainchain:
            if not np.isfinite(np.asarray(canonical)[:, residue, atom]).all():
                raise Atom37MappingError(f"missing required mainchain atom {name}")
        count = example_coords.shape[0]
        cf = {}
        bonds = {
            "disulfides",
            "cyclic_bonds",
            "covalent_bonds",
            "metal_coordination",
            "metal_sites",
        }
        for name, value in self.template.items():
            if (
                name in bonds
                and value is not None
                and count != self.template["coords"].shape[0]
            ):
                parts = []
                for p in range(count):
                    part = np.asarray(value).copy()
                    part[:, 0] = p
                    parts.append(part)
                cf[name] = np.concatenate(parts)
            elif (
                isinstance(value, np.ndarray)
                and value.ndim
                and value.shape[0] == self.template["coords"].shape[0]
            ):
                cf[name] = np.broadcast_to(value, (count, *value.shape[1:])).copy()
            else:
                cf[name] = value
        cf.update(
            coords=np.asarray(canonical),
            atom_present=np.isfinite(np.asarray(canonical)).all(-1),
            canonical_ordering=self.context.canonical_ordering,
        )
        pose = pose_stack_from_canonical_form(
            cf,
            self.context.packed_block_types,
            compact_padding=True,
            build_missing_context=True,
            ligand_hydrogens="input",
        )
        if self.fragment_mapping is not None:
            from tmol_jax.ligand.fragment_imposition import apply_fragment_connections

            metal = getattr(pose, "_canonical_metal_replay", None)
            context_plan = getattr(pose, "_canonical_context_replay", ())
            pose = apply_fragment_connections(pose, self.fragment_mapping)
            object.__setattr__(pose, "_canonical_metal_replay", metal)
            object.__setattr__(pose, "_canonical_context_replay", context_plan)
        replay = _CoordinateReplay(
            pose, canonical, self.context.canonical_ordering, cf["res_types"]
        )

        def bind(coords):
            if coords.shape[0] != count:
                raise ValueError("prepared Atom37 batch size changed; prepare again")
            return replay(self._canonical_coords(coords))

        return bind

    def __call__(self, coords, *, opt_h=False):
        return self.prepare(coords, opt_h=opt_h)(coords)


def prepare_atom37_pose_builder(structure, context):
    """Resolve shared atom identity and connectivity once for Atom37 binding."""
    import biotite.structure as struc
    from .pose_stack_from_biotite import canonical_form_from_biotite

    mapping = None
    if context.fragment_definitions:
        from tmol_core.ligand._fragmentation import _expand_fragmented_ligands

        structure, mapping = _expand_fragmented_ligands(
            structure, context.fragment_definitions
        )
    co = context.canonical_ordering
    structure = common._normalize_input_identifiers(structure, co.name3_aliases)
    structure = common._with_input_chemistry_normalized(
        structure, common._metal_atom_names(co=co)
    )
    structure, _ = common._filter_supported_atoms_and_connectivity(
        structure, co, filter_missing_mainchain=False
    )
    template = canonical_form_from_biotite(
        structure,
        canonical_ordering=co,
        prepare_input=True,
        missing_density_distance_threshold=0,
        filter_missing_mainchain=False,
    )
    valid, atoms, residues = map_atoms_to_canonical(
        co,
        struc.get_all_residue_positions(structure),
        structure.res_name,
        structure.atom_name,
        structure.element,
    )
    routing = atom37_mapping(structure, valid, residues, atoms)
    return PreparedAtom37PoseBuilder(
        context, template, routing, mapping, required_mainchain_entries(structure, co)
    )
