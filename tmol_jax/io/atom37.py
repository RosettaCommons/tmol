"""Atom37 routing over shared chemistry and native coordinate reconstruction."""

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.io.atom37 import (
    Atom37MappingError,
    atom37_mapping,
    required_mainchain_entries,
)


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
        observed = np.isfinite(np.asarray(canonical_coords)).all(-1)
        variable = np.zeros_like(observed)
        indices = np.asarray(self.canonical)[:, np.asarray(self.leaf)]
        variable[tuple(indices)] = True
        self.observed = jax.device_put(observed, pose.device)
        self.variable_presence = jax.device_put(variable, pose.device)
        missing = np.zeros((*types.shape, pbt.max_n_atoms), bool)
        flat = np.zeros(pose.coords.shape[:2], bool)
        self.leaf_metadata = jax.device_put(
            leaf_execution_metadata(pose, missing, flat), pose.device
        )
        self.h_distances, self.h_metadata = execution_metadata(pose, missing)

    def execution(self):
        return _replay_coordinates, dict(
            coords=self.pose.coords,
            canonical=self.canonical,
            packed=self.packed,
            leaf=self.leaf,
            observed=self.observed,
            variable_presence=self.variable_presence,
            leaf_metadata=self.leaf_metadata,
            h_distances=self.h_distances,
            h_metadata=self.h_metadata,
            metal=getattr(self.pose, "_canonical_metal_replay", None),
            context_plan=self.context_plan,
        )


def _valid_source(canonical_coords, state):
    finite = jnp.isfinite(canonical_coords).all(-1)
    return jnp.all(state["variable_presence"] | (finite == state["observed"]))


def _replay_coordinates(canonical_coords, state):
    from ._native import generate_leaf_atoms
    from ._hydrogen import complete_hydrogens

    source = canonical_coords[tuple(state["canonical"])]
    finite = jnp.isfinite(source).all(-1)
    p, b, atom, flat = state["packed"]
    # Discrete nonleaf presence belongs to preparation. Native leaf masks
    # may vary without changing residue identity or the packed atom layout.
    valid = _valid_source(canonical_coords, state)
    value = jnp.where(finite[:, None], source, state["coords"][p, flat])
    coords = state["coords"].at[p, flat].set(value)
    leaf_metadata = state["leaf_metadata"]
    missing = leaf_metadata[0].at[p, b, atom].set(state["leaf"] & ~finite)
    flat_missing = leaf_metadata[1].at[p, flat].set(state["leaf"] & ~finite)
    if state["metal"] is not None:
        from ._metal import execute_metal_placement

        coords = execute_metal_placement(coords, *state["metal"])
    if state["context_plan"]:
        from ._context import execute_context_plan

        coords = execute_context_plan(coords, state["context_plan"])
    coords = generate_leaf_atoms(coords, (missing, flat_missing, *leaf_metadata[2:]))
    h_metadata = (*state["h_metadata"][:4], missing, *state["h_metadata"][5:])
    coords = complete_hydrogens(coords, state["h_distances"], h_metadata)
    return jnp.where(valid, coords, jnp.nan)


class _PackedReplay:
    """Discrete packing supplies values; frozen and observed atoms retain AD."""

    def __init__(self, original, packed, frozen, canonical, ordering, types):
        self.original = original
        self.pose = packed
        final = _CoordinateReplay(packed, canonical, ordering, types)
        self.canonical, self.packed = final.canonical, final.packed
        rows, source, target = [], [], []
        old = original.pose
        for p, b in zip(*np.nonzero(np.asarray(frozen))):
            before, after = old.block_type(p, b), packed.block_type(p, b)
            if before.name != after.name:
                raise ValueError("input packing changed a frozen residue identity")
            count = len(before.atoms)
            rows.extend([p] * count)
            source.extend(np.asarray(old.block_coord_offset)[p, b] + np.arange(count))
            target.extend(
                np.asarray(packed.block_coord_offset)[p, b] + np.arange(count)
            )
        self.frozen_indices = jax.device_put(
            np.asarray((rows, source, target), np.int32), packed.device
        )

    def execution(self):
        return _packed_coordinates, (
            self.original.execution()[1],
            self.pose.coords,
            self.frozen_indices,
            self.canonical,
            self.packed,
        )


def _packed_coordinates(canonical, state):
    original, coords, frozen, canonical_indices, packed_indices = state
    rebuilt = _replay_coordinates(canonical, original)
    p, source, target = frozen
    coords = coords.at[p, target].set(rebuilt[p, source])
    source = canonical[tuple(canonical_indices)]
    p, _, _, target = packed_indices
    coords = coords.at[p, target].set(
        jnp.where(jnp.isfinite(source).all(-1)[:, None], source, coords[p, target])
    )
    return jnp.where(_valid_source(canonical, original), coords, jnp.nan)


def _bind_coordinates(coords, state):
    base, sources, targets, replay, operands = state
    source = coords[sources]
    valid = jnp.isfinite(source).all(-1) | jnp.isnan(source).all(-1)
    canonical = jnp.where(jnp.all(valid), base.at[targets].set(source), jnp.nan)
    return replay(canonical, operands)


class _BoundAtom37:
    """Fixed chemistry with fresh source coordinates and reusable array operands."""

    def __init__(self, builder, example, replay, outputs):
        self.builder = builder
        self.pose = replay.pose
        self.input_shape = example.shape
        self.outputs = outputs
        self.state = (*builder._execution_routing(example), *replay.execution())

    def _validate(self, coords):
        if coords.shape[0] != self.input_shape[0]:
            raise ValueError("prepared Atom37 batch size changed; prepare again")
        self.builder._validate_coords(coords)

    def _coordinate_execution(self):
        return _bind_coordinates, self.state

    def coordinates(self, coords):
        self._validate(coords)
        return _bind_coordinates(coords, self.state)

    def __call__(self, coords):
        result = self.pose.replace(coords=self.coordinates(coords))
        return (result, dict(self.outputs)) if self.outputs else result


class PreparedAtom37PoseBuilder:
    """Prepare chemistry eagerly; bind coordinates with native first derivatives.

    Call ``prepare(example_coords)`` before differentiating or jitting the
    returned callable. A prepared execution fixes nonleaf presence and chemistry;
    changing those requires another preparation. Missing leaf atoms are rebuilt
    natively. Optional packing choices are fixed during preparation; supplied
    atoms and frozen blocks retain coordinate derivatives.
    """

    def __init__(self, context, template, routing, fragment_mapping, mainchain):
        self.context = context
        self.template = template
        self.routing = routing
        self.fragment_mapping = fragment_mapping
        self.mainchain = mainchain

    def _validate_coords(self, coords):
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
        return device

    def _canonical_coords(self, coords):
        device = self._validate_coords(coords)
        tokens, slots, residues, atoms = self.routing
        template = self.template["coords"]
        base = jnp.broadcast_to(
            jax.device_put(template, device), (coords.shape[0], *template.shape[1:])
        )
        source = coords[:, tokens, slots]
        valid = jnp.isfinite(source).all(-1) | jnp.isnan(source).all(-1)
        return jnp.where(
            jnp.all(valid), base.at[:, residues, atoms].set(source), jnp.nan
        )

    def _execution_routing(self, example):
        tokens, slots, residues, atoms = self.routing
        count = example.shape[0]
        poses = np.repeat(np.arange(count, dtype=np.int32), len(tokens))
        source = (poses, np.tile(tokens, count), np.tile(slots, count))
        target = (poses, np.tile(residues, count), np.tile(atoms, count))
        template = self.template["coords"]
        base = np.broadcast_to(template, (count, *template.shape[1:]))
        return jax.device_put((base, source, target), example.device)

    def _mapped_source(self, coords):
        return coords[:, self.routing[0], self.routing[1]]

    def prepare(
        self,
        example_coords,
        *,
        opt_h=False,
        seed=None,
        return_chain_ind=False,
        return_atom_mapping=False,
        return_block_has_missing_atoms=False,
    ):
        if isinstance(example_coords, jax.core.Tracer):
            raise ValueError("prepare Atom37 chemistry outside JAX transformations")
        from .pose_stack_construction import pose_stack_from_canonical_form

        canonical = self._canonical_coords(example_coords)
        source = np.asarray(self._mapped_source(example_coords))
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
            atom_is_present=np.isfinite(np.asarray(canonical)).all(-1),
            canonical_ordering=self.context.canonical_ordering,
        )
        from .input_completion import input_packer_atoms, complete_input

        pose, details = pose_stack_from_canonical_form(
            cf,
            self.context.packed_block_types,
            compact_padding=True,
            build_missing_context=True,
            ligand_hydrogens="input",
            return_block_has_missing_atoms=True,
            return_chain_ind=return_chain_ind,
            packer_atoms=lambda rt: input_packer_atoms(self.context)(rt),
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

        packed, frozen = complete_input(
            pose,
            details["block_has_missing_atoms"],
            self.context,
            no_optH=not opt_h,
            seed=seed,
            return_frozen=True,
        )
        if packed is not pose:
            replay = _PackedReplay(
                replay,
                packed,
                frozen,
                canonical,
                self.context.canonical_ordering,
                cf["res_types"],
            )

        outputs = {}
        if return_chain_ind:
            outputs["chain_ind"] = details["chain_ind"]
        if return_block_has_missing_atoms:
            outputs["block_has_missing_atoms"] = details["block_has_missing_atoms"]
        if return_atom_mapping:
            from ._metadata import canonical_atom_mappings

            outputs.update(
                canonical_atom_mappings(packed, self.context.canonical_ordering)
            )

        return _BoundAtom37(self, example_coords, replay, outputs)

    def __call__(self, coords, *, opt_h=False, seed=None, **return_options):
        return self.prepare(coords, opt_h=opt_h, seed=seed, **return_options)(coords)


def prepare_atom37_pose_builder(structure, context):
    """Resolve shared atom identity and connectivity once for Atom37 binding."""
    from tmol_core.io.canonical_atom_array import prepare_canonical_atom_array
    from .pose_stack_from_biotite import _canonical_form_from_plan

    mapping = None
    if context.fragment_definitions:
        from tmol_core.ligand._fragmentation import _expand_fragmented_ligands

        structure, mapping = _expand_fragmented_ligands(
            structure, context.fragment_definitions
        )
    co = context.canonical_ordering
    plan = prepare_canonical_atom_array(
        structure,
        co,
        missing_density_distance_threshold=0,
        filter_missing_mainchain=False,
        cut_partners=context.cut_covalent_partners,
        chemical_database=context.parameter_database.chemical,
    )
    structure = plan.structure
    template = _canonical_form_from_plan(plan)
    template.update(
        find_additional_disulfides=False, find_additional_cyclic_closures=False
    )
    routing = atom37_mapping(
        structure, plan.valid_atom_mask, plan.valid_res_inds, plan.valid_atom_inds
    )
    return PreparedAtom37PoseBuilder(
        context, template, routing, mapping, required_mainchain_entries(structure, co)
    )


def pose_stack_from_atom37_and_topology(
    atom37_coords,
    biotite_structure,
    context,
    no_optH=True,
    *,
    packer_seed=None,
    return_context=False,
    return_chain_ind=False,
    return_atom_mapping=False,
    return_block_has_missing_atoms=False,
    **canonical_options,
):
    """Construct eagerly from shared atom identities and authoritative coordinates.

    Use ``prepare_atom37_pose_builder(...).prepare(...)`` for transformed execution
    with fixed discrete chemistry and packing choices.
    """
    supported = {
        "find_additional_disulfides",
        "find_additional_cyclic_closures",
        "find_additional_metal_coordination",
        "protonation_variants",
        "metal_sites",
        "require_all_metal_donors",
    }
    unknown = canonical_options.keys() - supported
    if unknown:
        raise TypeError(
            "unsupported Atom37 constructor options: " + ", ".join(sorted(unknown))
        )
    builder = prepare_atom37_pose_builder(biotite_structure, context)
    builder.template.update(canonical_options)
    pose = builder(
        atom37_coords,
        opt_h=not no_optH,
        seed=packer_seed,
        return_chain_ind=return_chain_ind and not return_context,
        return_atom_mapping=return_atom_mapping and not return_context,
        return_block_has_missing_atoms=return_block_has_missing_atoms
        and not return_context,
    )
    return (pose, context) if return_context else pose


class _TensorAtom37Builder(PreparedAtom37PoseBuilder):
    def __init__(self, context, template, routing, n_slots):
        super().__init__(context, template, routing, None, ())
        self.n_slots = n_slots

    def _mapped_source(self, coords):
        return coords[self.routing.source_indices]

    def _validate_coords(self, coords):
        expected = (*self.routing.canonical_shape[:2], self.n_slots, 3)
        if coords.shape != expected:
            raise ValueError(f"Atom37 coordinates must have shape {expected}")
        if coords.dtype != jnp.float32:
            raise TypeError("atom37_coords must have dtype float32")
        device = self.context.packed_block_types.n_atoms.device
        if not isinstance(coords, jax.core.Tracer) and coords.device != device:
            raise ValueError("atom37_coords and context must be on the same device")
        return device

    def _canonical_coords(self, coords):
        device = self._validate_coords(coords)
        source = self._mapped_source(coords)
        valid = jnp.isfinite(source).all(-1) | jnp.isnan(source).all(-1)
        base = jax.device_put(
            np.full((*self.routing.canonical_shape, 3), np.nan, np.float32), device
        )
        return jnp.where(
            jnp.all(valid), base.at[self.routing.target_indices].set(source), jnp.nan
        )

    def _execution_routing(self, example):
        base = np.full((*self.routing.canonical_shape, 3), np.nan, np.float32)
        return jax.device_put(
            (base, self.routing.source_indices, self.routing.target_indices),
            example.device,
        )


def prepare_atom37_tensor_builder(
    res_types,
    chain_id,
    context,
    *,
    n_slots=37,
    slot_map=None,
    disulfides=None,
    cyclic_bonds=None,
    covalent_bonds=None,
    res_not_connected=None,
):
    """Prepare static tensor topology using the common validated slot routing.

    Residue indices refer to the context's canonical ordering. The default layout
    covers standard proteins; other chemistry requires an explicit slot map.
    Call ``builder.prepare(example_coords)`` outside transformations, then reuse
    its returned callable with float32 coordinates of the same shape.
    """
    from tmol_core.io.atom37 import prepare_atom37_routing

    types, chains = np.asarray(res_types), np.asarray(chain_id)
    if chains.shape != types.shape or not np.issubdtype(chains.dtype, np.integer):
        raise ValueError(
            "chain_id must have the same shape as res_types and integer dtype"
        )
    routing = prepare_atom37_routing(
        types, slot_map, n_slots, context.canonical_ordering
    )
    template = dict(
        coords=np.full((*routing.canonical_shape, 3), np.nan, np.float32),
        res_types=types.astype(np.int32),
        chain_id=chains.astype(np.int32),
        disulfides=disulfides,
        cyclic_bonds=cyclic_bonds,
        covalent_bonds=covalent_bonds,
        # The public tensor API uses mature suppressed-port semantics.
        res_gap=(
            np.zeros((*types.shape, 2), bool)
            if res_not_connected is None
            else np.asarray(res_not_connected)
        ),
        find_additional_disulfides=False,
        find_additional_cyclic_closures=False,
    )
    return _TensorAtom37Builder(context, template, routing, n_slots)


def pose_stack_from_atom37(
    atom37_coords,
    res_types,
    chain_id,
    context,
    *,
    slot_map=None,
    disulfides=None,
    cyclic_bonds=None,
    covalent_bonds=None,
    res_not_connected=None,
    no_optH=True,
    packer_seed=None,
    return_context=False,
    return_chain_ind=False,
    return_atom_mapping=False,
    return_block_has_missing_atoms=False,
):
    """Build eagerly from residue/chain tensors and authoritative slot coordinates."""
    builder = prepare_atom37_tensor_builder(
        res_types,
        chain_id,
        context,
        n_slots=atom37_coords.shape[-2],
        slot_map=slot_map,
        disulfides=disulfides,
        cyclic_bonds=cyclic_bonds,
        covalent_bonds=covalent_bonds,
        res_not_connected=res_not_connected,
    )
    result = builder(
        atom37_coords,
        opt_h=not no_optH,
        seed=packer_seed,
        return_chain_ind=return_chain_ind and not return_context,
        return_atom_mapping=return_atom_mapping and not return_context,
        return_block_has_missing_atoms=return_block_has_missing_atoms
        and not return_context,
    )
    return (result, context) if return_context else result
