"""Covalently coupled packing groups, motion boundaries and shared chi budgets."""

from collections import deque
from typing import Tuple
import attr
import itertools
import math
import networkx
import numpy
from tmol_core.kinematics.chemical import _block_group_kinforest_data
from tmol_core.pack.chi import budgeted_chi_samples, chi_depths


def conjugation_connections(block_types, kinematic_connections):
    """Kinematic connections excluding each type's actual polymer ports."""
    result = numpy.asarray(kinematic_connections).copy()
    for i, bt in enumerate(block_types):
        for port in (bt.down_connection_ind, bt.up_connection_ind):
            if port >= 0:
                result[i, port] = False
    return result


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ConjugatedGroup:
    """One anchor residue and the blocks bonded to its sidechain.

    ``blocks`` lists the anchor first, then the attached blocks in the order a
    breadth-first walk out from the anchor reaches them, so a block always
    follows the one it is bonded to. ``links`` gives each bond as (parent index
    within ``blocks``, connection on the parent, child index within ``blocks``,
    connection on the child) -- the form block_group_kinforest_data takes.
    It includes cycle-closing bonds, not only the breadth-first tree.
    ``external_links`` uses the same format except its third entry is a pose
    block index outside the group. These bonds constrain which parts can move.
    """

    pose: int
    blocks: Tuple[int, ...]
    links: Tuple[Tuple[int, int, int, int], ...]
    external_links: Tuple[Tuple[int, int, int, int], ...] = ()

    @property
    def anchor(self) -> int:
        return self.blocks[0]

    def __len__(self) -> int:
        return len(self.blocks)


def find_conjugated_groups(bti, irc, conj, n_conn, is_polymer):
    """Find groups from NumPy topology, preserving cycles and external links."""
    if not numpy.any(conj):
        return []
    # an anchor is a polymer residue carrying a conjugation; find the candidates
    #    with a gather rather than a scan over every block
    # Terminal patches can remove both ordinary polymer ports. Chemical
    # identity, rather than surviving connection names, defines an anchor.
    has_conj = conj.any(axis=1)
    candidate_bt = is_polymer & has_conj
    real = bti >= 0
    candidates = numpy.logical_and(real, candidate_bt[numpy.where(real, bti, 0)])

    groups = []
    for pose in range(bti.shape[0]):
        claimed = set()
        for block in numpy.nonzero(candidates[pose])[0]:
            block = int(block)
            if block in claimed:
                continue

            blocks = [block]
            links = []
            index_of = {block: 0}
            queue = deque([block])
            while queue:
                cur = queue.popleft()
                cur_bt = int(bti[pose, cur])
                for c in range(int(n_conn[cur_bt])):
                    if not conj[cur_bt, c]:
                        continue
                    partner = int(irc[pose, cur, c, 0])
                    if partner < 0 or int(bti[pose, partner]) < 0:
                        continue
                    if partner in index_of:
                        continue  # Remaining internal bonds are collected below.
                    index_of[partner] = len(blocks)
                    blocks.append(partner)
                    links.append(
                        (
                            index_of[cur],
                            c,
                            index_of[partner],
                            int(irc[pose, cur, c, 1]),
                        )
                    )
                    queue.append(partner)

            # Preserve every constraint, including non-conjugation bonds
            # between members (e.g. a disulfide or polymer bond closing a loop).
            seen = {
                tuple(sorted(((blocks[a], ac), (blocks[b], bc))))
                for a, ac, b, bc in links
            }
            external = []
            for owner, member in enumerate(blocks):
                member_bt = int(bti[pose, member])
                for conn in range(int(n_conn[member_bt])):
                    partner, partner_conn = (int(v) for v in irc[pose, member, conn])
                    if partner < 0 or int(bti[pose, partner]) < 0:
                        continue
                    if partner not in index_of:
                        external.append((owner, conn, partner, partner_conn))
                        continue
                    edge = tuple(sorted(((member, conn), (partner, partner_conn))))
                    if edge not in seen:
                        links.append((owner, conn, index_of[partner], partner_conn))
                        seen.add(edge)
            if links:
                claimed.update(blocks)
                groups.append(
                    ConjugatedGroup(
                        pose=pose,
                        blocks=tuple(blocks),
                        links=tuple(links),
                        external_links=tuple(external),
                    )
                )
    return groups


def group_atom_context(group, types):
    """Local residue types, group atom offsets and connection partners."""
    offsets = numpy.cumsum([0] + [bt.n_atoms for bt in types])
    partners = {}
    for a, ac, b, bc in group.links:
        partners[a, ac] = (b, bc)
        partners[b, bc] = (a, ac)
    return types, offsets, partners


def resolve_group_atom(uaid, owner, block_types, offsets, partners):
    """Resolve a torsion atom into the group's concatenated atom numbering."""
    atom, conn, sep = (int(v) for v in uaid)
    if conn != -1:
        partner = partners.get((owner, conn))
        if partner is None:
            return -1
        owner, conn = partner
        downstream = block_types[owner].atom_downstream_of_conn
        if not 0 <= sep < downstream.shape[1]:
            return -1
        atom = int(downstream[conn, sep])
    return int(offsets[owner]) + atom if atom >= 0 else -1


@attr.s(auto_attribs=True, frozen=True, slots=True)
class GroupSamplingTopology:
    block_types: list
    offsets: numpy.ndarray
    partners: dict
    kinforest: object
    # Unordered axis -> its child in the rooted tree, for budget depth.
    movable_axes: dict


@attr.s(auto_attribs=True, frozen=True, slots=True)
class GroupSamplingPlan:
    group: ConjugatedGroup
    topology: GroupSamplingTopology
    owners: tuple


def group_sampling_topology(group, active_types, type_indices, fixed_owners=()):
    types, offsets, partners = group_atom_context(
        group, [active_types[int(i)] for i in type_indices]
    )
    rkd, offsets = _block_group_kinforest_data(types, group.links, anchor=0)
    graph = networkx.Graph()
    graph.add_nodes_from(range(int(offsets[-1])))
    for i, bt in enumerate(types):
        graph.add_edges_from(numpy.asarray(bt.bond_indices) + offsets[i])
    for a, ac, b, bc in group.links:
        graph.add_edge(
            int(offsets[a]) + int(types[a].ordered_connection_atoms[ac]),
            int(offsets[b]) + int(types[b].ordered_connection_atoms[bc]),
        )
    fixed = numpy.zeros(int(offsets[-1]), dtype=bool)
    fixed[rkd.jump_atom] = True
    for owner in fixed_owners:
        fixed[int(offsets[owner]) : int(offsets[owner + 1])] = True
    backbone_owners = {0}
    for owner, conn, _partner, _partner_conn in group.external_links:
        atom = int(offsets[owner]) + int(types[owner].ordered_connection_atoms[conn])
        fixed[atom] = True
        fixed[list(graph[atom])] = True
        backbone_owners.add(owner)
    for owner in backbone_owners:
        bt = types[owner]
        polymer = getattr(getattr(bt, "properties", None), "polymer", None)
        mainchain = set(getattr(polymer, "mainchain_atoms", None) or ())
        for atom, definition in enumerate(bt.atoms):
            if definition.name in mainchain:
                fixed[int(offsets[owner]) + atom] = True
    for child in reversed(rkd.bfto_2_orig):
        parent = rkd.preds[child]
        if parent >= 0:
            fixed[parent] |= fixed[child]
    axes = {}
    for a, b in networkx.bridges(graph):
        child = a if rkd.preds[a] == b else b
        if not fixed[child]:
            axes[frozenset((a, b))] = int(child)
    return GroupSamplingTopology(types, offsets, partners, rkd, axes)


def sampled_group_chi(
    group,
    topology,
    expanded_limit,
    limit,
    library_size=1,
    reserve_current=False,
    exclude_axes=(),
    include_anchor=False,
    sampled_members=None,
):
    """Which chi of a group's attached blocks survive the budget.

    The whole group enumerates one product, so the budget is applied across it
    rather than to each block: seven sugars of six chi each would otherwise
    multiply out. Chi freeze from the tip of the group inward, using depth in
    the kinforest that builds the group, so the linkage torsions nearest the
    anchor -- the ones that move the most atoms -- are the last to go. Library
    axes are excluded. Additional anchor axes may be included explicitly.

    The budget counts rotamers, not conformers. Every active member costs one
    rotamer per conformer; inactive members each cost one background rotamer.
    Subtract that background cost before dividing by the active member count.

    Returns a list of (index within the group, ChiSamples) for the survivors.
    """
    block_types, offsets, partners = (
        topology.block_types,
        topology.offsets,
        topology.partners,
    )

    selected = []
    seen = set(exclude_axes)
    # Prefer the downstream owner's definition when both sides name one axis.
    for i, bt in reversed(list(enumerate(block_types))):
        if i == 0 and not include_anchor:
            continue
        for j, cs in enumerate(bt.chi_samples):
            if cs.is_proton:
                continue
            atoms = [
                resolve_group_atom(u, i, block_types, offsets, partners)
                for u in bt.torsion_to_uaids[cs.chi_dihedral][1:3]
            ]
            if min(atoms) < 0:
                raise ValueError(f"Cannot resolve group torsion {cs.chi_dihedral}")
            axis = frozenset(atoms)
            if axis in seen or axis not in topology.movable_axes:
                continue
            seen.add(axis)
            selected.append((i, j, cs, topology.movable_axes[axis]))
    selected.sort(key=lambda item: item[:2])
    entries = [item[2] for item in selected]
    owners = [item[0] for item in selected]
    depths = chi_depths(topology.kinforest, [item[3] for item in selected])
    # Only active members carry a copy of each conformer.
    n_blocks = len(group.blocks) if sampled_members is None else sampled_members
    fixed_members = len(group.blocks) - n_blocks
    expanded_limit = (expanded_limit - fixed_members) // n_blocks
    limit = (limit - fixed_members) // n_blocks
    # With no library axes, a one-state budget keeps the input itself. There
    # is no need to reserve a second identical empty-product conformer.
    if (
        reserve_current
        and library_size == 1
        and not exclude_axes
        and max(expanded_limit, limit) == 1
    ):
        return []
    reserve_current = reserve_current and bool(entries or exclude_axes)
    kept = budgeted_chi_samples(
        entries,
        depths,
        max(expanded_limit - int(reserve_current), 0),
        max(limit - int(reserve_current), 0),
        library_size=library_size,
    )

    return [(owners[index], cs) for index, cs in kept]


def group_conformer_grid(
    sampling_plan,
    anchor_entry,
    expanded_limit,
    limit,
    *,
    include_anchor=False,
    reserve_current=False,
):
    """Enumerate coupled library/sample states with shared topology and budgets."""
    group, topology = sampling_plan.group, sampling_plan.topology
    block_types, offsets, partners = (
        topology.block_types,
        topology.offsets,
        topology.partners,
    )
    anchor_cols = []
    lib = numpy.empty((1, 0), dtype=numpy.float32)
    entry = anchor_entry
    if entry is not None and entry[0].shape[0]:
        anchor_atoms, anchor_values = entry
        anchor_bt = block_types[0]
        atom_to_chi = {
            int(uaids[2][0]): (name, uaids[1][0], uaids[2][0])
            for name, uaids in anchor_bt.torsion_to_uaids.items()
            if name.startswith("chi") and uaids[1][0] >= 0 and uaids[2][0] >= 0
        }
        keep_cols = []
        for j in range(anchor_atoms.shape[1]):
            atom = int(anchor_atoms[0, j])
            if atom < 0 or atom not in atom_to_chi:
                continue
            name, ab, ac = atom_to_chi[atom]
            axis = frozenset((int(ab), int(ac)))
            if axis not in topology.movable_axes or any(
                axis == frozenset(c[2:]) for c in anchor_cols
            ):
                continue
            anchor_cols.append((0, name, ab, ac))
            keep_cols.append(j)
        if anchor_cols:
            lib = anchor_values[:, keep_cols]
            if len(keep_cols) < anchor_values.shape[1]:
                # Projecting away constrained angles can make library
                # rows identical. Preserve the first occurrence order.
                _, first = numpy.unique(lib, axis=0, return_index=True)
                lib = lib[numpy.sort(first)]

    kept = sampled_group_chi(
        group,
        topology,
        expanded_limit,
        limit,
        library_size=lib.shape[0],
        reserve_current=reserve_current,
        exclude_axes={frozenset(c[2:]) for c in anchor_cols},
        include_anchor=include_anchor,
        sampled_members=len(sampling_plan.owners),
    )
    columns, per_chi = [], []
    for owner, cs in kept:
        bt = block_types[owner]
        uaids = bt.torsion_to_uaids[cs.chi_dihedral]
        axis = [
            resolve_group_atom(u, owner, block_types, offsets, partners)
            for u in uaids[1:3]
        ]
        if min(axis) < 0:
            raise ValueError(f"Cannot resolve group torsion {cs.chi_dihedral}")
        # Central atoms use group-wide numbering, including connections.
        columns.append((owner, cs.chi_dihedral, *axis))
        values = []
        for value in cs.samples:
            values.append(value)
            for expansion in cs.expansions:
                values.extend((value - expansion, value + expansion))
        per_chi.append([math.radians(v) for v in values])
    # The empty product is one state, not an absent group: the anchor
    # can still move while all child chi are frozen.
    tree = numpy.array(list(itertools.product(*per_chi)), dtype=numpy.float32)
    conformers = numpy.concatenate(
        [numpy.repeat(lib, len(tree), axis=0), numpy.tile(tree, (len(lib), 1))],
        axis=1,
    )
    columns = anchor_cols + columns
    return columns, conformers


def append_current_conformer(conformers, current_angles):
    """Append the measured state only when every requested torsion is finite."""
    width = conformers.shape[1]
    if width == 0:
        return conformers
    current = numpy.empty((1, width), dtype=numpy.float32)
    count = 0
    for count, value in enumerate(current_angles, 1):
        if count > width:
            raise ValueError("Current group torsions exceed the conformer columns")
        if not math.isfinite(value):
            return conformers
        current[0, count - 1] = value
    if count != width:
        raise ValueError("Current group torsions do not cover the conformer columns")
    return numpy.concatenate([conformers, current], axis=0)


def registration_masks(groups, pose_shape, with_anchor):
    """Masks for group-owned sampling and exclusion of independent samplers."""
    sampled = numpy.zeros(pose_shape, dtype=bool)
    members = numpy.zeros_like(sampled)
    for group in groups:
        members[group.pose, list(group.blocks)] = True
        for block in group.blocks if with_anchor else group.blocks[1:]:
            sampled[group.pose, block] = True

    return sampled, members


def group_torsion_nodes(rot_kf, columns, n_atoms):
    """Locate each torsion's child using the actual joint tree orientation."""
    kfo_for_atom = numpy.full(n_atoms + 1, -1, dtype=numpy.int64)
    for kfo, orig in enumerate(rot_kf.id):
        kfo_for_atom[int(orig)] = kfo + 1

    parent_of = {int(k) + 1: int(p) + 1 for k, p in enumerate(rot_kf.parent)}
    nodes = []
    for owner, chi_name, atom_b, atom_c in columns:
        nb = int(kfo_for_atom[int(atom_b)])
        nc = int(kfo_for_atom[int(atom_c)])
        if parent_of.get(nc) == nb:
            node = nc
        elif parent_of.get(nb) == nc:
            node = nb
        else:
            raise ValueError(
                f"{chi_name} of group block {owner} turns a bond that this "
                "group's kinforest does not contain"
            )
        nodes.append(node)
    return numpy.asarray(nodes, dtype=numpy.int64)


def group_task_rows(pose_of, block_of, type_of, allowed, sampler_allowed):
    """Concrete original-type rows admitted by both task and sampler masks."""
    return {
        (int(pose), int(block), int(kind)): i
        for i, (pose, block, kind) in enumerate(zip(pose_of, block_of, type_of))
        if allowed[i] and sampler_allowed[pose, block]
    }


def group_sampling_plans(
    groups,
    active_types,
    block_types,
    allowed_rows=None,
    *,
    include_anchor=False,
    topology_for_group=None,
):
    """Freeze excluded members before determining each group's movable axes."""
    first_owner = int(not include_anchor)
    plans = []
    for group in groups:
        owners = tuple(
            owner
            for owner in range(first_owner, len(group))
            if allowed_rows is None
            or (
                group.pose,
                group.blocks[owner],
                int(block_types[group.pose, group.blocks[owner]]),
            )
            in allowed_rows
        )
        if owners:
            fixed = tuple(owner for owner in range(len(group)) if owner not in owners)
            plans.append(
                GroupSamplingPlan(
                    group,
                    (
                        topology_for_group(group, fixed)
                        if topology_for_group is not None
                        else group_sampling_topology(
                            group,
                            active_types,
                            block_types[group.pose, list(group.blocks)],
                            fixed,
                        )
                    ),
                    owners,
                )
            )
    return plans


def group_sample_layout(plans, conformer_counts, allowed_rows, block_types, n_rows):
    """List each active member's joint states in concrete block-type row order."""
    counts = numpy.zeros(n_rows, numpy.int32)
    emitted = []
    for gi, (plan, count) in enumerate(zip(plans, conformer_counts)):
        for owner in plan.owners:
            group = plan.group
            block = group.blocks[owner]
            gbt = allowed_rows[group.pose, block, int(block_types[group.pose, block])]
            counts[gbt] = count
            emitted.append((gbt, gi, owner, count))
    emitted.sort()
    mapping, rows = [], []
    correlated = [[] for _ in plans]
    for gbt, gi, owner, count in emitted:
        rows.append((gi, owner, gbt, len(mapping)))
        correlated[gi].append(gbt)
        mapping.extend([gbt] * count)
    return (
        counts,
        numpy.asarray(mapping, numpy.int32),
        rows,
        tuple(tuple(g) for g in correlated),
    )


def anchor_library_mask(plans, shape, sampler):
    """Select only anchors whose library chi include an unconstrained axis."""
    allowed = numpy.zeros(shape, dtype=bool)
    for plan in plans:
        topology, group = plan.topology, plan.group
        if any(
            name.startswith("chi")
            and frozenset((int(u[1][0]), int(u[2][0]))) in topology.movable_axes
            for name, u in topology.block_types[0].torsion_to_uaids.items()
        ):
            allowed[group.pose, group.anchor, sampler] = True
    return allowed


def anchor_library_rows(groups, block_types, pose_of, block_of, type_of, sample_rows):
    """Map each original anchor identity to its sampled source-library rows."""
    out = {}
    for group in groups:
        kind = int(block_types[group.pose, group.anchor])
        wanted = [
            i
            for i, identity in enumerate(zip(pose_of, block_of, type_of))
            if tuple(map(int, identity)) == (group.pose, group.anchor, kind)
        ]
        rows = numpy.flatnonzero(numpy.isin(sample_rows, wanted))
        if rows.size:
            out[group.pose, group.anchor] = rows
    return out
