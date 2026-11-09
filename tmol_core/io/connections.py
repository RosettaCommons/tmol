"""Prepared covalent and metal attachment policies, independent of execution."""

from collections import defaultdict
import attr
import numpy
from tmol_core.chemical import _site_connections


@attr.s(auto_attribs=True, frozen=True)
class ConnectionPlan:
    block_type_rows: list[list[int]]
    shape: tuple[int, int]
    # pose, residue1, connection1, residue2, connection2
    connections: list[tuple[int, int, int, int, int]]
    warnings: tuple = ()

    @property
    def block_types(self):
        """Array view for array consumers; structural consumers use the rows."""
        return numpy.asarray(self.block_type_rows, dtype=numpy.int64).reshape(
            self.shape
        )


def _rows_and_shape(indices):
    if isinstance(indices, numpy.ndarray):
        return indices.tolist(), indices.shape
    rows = [list(row) for row in indices]
    return rows, (len(rows), len(rows[0]) if rows else 0)


def _plan(block_types, shape, connections, warnings=()):
    return ConnectionPlan(block_types, shape, connections, tuple(warnings))


def is_polymer_link(first, second):
    """Polymer bonds join an up port to a down port."""
    return {first, second} == {"up", "down"}


def prepare_covalent_connections(  # noqa: C901
    canonical_ordering,
    active_block_types,
    conjugations,
    block_type_indices,
    covalent_bonds,
):
    """Choose each instance's prepared attachment context and resolve its ports.

    Inputs are common chemistry objects and integer NumPy/sequence metadata.
    Returns owned type indices and connection rows without mutating inputs.
    """
    block_types, shape = _rows_and_shape(block_type_indices)
    if covalent_bonds is None or len(covalent_bonds) == 0:
        return _plan(block_types, shape, [])
    names_for_class = canonical_ordering.restypes_ordered_atom_names
    bonds = (
        covalent_bonds.tolist()
        if isinstance(covalent_bonds, numpy.ndarray)
        else covalent_bonds
    )

    def atom_name(bt_ind, canonical_atom):
        equiv = active_block_types[bt_ind].io_equiv_class
        return names_for_class[equiv][canonical_atom]

    def is_polymer_pair(pose_ind, res1, atom1, res2, atom2):
        endpoints = []
        for res, atom in ((res1, atom1), (res2, atom2)):
            bt = active_block_types[block_types[pose_ind][res]]
            name = atom_name(block_types[pose_ind][res], atom)
            endpoints.append(
                {
                    connection.name
                    for connection in bt.connections
                    if connection.atom == name
                }
            )
        return any(
            is_polymer_link(first, second)
            for first in endpoints[0]
            for second in endpoints[1]
        )

    # the sites each residue is attached at, from the bonds themselves
    sites, contexts = {}, {}
    for pose_ind, res1, atom1, res2, atom2 in bonds:
        if (
            block_types[pose_ind][res1] < 0
            or block_types[pose_ind][res2] < 0
            or is_polymer_pair(pose_ind, res1, atom1, res2, atom2)
        ):
            continue
        for res, atom, partner, partner_atom in (
            (res1, atom1, res2, atom2),
            (res2, atom2, res1, atom1),
        ):
            bt_ind = block_types[pose_ind][res]
            partner_ind = block_types[pose_ind][partner]
            if bt_ind < 0 or partner_ind < 0:
                continue
            sites.setdefault((pose_ind, res), set()).add(atom_name(bt_ind, atom))
            contexts.setdefault((pose_ind, res), []).append(
                (
                    atom_name(bt_ind, atom),
                    active_block_types[partner_ind].io_equiv_class,
                    atom_name(partner_ind, partner_atom),
                )
            )

    for (pose_ind, res), atoms in sites.items():
        bt_ind = block_types[pose_ind][res]
        bt = active_block_types[bt_ind]
        # Explicitly constructed types, including ligand fragments, may
        # already declare the requested site. Keep the caller's bond between
        # these block instances; equal component names do not make it a cut
        # bond handled elsewhere. Fragment expansion removes exact cut bonds.
        atoms = atoms - {
            connection.atom
            for connection in bt.connections
            if connection.name not in ("up", "down")
        }
        if not atoms:
            continue
        base = bt.name
        key = (base, tuple(sorted((False, atom) for atom in atoms)))
        conjugated = conjugations.conjugated_bt_for_base_and_atoms.get(key)
        if conjugated is None:
            raise ValueError(
                f"no residue type for {base} attached at {sorted(atoms)}; the "
                "component was prepared without a connection at every site the "
                "input bonds it through"
            )
        block_types[pose_ind][res] = conjugated

    for (pose_ind, res), context in contexts.items():
        name = active_block_types[block_types[pose_ind][res]].name
        choices = conjugations.contextual_conjugates.get(name)
        if choices is not None:
            target = choices.get(tuple(sorted(context)))
            if target is None:
                raise ValueError(
                    f"No prepared attachment context for {name}: {context}"
                )
            block_types[pose_ind][res] = target

    connections, occupied = [], {}

    def endpoint_label(pose_ind, endpoint):
        res, conn = endpoint
        bt = active_block_types[block_types[pose_ind][res]]
        site = bt.connections[conn]
        return f"residue {res} {bt.name}.{site.atom} ({site.name})"

    for pose_ind, res1, atom1, res2, atom2 in bonds:
        candidates = []
        for res, atom in ((res1, atom1), (res2, atom2)):
            bt = active_block_types[block_types[pose_ind][res]]
            name = names_for_class[bt.io_equiv_class][atom]
            choices = [
                i
                for i, connection in enumerate(bt.connections)
                if connection.atom == name
            ]
            if not choices:
                raise ValueError(
                    f"{bt.name} has no connection at {name} for a declared bond"
                )
            candidates.append((res, choices))
        polymer_pair = next(
            (
                (first, second)
                for first in candidates[0][1]
                for second in candidates[1][1]
                if {
                    active_block_types[block_types[pose_ind][res1]]
                    .connections[first]
                    .name,
                    active_block_types[block_types[pose_ind][res2]]
                    .connections[second]
                    .name,
                }
                == {"up", "down"}
            ),
            None,
        )
        selected = (
            polymer_pair
            if polymer_pair is not None
            else tuple(
                next(
                    (
                        choice
                        for choice in choices
                        if active_block_types[block_types[pose_ind][res]]
                        .connections[choice]
                        .name.startswith("conj_")
                    ),
                    choices[0],
                )
                for res, choices in candidates
            )
        )
        resolved = [(candidates[side][0], selected[side]) for side in (0, 1)]
        for endpoint, partner in (resolved, resolved[::-1]):
            previous = occupied.setdefault((pose_ind, *endpoint), partner)
            if previous != partner:
                raise ValueError(
                    f"pose {pose_ind}: {endpoint_label(pose_ind, endpoint)} has "
                    f"multiple declared partners: {endpoint_label(pose_ind, previous)} "
                    f"and {endpoint_label(pose_ind, partner)}. Each connection accepts "
                    "one partner; resolve the input bond graph."
                )
        connections.append((pose_ind, *resolved[0], *resolved[1]))
    return _plan(block_types, shape, connections)


def prepare_metal_connections(
    canonical_ordering,
    active_block_types,
    conjugations,
    block_type_indices,
    metal_connections,
):
    """Choose coordinating forms and resolve metal sites with protonation intact.

    Uncoordinatable donors retain their given state and produce diagnostics.
    The returned plan owns its arrays; source metadata is unchanged.
    """
    block_types, shape = _rows_and_shape(block_type_indices)
    if metal_connections is None or len(metal_connections) == 0:
        return _plan(block_types, shape, [])
    warnings = []
    names_for_class = canonical_ordering.restypes_ordered_atom_names

    def atom_name(pose, res, atom):
        bt = active_block_types[block_types[pose][res]]
        return names_for_class[bt.io_equiv_class][atom]

    rows = [
        (int(pose), int(metal), int(site), int(res), int(atom))
        for pose, metal, site, res, atom in metal_connections
        if block_types[pose][metal] >= 0 and block_types[pose][res] >= 0
    ]
    donated = defaultdict(list)
    for pose, _, _, res, atom in rows:
        donated[(pose, res)].append(atom_name(pose, res, atom))
    uncoordinated = set()
    for (pose, res), atoms in donated.items():
        bt_ind = block_types[pose][res]
        attached = (
            *conjugations.conjugation_atoms_for_bt[bt_ind],
            *((True, a) for a in atoms),
        )
        key = (conjugations.conjugation_base_for_bt[bt_ind], tuple(sorted(attached)))
        coordinating = conjugations.conjugated_bt_for_base_and_atoms.get(key)
        if coordinating is None:
            # 6YV5 A:45: the given protonation state wins; a neutral TYR's OH has no
            #    donor form (TYR_DEP has), so its metal bond is dropped
            warnings.append(
                f"pose {pose}: {active_block_types[bt_ind].name} {res} "
                f"cannot coordinate a metal at {sorted(atoms)} in its given "
                "protonation state; left uncoordinated"
            )
            uncoordinated.add((pose, res))
            continue
        block_types[pose][res] = coordinating
    rows = [row for row in rows if (row[0], row[3]) not in uncoordinated]

    connections, filled = [], set()
    for pose, metal, site, res, atom in rows:
        metal_bt = active_block_types[block_types[pose][metal]]
        donor_bt = active_block_types[block_types[pose][res]]
        sites = _site_connections(metal_bt)
        name = atom_name(pose, res, atom)
        if site >= len(sites):
            warnings.append(
                f"{metal_bt.name} has {len(sites)} sites; "
                f"{donor_bt.name} {name} left uncoordinated"
            )
            continue
        # an atom bridging several metals has a connection for each
        donor_conn = next(
            i
            for i, c in enumerate(donor_bt.connections)
            if c.atom == name and not c.kinematic and (pose, res, i) not in filled
        )
        filled.add((pose, res, donor_conn))
        metal_conn = metal_bt.connection_to_cidx[sites[site]]
        connections.append((pose, metal, metal_conn, res, donor_conn))

    return _plan(block_types, shape, connections, warnings)
