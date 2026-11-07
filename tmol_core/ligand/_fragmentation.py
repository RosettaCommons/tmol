"""User-defined fragmentation of fully prepared ligands.

Fragments are specified by the integer ``tmol_fragment_id`` annotation on the
input Biotite AtomArray.  Chemistry is perceived once for the complete ligand;
the functions in this module only partition that prepared chemistry.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

import biotite.structure as struc
import numpy as np

from tmol_core.chemical.ideal_coords import full_ideal_coords as _full_ideal_coords
from tmol_core.chemical import Connection, Icoor, RawResidueType
from tmol_core.ligand._icoor_tree import _signed_dihedral_angle, _vertex_angle
from tmol_core.ligand._registry import LigandPreparation

FRAGMENT_ID_ANNOTATION = "tmol_fragment_id"
MAX_FRAGMENT_CONNECTIONS = 4
MIN_FRAGMENT_HEAVY_ATOMS = 3


@dataclass(frozen=True)
class FragmentConnection:
    """One directed side of a cut bond."""

    fragment_id: int
    partner_fragment_id: int
    connection_name: str
    partner_connection_name: str
    atom_name: str
    partner_atom_name: str
    bond_type: str
    in_ring: bool = False


@dataclass(frozen=True)
class LigandFragmentDefinition:
    """Structure-independent definition of one fragmented ligand type."""

    ligand_name: str
    atom_to_fragment: Mapping[str, int]
    fragment_ids: tuple[int, ...]
    fragment_preparations: tuple[LigandPreparation, ...]
    connections: tuple[FragmentConnection, ...]

    def fragment_name(self, fragment_id: int) -> str:
        return f"{self.ligand_name}.{fragment_id}"


@dataclass(frozen=True)
class LigandFragmentBlockMapping:
    """Map a user fragment ID onto its block in a built pose."""

    pose_index: int
    ligand_name: str
    residue_label: int
    pose_residue_label: int
    chain_label: str
    insertion_code: str
    fragment_id: int
    block_index: int
    atom_names: tuple[str, ...]

    @property
    def fragment_name(self) -> str:
        return f"{self.ligand_name}.{self.fragment_id}"


@dataclass(frozen=True)
class FragmentedLigandPoseMapping:
    """Runtime mapping and connection list for a fragmented pose."""

    blocks: tuple[LigandFragmentBlockMapping, ...]
    connection_pairs: tuple[tuple[int, str, int, str], ...]

    def split(self, pose_index: int) -> "FragmentedLigandPoseMapping":
        """Return the mapping for one pose, reindexed as pose zero."""

        return replace(
            self,
            blocks=tuple(
                replace(block, pose_index=0)
                for block in self.blocks
                if block.pose_index == pose_index
            ),
        )


def _fragment_ids_from_atom_array(atom_array: struc.AtomArray) -> np.ndarray | None:
    """Return validated fragment IDs, or ``None`` when no split is requested."""

    if FRAGMENT_ID_ANNOTATION not in atom_array.get_annotation_categories():
        return None
    raw_ids = np.asarray(getattr(atom_array, FRAGMENT_ID_ANNOTATION))
    if raw_ids.ndim != 1 or raw_ids.shape[0] != atom_array.array_length():
        raise ValueError(f"{FRAGMENT_ID_ANNOTATION} must contain one integer per atom")
    if not np.issubdtype(raw_ids.dtype, np.integer):
        raise ValueError(f"{FRAGMENT_ID_ANNOTATION} values must be integers")
    fragment_ids = raw_ids.astype(np.int64)
    if np.any(fragment_ids < 0):
        raise ValueError(f"{FRAGMENT_ID_ANNOTATION} values must be non-negative")
    if np.unique(fragment_ids).size <= 1:
        return None
    return fragment_ids


def _bond_type_name(bond: tuple) -> str:
    if len(bond) <= 2:
        return "SINGLE"
    value = bond[2]
    if isinstance(value, str):
        return value.upper()
    if hasattr(value, "name"):
        return str(value.name)
    return {1: "SINGLE", 2: "DOUBLE", 3: "TRIPLE", 4: "AROMATIC"}.get(
        int(value), "SINGLE"
    )


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b, in radians; zero where the points coincide."""
    if float(np.linalg.norm(a - b) * np.linalg.norm(c - b)) < 1e-12:
        return 0.0
    return _vertex_angle(a, b, c)


def _fragment_atom_tree(  # noqa: C901
    atom_names: Sequence[str],
    bonds: Sequence[tuple],
    coords: Mapping[str, np.ndarray],
) -> tuple[list[str], dict[str, str], dict[str, tuple[str, str]]]:
    adjacency = {name: [] for name in atom_names}
    for a, b, *_ in bonds:
        adjacency[a].append(b)
        adjacency[b].append(a)

    root = max(
        atom_names, key=lambda name: (len(adjacency[name]), -atom_names.index(name))
    )
    order: list[str] = []
    parent = {root: root}
    queue = deque([root])
    while queue:
        current = queue.popleft()
        order.append(current)
        for neighbor in sorted(adjacency[current]):
            if neighbor in parent:
                continue
            parent[neighbor] = current
            queue.append(neighbor)
    if len(order) != len(atom_names):
        raise ValueError("fragment atoms must form one connected component")

    position = {name: i for i, name in enumerate(order)}
    grandparents: dict[str, tuple[str, str]] = {}
    for name in order:
        par = parent[name]
        gp = parent.get(par, par)
        if name != root and gp == par:
            alternatives = [
                n
                for n in adjacency[par]
                if n != name and position.get(n, len(order)) < position[name]
            ]
            if alternatives:
                gp = alternatives[0]
        ggp = parent.get(gp, gp)
        if name != root and ggp in (name, par, gp):
            alternatives = [
                n
                for n in (*adjacency[par], *adjacency.get(gp, []))
                if n not in (name, par, gp)
                and position.get(n, len(order)) < position[name]
            ]
            if alternatives:
                ggp = alternatives[0]
        grandparents[name] = (gp, ggp)
    return order, parent, grandparents


def _atom_icoors(
    order: Sequence[str],
    parent: Mapping[str, str],
    grandparents: Mapping[str, tuple[str, str]],
    coords: Mapping[str, np.ndarray],
) -> list[Icoor]:
    result: list[Icoor] = []
    for i, name in enumerate(order):
        par = parent[name]
        gp, ggp = grandparents[name]
        if i == 0:
            phi = theta = d = 0.0
        elif i == 1:
            phi, theta = 0.0, math.pi
            d = float(np.linalg.norm(coords[name] - coords[par]))
        elif i == 2:
            phi = 0.0
            theta = math.pi - _angle(coords[name], coords[par], coords[gp])
            d = float(np.linalg.norm(coords[name] - coords[par]))
        else:
            phi = -_signed_dihedral_angle(
                coords[name], coords[par], coords[gp], coords[ggp]
            )
            theta = math.pi - _angle(coords[name], coords[par], coords[gp])
            d = float(np.linalg.norm(coords[name] - coords[par]))
        result.append(
            Icoor(
                name=name,
                phi=phi,
                theta=theta,
                d=d,
                parent=par,
                grand_parent=gp,
                great_grand_parent=ggp,
            )
        )
    return result


def _connection_icoor(
    connection: FragmentConnection,
    local_atoms: Sequence[str],
    local_bonds: Sequence[tuple],
    coords: Mapping[str, np.ndarray],
) -> Icoor:
    adjacency = {name: [] for name in local_atoms}
    for a, b, *_ in local_bonds:
        adjacency[a].append(b)
        adjacency[b].append(a)
    parent = connection.atom_name
    gp_candidates = adjacency[parent]
    if not gp_candidates:
        raise ValueError(f"connection atom {parent} has no local frame atom")
    gp = sorted(gp_candidates)[0]
    ggp_candidates = [name for name in adjacency[gp] if name != parent]
    if not ggp_candidates:
        ggp_candidates = [name for name in local_atoms if name not in (parent, gp)]
    if not ggp_candidates:
        raise ValueError(f"connection at {parent} lacks a third local frame atom")
    ggp = sorted(ggp_candidates)[0]
    remote = coords[connection.partner_atom_name]
    return Icoor(
        name=connection.connection_name,
        phi=-_signed_dihedral_angle(remote, coords[parent], coords[gp], coords[ggp]),
        theta=math.pi - _angle(remote, coords[parent], coords[gp]),
        d=float(np.linalg.norm(remote - coords[parent])),
        parent=parent,
        grand_parent=gp,
        great_grand_parent=ggp,
    )


def _unresolved_atom_name(unresolved) -> str | None:
    return unresolved.atom


def _validate_bonded_cut_layout(
    ligand_name: str,
    adjacency: Mapping[str, Sequence[str]],
    cut_bonds: Sequence[tuple],
) -> None:
    """Reject layouts containing bonded terms that span three or more blocks."""

    cut_edges = {frozenset(bond[:2]) for bond in cut_bonds}
    cut_degree = {
        atom: sum(frozenset((atom, neighbor)) in cut_edges for neighbor in neighbors)
        for atom, neighbors in adjacency.items()
    }
    if any(degree > 1 for degree in cut_degree.values()):
        raise ValueError(
            f"{ligand_name}: no atom may participate in more than one fragment "
            "cut; impropers spanning three blocks are not supported"
        )

    def visit(path: tuple[str, ...]) -> None:
        if len(path) == 4:
            n_cuts = sum(
                frozenset((path[i], path[i + 1])) in cut_edges for i in range(3)
            )
            if n_cuts > 1:
                raise ValueError(
                    f"{ligand_name}: fragment cuts are too close along bonded path "
                    f"{'-'.join(path)}; torsions spanning three blocks are not supported"
                )
            return
        for neighbor in adjacency[path[-1]]:
            if neighbor not in path:
                visit(path + (neighbor,))

    for atom in adjacency:
        visit((atom,))


def _validate_scoring_cut_layout(restype: RawResidueType, cut_bonds: Sequence[tuple]):
    """Reject cuts that break nonbonded scoring geometry assumptions."""

    from tmol_core.ligand._chemistry_tables import _get_hbond_properties

    hbond_properties = _get_hbond_properties()
    atom_type_by_name = {atom.name: atom.atom_type for atom in restype.atoms}
    bad_cuts: list[str] = []
    for bond in cut_bonds:
        a, b = bond[:2]
        for atom_name, neighbor_name in ((a, b), (b, a)):
            atom_type = atom_type_by_name[atom_name]
            if hbond_properties.get(atom_type, {}).get("is_acceptor", False):
                bad_cuts.append(f"{atom_name}-{neighbor_name} ({atom_type})")

    if bad_cuts:
        raise ValueError(
            f"{restype.name}: fragment cuts through hbond/lk_ball acceptor "
            "geometry are not supported; acceptor atoms must remain in the "
            "same fragment as their bonded frame atoms. Unsupported cut(s): "
            + ", ".join(bad_cuts)
        )


def _build_ligand_fragment_definition(  # noqa: C901
    preparation: LigandPreparation,
    source_atom_array: struc.AtomArray,
) -> LigandFragmentDefinition | None:
    """Partition a fully prepared ligand according to its source annotation."""

    restype = preparation.residue_type
    source_fragment_ids = _fragment_ids_from_atom_array(source_atom_array)
    if source_fragment_ids is None:
        return None
    if np.any(source_fragment_ids == 0):
        zero_atom_names = list(
            map(str, source_atom_array.atom_name[source_fragment_ids == 0][:8])
        )
        raise ValueError(
            f"{restype.name}: every atom in a fragmented residue must have a "
            f"positive {FRAGMENT_ID_ANNOTATION}; found 0 for "
            f"{zero_atom_names}"
        )

    source_name_to_id = {
        str(name): int(fragment_id)
        for name, fragment_id in zip(source_atom_array.atom_name, source_fragment_ids)
    }
    source_elements = {
        str(name): str(element).upper()
        for name, element in zip(source_atom_array.atom_name, source_atom_array.element)
    }
    atom_names = [atom.name for atom in restype.atoms]
    adjacency = {name: [] for name in atom_names}
    for a, b, *_ in restype.bonds:
        adjacency[a].append(b)
        adjacency[b].append(a)

    atom_to_fragment: dict[str, int] = {
        name: source_name_to_id[name]
        for name in atom_names
        if name in source_name_to_id
    }
    unresolved = set(atom_names) - set(atom_to_fragment)
    while unresolved:
        progressed = False
        for name in tuple(unresolved):
            neighbor_ids = {
                atom_to_fragment[n] for n in adjacency[name] if n in atom_to_fragment
            }
            if len(neighbor_ids) == 1:
                atom_to_fragment[name] = neighbor_ids.pop()
                unresolved.remove(name)
                progressed = True
        if not progressed:
            raise ValueError(
                f"{restype.name}: could not assign prepared atoms to fragments: "
                f"{sorted(unresolved)}"
            )
    # Keep source-only hydrogen names in the public/input mapping too. Prepared
    # ligands may regenerate and rename these hydrogens.
    atom_to_fragment.update(
        {
            name: fragment_id
            for name, fragment_id in source_name_to_id.items()
            if name not in atom_to_fragment
        }
    )

    # Only prepared atoms define fragment blocks. Source-only names may retain
    # orphan IDs in the public mapping but cannot create empty blocks.
    fragment_ids = sorted({atom_to_fragment[atom.name] for atom in restype.atoms})
    orphan_ids = set(atom_to_fragment.values()) - set(fragment_ids)
    if orphan_ids:
        raise ValueError(
            f"{restype.name}: fragment id(s) {sorted(orphan_ids)} have no "
            "prepared atoms"
        )
    cut_bonds = [
        bond
        for bond in restype.bonds
        if atom_to_fragment[bond[0]] != atom_to_fragment[bond[1]]
    ]
    _validate_bonded_cut_layout(restype.name, adjacency, cut_bonds)
    _validate_scoring_cut_layout(restype, cut_bonds)
    connections_by_fragment: dict[int, list[FragmentConnection]] = {
        fragment_id: [] for fragment_id in fragment_ids
    }
    directed_connections: list[FragmentConnection] = []
    for cut_index, bond in enumerate(cut_bonds, start=1):
        a, b = bond[:2]
        fa, fb = atom_to_fragment[a], atom_to_fragment[b]
        name_a = f"conn_{cut_index}_{fa}_to_{fb}"
        name_b = f"conn_{cut_index}_{fb}_to_{fa}"
        # A cut through a ring leaves a connection whose bond is still a ring
        # bond; only the residue it belonged to has been split.
        in_ring = bool(bond[3]) if len(bond) > 3 else False
        conn_a = FragmentConnection(
            fa, fb, name_a, name_b, a, b, _bond_type_name(bond), in_ring
        )
        conn_b = FragmentConnection(
            fb, fa, name_b, name_a, b, a, _bond_type_name(bond), in_ring
        )
        connections_by_fragment[fa].append(conn_a)
        connections_by_fragment[fb].append(conn_b)
        directed_connections.extend((conn_a, conn_b))

    coords = _full_ideal_coords(restype)
    fragment_preparations: list[LigandPreparation] = []
    for fragment_id in fragment_ids:
        names = [
            atom.name
            for atom in restype.atoms
            if atom_to_fragment[atom.name] == fragment_id
        ]
        name_set = set(names)
        local_bonds = tuple(
            bond
            for bond in restype.bonds
            if bond[0] in name_set and bond[1] in name_set
        )

        seen = set()
        queue = deque([names[0]])
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            queue.extend(
                b if a == current else a
                for a, b, *_ in local_bonds
                if a == current or b == current
            )
        if seen != name_set:
            raise ValueError(
                f"{restype.name}.{fragment_id}: fragment must be one connected component"
            )

        element_by_name = {
            atom.name: (
                source_elements.get(atom.name)
                or (preparation.atom_type_elements or {}).get(atom.atom_type, "")
                or ("H" if atom.atom_type.upper().startswith("H") else "")
            ).upper()
            for atom in restype.atoms
            if atom.name in name_set
        }
        n_heavy = sum(element != "H" for element in element_by_name.values())
        if n_heavy < MIN_FRAGMENT_HEAVY_ATOMS:
            raise ValueError(
                f"{restype.name}.{fragment_id}: fragments require at least "
                f"{MIN_FRAGMENT_HEAVY_ATOMS} heavy atoms; found {n_heavy}"
            )
        if len(connections_by_fragment[fragment_id]) > MAX_FRAGMENT_CONNECTIONS:
            raise ValueError(
                f"{restype.name}.{fragment_id}: at most {MAX_FRAGMENT_CONNECTIONS} "
                "inter-block connections are supported"
            )

        order, parent, grandparents = _fragment_atom_tree(names, local_bonds, coords)
        icoors = _atom_icoors(order, parent, grandparents, coords)
        icoors.extend(
            _connection_icoor(conn, names, local_bonds, coords)
            for conn in connections_by_fragment[fragment_id]
        )

        torsions = tuple(
            torsion
            for torsion in restype.torsions
            if all(
                _unresolved_atom_name(atom) in name_set
                for atom in (torsion.a, torsion.b, torsion.c, torsion.d)
            )
        )
        torsion_names = {torsion.name for torsion in torsions}
        chi_samples = tuple(
            sample
            for sample in restype.chi_samples
            if sample.chi_dihedral in torsion_names
        )
        fragment_name = f"{restype.name}.{fragment_id}"
        fragment_restype = RawResidueType(
            name=fragment_name,
            base_name=restype.name,
            name3=fragment_name,
            io_equiv_class=fragment_name,
            atoms=tuple(atom for atom in restype.atoms if atom.name in name_set),
            atom_aliases=tuple(
                alias for alias in restype.atom_aliases if alias.name in name_set
            ),
            bonds=local_bonds,
            io_bond_orders=tuple(
                b
                for b in restype.io_bond_orders
                if b[0] in name_set and b[1] in name_set
            ),
            connections=tuple(
                Connection(
                    name=conn.connection_name,
                    atom=conn.atom_name,
                    type=conn.bond_type,
                    in_ring=conn.in_ring,
                )
                for conn in connections_by_fragment[fragment_id]
            ),
            torsions=torsions,
            icoors=tuple(icoors),
            properties=restype.properties,
            chi_samples=chi_samples,
            default_jump_connection_atom=order[0],
            # Fragment poses should honor provided H coordinates; absent leaf H
            # atoms can still be rebuilt from the fragment icoors.
            hydrogens_regenerated=False,
            is_ligand_fragment=True,
        )
        fragment_preparations.append(
            LigandPreparation(
                residue_type=fragment_restype,
                partial_charges={
                    name: charge
                    for name, charge in preparation.partial_charges.items()
                    if name in name_set
                },
                cartbonded_params=preparation.cartbonded_params,
                atom_type_elements=preparation.atom_type_elements,
            )
        )

    return LigandFragmentDefinition(
        ligand_name=restype.name,
        atom_to_fragment=atom_to_fragment,
        fragment_ids=tuple(fragment_ids),
        fragment_preparations=tuple(fragment_preparations),
        connections=tuple(directed_connections),
    )


def _expand_fragmented_ligands(  # noqa: C901
    structure: struc.AtomArray | struc.AtomArrayStack,
    definitions: Sequence[LigandFragmentDefinition],
) -> tuple[struc.AtomArray | struc.AtomArrayStack, FragmentedLigandPoseMapping]:
    """Replace each annotated ligand residue with contiguous fragment residues."""

    if not definitions:
        return (
            structure,
            FragmentedLigandPoseMapping(blocks=(), connection_pairs=()),
        )
    definition_by_name = {
        definition.ligand_name: definition for definition in definitions
    }
    representative = (
        structure[0] if isinstance(structure, struc.AtomArrayStack) else structure
    )
    residue_starts = struc.get_residue_starts(representative)
    residue_ends = np.append(residue_starts[1:], representative.array_length())

    atom_order: list[int] = []
    output_names: list[str] = []
    output_residue_labels: list[int] = []
    block_records: list[LigandFragmentBlockMapping] = []
    connection_pairs: list[tuple[int, str, int, str]] = []
    output_block_index = 0
    next_synthetic_residue_label = int(np.max(representative.res_id)) + 1

    for start, end in zip(residue_starts, residue_ends):
        ligand_name = str(representative.res_name[start])
        definition = definition_by_name.get(ligand_name)
        if definition is None:
            inds = list(range(int(start), int(end)))
            atom_order.extend(inds)
            output_names.extend(str(representative.res_name[i]) for i in inds)
            output_residue_labels.extend(int(representative.res_id[i]) for i in inds)
            output_block_index += 1
            continue

        local_names = [str(name) for name in representative.atom_name[start:end]]
        local_ids: list[int] = []
        annotation = (
            np.asarray(getattr(representative, FRAGMENT_ID_ANNOTATION))[start:end]
            if FRAGMENT_ID_ANNOTATION in representative.get_annotation_categories()
            else None
        )
        for local_index, atom_name in enumerate(local_names):
            if atom_name not in definition.atom_to_fragment:
                raise ValueError(
                    f"{ligand_name}: atom {atom_name!r} is absent from the prepared "
                    "fragment mapping"
                )
            fragment_id = definition.atom_to_fragment[atom_name]
            if annotation is not None and int(annotation[local_index]) != fragment_id:
                raise ValueError(
                    f"{ligand_name}: {FRAGMENT_ID_ANNOTATION} changed for atom "
                    f"{atom_name!r} after the build context was created"
                )
            local_ids.append(fragment_id)

        block_for_fragment: dict[int, int] = {}
        for fragment_id in definition.fragment_ids:
            selected = [
                int(start) + i
                for i, value in enumerate(local_ids)
                if value == fragment_id
            ]
            if not selected:
                raise ValueError(
                    f"{ligand_name}.{fragment_id}: no atoms found in this structure"
                )
            fragment_name = definition.fragment_name(fragment_id)
            atom_order.extend(selected)
            output_names.extend([fragment_name] * len(selected))
            synthetic_residue_label = next_synthetic_residue_label
            next_synthetic_residue_label += 1
            output_residue_labels.extend([synthetic_residue_label] * len(selected))
            block_for_fragment[fragment_id] = output_block_index
            block_records.append(
                LigandFragmentBlockMapping(
                    pose_index=0,
                    ligand_name=ligand_name,
                    residue_label=int(representative.res_id[start]),
                    pose_residue_label=synthetic_residue_label,
                    chain_label=str(representative.chain_id[start]),
                    insertion_code=str(representative.ins_code[start]),
                    fragment_id=fragment_id,
                    block_index=output_block_index,
                    atom_names=tuple(
                        str(representative.atom_name[i]) for i in selected
                    ),
                )
            )
            output_block_index += 1

        seen_pairs: set[frozenset[str]] = set()
        for connection in definition.connections:
            key = frozenset(
                (connection.connection_name, connection.partner_connection_name)
            )
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            connection_pairs.append(
                (
                    block_for_fragment[connection.fragment_id],
                    connection.connection_name,
                    block_for_fragment[connection.partner_fragment_id],
                    connection.partner_connection_name,
                )
            )

    if isinstance(structure, struc.AtomArrayStack):
        expanded = structure[:, atom_order].copy()
    else:
        expanded = structure[atom_order].copy()
    # Biotite's built-in res_name annotation is commonly U3. Replace it instead
    # of assigning into it so deterministic names such as ``XYZ.12`` are retained.
    expanded.set_annotation(
        "res_name",
        np.asarray(
            output_names, dtype=f"U{max(3, max(map(len, output_names), default=3))}"
        ),
    )
    expanded.set_annotation("res_id", np.asarray(output_residue_labels, dtype=np.int32))

    # Cut bonds belong to the prepared fragment mapping and are installed by
    # apply_fragment_connections. Leaving them in the input bond table makes
    # canonical conversion request conjugation patches for the same cuts.
    # Preserve all bonds within fragments and between original residues.
    if expanded.bonds is not None:
        original_residue = np.repeat(
            np.arange(len(residue_starts)), residue_ends - residue_starts
        )[atom_order]
        bonds = expanded.bonds.as_array()
        a, b = bonds[:, 0], bonds[:, 1]
        is_cut = (original_residue[a] == original_residue[b]) & (
            expanded.res_id[a] != expanded.res_id[b]
        )
        expanded.bonds = struc.BondList(expanded.array_length(), bonds[~is_cut])

    n_poses = len(structure) if isinstance(structure, struc.AtomArrayStack) else 1
    blocks = tuple(
        replace(record, pose_index=pose_index)
        for pose_index in range(n_poses)
        for record in block_records
    )
    return (
        expanded,
        FragmentedLigandPoseMapping(
            blocks=blocks, connection_pairs=tuple(connection_pairs)
        ),
    )
