"""Shared structure chemistry and parameter context preparation.

Coordinate storage and canonical conversion are frontend execution boundaries.
All residue normalization, protonation, chemistry filtering and ligand routing
retain the mature host policy here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import logging

import numpy
import biotite.structure
from tmol_core.chemical import _metal_table
from tmol_core.io.canonical_ordering import CanonicalOrdering
from tmol_core.io.atom_array import (
    METAL_ORIGIN,
    template_array as _template_array,
    is_metal as _is_metal,
    metal_coordination_bond_mask as _metal_coordination_bond_mask,
)
from tmol_core.io.input_termini import (
    EXPLICIT_TERMINI,
    _with_stated_termini,
    _validate_connection_hydrogens,
)
from tmol_core.io.protonation_input import (
    PROTONATION_VARIANT,
    _database_forms,
    _residues_lacking_hydrogens,
    _with_atomworks_hydrogens,
)

logger = logging.getLogger(__name__)
RESIDUE_ANNOTATIONS = ("hetero", "is_polymer", "chain_type")


def _get_all_residue_positions(array):
    return biotite.structure.get_all_residue_positions(array)


def _without_metal_coordination_bonds(structure):
    """(structure without its metal coordination bonds, which ligand preparation does
    not read, {residue name: its atoms with a declared metal bond})."""
    if structure.bonds is None:
        return structure, {}
    bonds, coordination = _metal_coordination_bond_mask(structure)
    if not coordination.any():
        return structure, {}
    template = _template_array(structure)
    is_metal = _is_metal(template)
    bound = defaultdict(set)
    for i in bonds[coordination, :2].ravel():
        if not is_metal[i]:
            bound[str(template.res_name[i])].add(str(template.atom_name[i]))
    structure = structure.copy()
    structure.bonds = biotite.structure.BondList(
        template.array_length(), bonds[~coordination]
    )
    return structure, {name: frozenset(atoms) for name, atoms in bound.items()}


def _metal_atom_names(chemdb=None, co=None):
    """The metal atom of each ion residue class, by name3."""
    if chemdb is not None:
        return {
            r.io_equiv_class: r.metal_sites[0].metal_atom
            for r in chemdb.residues
            if r.metal_sites
        }
    out = {}
    for ion in _metal_table()["ions"]:
        names = co.restypes_ordered_atom_names.get(ion["name3"])
        if names:
            virtual = co.restypes_virtual_atoms.get(ion["name3"], frozenset())
            out[ion["name3"]] = next(n for n in names if n and n not in virtual)
    return out


def _chelated_metals(template, metal_atom):
    """{atom: (ion, atom name)} for resolved supported metals bonded only to N/O/S of a
    component that stays one organic molecule without them (a heme's Fe, not a cluster).
    """
    ion_for_element = {}
    for ion in _metal_table()["ions"]:
        element = ion["element"].upper()
        ion_for_element.setdefault((element, 0), ion["name3"])
        ion_for_element[element, ion["oxidation_state"]] = ion["name3"]
    elements = numpy.char.upper(template.element.astype(str))
    charge = getattr(template, "charge", numpy.zeros(len(template), dtype=int))
    starts = biotite.structure.get_residue_starts(template, add_exclusive_stop=True)
    bonds = template.bonds
    moved = {}
    for start, stop in zip(starts[:-1], starts[1:]):
        if stop - start < 2:
            continue
        members = set(range(start, stop))
        metals = [i for i in members if (elements[i], 0) in ion_for_element]
        rest = members - set(metals)
        if not metals or "C" not in elements[list(rest)]:
            continue
        # bonds to other residues are ordinary coordination; within, only N/O/S
        partners = {i: set(bonds.get_bonds(i)[0].tolist()) & rest for i in metals}
        if not all(partners.values()):
            continue
        if not all(
            elements[j] in ("N", "O", "S") for p in partners.values() for j in p
        ):
            continue
        # the remainder must be one molecule
        seen, stack = set(), [min(rest)]
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            stack.extend(j for j in bonds.get_bonds(i)[0].tolist() if j in rest)
        if seen != rest:
            continue
        for i in metals:
            name3 = ion_for_element.get((elements[i], int(charge[i])))
            if name3 in metal_atom and numpy.isfinite(template.coord[i]).all():
                moved[i] = (name3, metal_atom[name3])
    return moved


def _residue_annotations(structure):
    """Per residue, the input's RESIDUE_ANNOTATIONS as one structured array."""
    names = [
        name
        for name in RESIDUE_ANNOTATIONS
        if name in structure.get_annotation_categories()
    ]
    if not names:
        return None
    starts = biotite.structure.get_residue_starts(structure)
    values = [structure.get_annotation(name)[starts] for name in names]
    # object strings, so poses from different inputs share one dtype
    values = [v.astype(object) if v.dtype.kind in "US" else v for v in values]
    return numpy.rec.fromarrays(values, names=names).view(numpy.ndarray)


def _metal_origins(structure):
    """Per residue, (res label, ins code, res name, atom name) a metal came from."""
    if METAL_ORIGIN not in structure.get_annotation_categories():
        return None
    starts = biotite.structure.get_residue_starts(structure)
    out = numpy.full(len(starts), None, dtype=object)
    for r, value in enumerate(structure.get_annotation(METAL_ORIGIN)[starts]):
        if value:
            res_id, ins, name, atom = str(value).split("\t")
            out[r] = (int(res_id), ins, name, atom)
    return out if any(o is not None for o in out) else None


def _with_peptide_tautomers(structure):
    """Rewrite an inter-residue C=N link whose C carries a single-bonded, uncharged
    terminal O/S as the amide (C=X, hydrogens on X dropped), as 1MRO's GL3 reads."""
    template = _template_array(structure)
    if template.bonds is None or template.array_length() == 0:
        return structure
    bonds = template.bonds.as_array()
    elements = numpy.char.upper(template.element.astype(str))
    residue = biotite.structure.get_residue_positions(
        template, numpy.arange(template.array_length())
    )
    double = bonds[:, 2] == biotite.structure.BondType.DOUBLE
    charge = (
        template.charge
        if "charge" in template.get_annotation_categories()
        else numpy.zeros(template.array_length(), dtype=int)
    )
    rewrites, dropped = [], set()
    for a, b, _ in bonds[double & (residue[bonds[:, 0]] != residue[bonds[:, 1]])]:
        carbon, nitrogen = (a, b) if elements[a] == "C" else (b, a)
        if elements[carbon] != "C" or elements[nitrogen] != "N":
            continue
        for x, order in zip(*template.bonds.get_bonds(carbon)):
            if (
                elements[x] not in ("O", "S")
                or order != biotite.structure.BondType.SINGLE
                or charge[x] != 0
            ):
                continue
            partners = template.bonds.get_bonds(x)[0]
            hydrogens = [h for h in partners if elements[h] in ("H", "D")]
            if len(partners) - len(hydrogens) == 1:
                rewrites.append((carbon, nitrogen, x))
                dropped.update(hydrogens)
                break
    if not rewrites:
        return structure
    orders = {}
    for carbon, nitrogen, x in rewrites:
        orders[frozenset((carbon, nitrogen))] = biotite.structure.BondType.SINGLE
        orders[frozenset((carbon, x))] = biotite.structure.BondType.DOUBLE
    for row in bonds:
        order = orders.get(frozenset((int(row[0]), int(row[1]))))
        if order is not None:
            row[2] = order
    structure = structure.copy()
    structure.bonds = biotite.structure.BondList(template.array_length(), bonds)
    keep = numpy.ones(template.array_length(), dtype=bool)
    keep[list(dropped)] = False
    if isinstance(structure, biotite.structure.AtomArrayStack):
        return structure[:, keep]
    return structure[keep]


def _with_input_chemistry_normalized(structure, metal_atom):
    """Input chemistry as tmol models it: amide peptide links, chelated metals split."""
    return _with_chelated_metals_split(_with_peptide_tautomers(structure), metal_atom)


def _with_chelated_metals_split(structure, metal_atom):
    """Move each chelated metal into an ion residue of its own, bonded by coordination;
    METAL_ORIGIN keeps where it sat, for export to put it back."""
    template = _template_array(structure)
    if template.bonds is None:
        return structure
    moved = _chelated_metals(template, metal_atom)
    if not moved:
        return structure
    starts = biotite.structure.get_residue_starts(template, add_exclusive_stop=True)
    structure = structure.copy()
    template = _template_array(structure)
    bonds = template.bonds.as_array()
    touches = numpy.isin(bonds[:, 0], list(moved)) | numpy.isin(
        bonds[:, 1], list(moved)
    )
    bonds[touches, 2] = biotite.structure.BondType.COORDINATION
    bond_list = biotite.structure.BondList(template.array_length(), bonds)

    next_id = {}
    for chain in numpy.unique(structure.chain_id):
        next_id[chain] = int(structure.res_id[structure.chain_id == chain].max()) + 1
    origin = numpy.full(template.array_length(), "", dtype=object)
    if METAL_ORIGIN in structure.get_annotation_categories():
        origin[:] = structure.get_annotation(METAL_ORIGIN)
    for i, (name3, atom_name) in sorted(moved.items()):
        origin[i] = "\t".join(
            str(a[i])
            for a in (
                structure.res_id,
                structure.ins_code,
                structure.res_name,
                structure.atom_name,
            )
        )
        chain = structure.chain_id[i]
        structure.res_name[i] = name3
        structure.atom_name[i] = atom_name
        structure.res_id[i] = next_id[chain]
        next_id[chain] += 1
        structure.hetero[i] = True
    structure.bonds = bond_list
    structure.set_annotation(METAL_ORIGIN, origin.astype(str))
    # each ion follows the component it came from, so residues stay contiguous
    order, components = [], set()
    for start, stop in zip(starts[:-1], starts[1:]):
        members = range(start, stop)
        order += [i for i in members if i not in moved]
        order += [i for i in members if i in moved]
        kept = [i for i in members if i not in moved]
        if kept and len(kept) < len(members):
            components.add(str(template.res_name[kept[0]]))
    registry = getattr(structure, "_custom_ccd_registry", None)
    if isinstance(structure, biotite.structure.AtomArrayStack):
        structure = structure[:, order]
    else:
        structure = structure[order]
    if registry:
        structure._custom_ccd_registry = _without_chelated_metals(
            registry, components, _ion_elements(metal_atom)
        )
    return structure


def _ion_elements(metal_atom):
    """Elements of the ions a chelated metal can become."""
    return {
        ion["element"].upper()
        for ion in _metal_table()["ions"]
        if ion["name3"] in metal_atom
    }


def _without_chelated_metals(registry, components, elements):
    """Component templates as their split residues are: no metal, and no declared
    stereo on its partners (a heme pyrrole N is planar without it)."""
    out = dict(registry)
    for name in components:
        template = registry.get(name)
        if template is None:
            continue
        is_metal = numpy.isin(
            numpy.char.upper(template.element.astype(str)), list(elements)
        )
        if not is_metal.any():
            continue
        template = template.copy()
        if "stereo" in template.get_annotation_categories():
            for i in numpy.flatnonzero(is_metal):
                for j in template.bonds.get_bonds(i)[0]:
                    template.stereo[j] = "N"
        out[name] = template[~is_metal]
    return out


def _res_names_for_structure(
    biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack,
):
    if isinstance(biotite_structure, biotite.structure.AtomArrayStack):
        return biotite_structure[0].res_name
    return biotite_structure.res_name


def _validate_filtered_covalent_partners(
    array, co, atom_res, valid_res, res_names, cut_partners=frozenset()
):
    """Allow sequential backbone gaps, but not a dangling chemical partner.

    This runs only when a non-water residue is removed. Complete structures
    and ordinary water filtering need no extra bond-table scan.
    """
    if array.bonds is None:
        return
    # A residue whose covalent bond ligand preparation already cut, because the
    # caller passed strict_ligands=False, is a partner it was told to drop.
    removed = (
        ~valid_res & (res_names != "HOH") & ~numpy.isin(res_names, list(cut_partners))
    )
    if not numpy.any(removed):
        return
    bonds = array.bonds.as_array()
    if not len(bonds):
        return
    ends = atom_res[bonds[:, :2]]
    crosses = (removed[ends[:, 0]] & valid_res[ends[:, 1]]) | (
        removed[ends[:, 1]] & valid_res[ends[:, 0]]
    )
    type_indices = {name: i for i, name in enumerate(co.restype_io_equiv_classes)}
    connections = co.polymer_conn_inds
    for first, second, order in bonds[crosses]:
        if order == biotite.structure.BondType.COORDINATION:
            continue
        first_res, second_res = atom_res[[first, second]]
        # Orient the candidate in input residue order, preserving insertion codes.
        if first_res > second_res:
            first, second = second, first
            first_res, second_res = second_res, first_res
        first_name, second_name = array.res_name[[first, second]]
        first_type = type_indices.get(first_name)
        second_type = type_indices.get(second_name)
        if (
            second_res == first_res + 1
            and array.chain_id[first] == array.chain_id[second]
            and first_type is not None
            and second_type is not None
        ):
            first_atom = co.restypes_atom_index_mapping[first_name].get(
                array.atom_name[first], -1
            )
            second_atom = co.restypes_atom_index_mapping[second_name].get(
                array.atom_name[second], -1
            )
            if (
                first_atom >= 0
                and second_atom >= 0
                and (
                    (
                        first_atom == connections.up_atom_for_co_restype[first_type]
                        and second_atom
                        == connections.down_atom_for_co_restype[second_type]
                    )
                    or (
                        # Reordering an explicitly numbered chain does not turn
                        # an ordinary gap into a cyclic/crosslinked attachment.
                        # Require the source sequence direction; a backward link
                        # in forward residue order still closes a cycle.
                        first_atom == connections.down_atom_for_co_restype[first_type]
                        and second_atom
                        == connections.up_atom_for_co_restype[second_type]
                        and (array.res_id[first], array.ins_code[first])
                        > (array.res_id[second], array.ins_code[second])
                    )
                )
            ):
                continue

        def label(index):
            return (
                f"{array.res_name[index]} {array.chain_id[index]}:"
                f"{array.res_id[index]}{array.ins_code[index]}"
                f"/{array.atom_name[index]}"
            )

        raise ValueError(
            "Cannot discard an incomplete or unsupported residue while retaining "
            "its covalent partner: declared bond "
            f"{label(first)} -- {label(second)} would be lost. "
            "Supply the required backbone coordinates/chemical definition, or "
            "explicitly select a complete covalent component before construction."
        )


def _filter_supported_atoms_and_connectivity(  # noqa: C901
    biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack,
    co: CanonicalOrdering,
    *,
    filter_missing_mainchain: bool = True,
    cut_partners: frozenset[str] = frozenset(),
):
    biotite_residues = biotite.structure.get_residues(biotite_structure)[1]
    to_remove = {"HOH"}
    known_residue_names = set(co.restype_io_equiv_classes)
    for i_3lc in biotite_residues:
        if i_3lc in to_remove:
            continue
        if i_3lc not in known_residue_names:
            logger.warning("Unrecognized 3lc %s", i_3lc)
            to_remove.add(i_3lc)

    res_names = _res_names_for_structure(biotite_structure)
    biotite_residue_starts = biotite.structure.get_residue_starts(biotite_structure)
    valid_res = numpy.array([name not in to_remove for name in res_names], dtype=bool)[
        biotite_residue_starts
    ]

    # Filter residues missing mainchain atoms required for rotamer building.
    # Only atoms present in every variant count as required, so an atom a terminus
    # patch removes (the DNA 5' phosphate) does not disqualify the residue.
    # Residues with no mainchain definition (non-polymer) are skipped.
    n_missing_mainchain = 0
    if filter_missing_mainchain:
        atom_names = biotite_structure.atom_name
        if isinstance(biotite_structure, biotite.structure.AtomArrayStack):
            coords = biotite_structure.coord  # (n_poses, n_atoms, 3)
        else:
            coords = biotite_structure.coord[numpy.newaxis, :]  # (1, n_atoms, 3)
        residue_ends = numpy.append(
            biotite_residue_starts[1:], biotite_structure.array_length()
        )
        for i in range(len(valid_res)):
            if not valid_res[i]:
                continue
            start, end = biotite_residue_starts[i], residue_ends[i]
            res_name3 = biotite_structure.res_name[start]
            required = co.restypes_required_mainchain_atoms.get(res_name3)
            if not required:
                continue
            mapping = co.restypes_atom_index_mapping[res_name3]
            resolved = numpy.isfinite(coords[:, start:end, :]).all(axis=(0, 2))
            present = {
                mapping[name]
                for name in atom_names[start:end][resolved]
                if name in mapping
            }
            missing = {name for name in required if mapping[name] not in present}
            if missing:
                logger.warning(
                    "Residue %s %s %d is missing mainchain atoms %s; skipping",
                    biotite_structure.chain_id[start],
                    res_name3,
                    biotite_structure.res_id[start],
                    sorted(missing),
                )
                valid_res[i] = False
                n_missing_mainchain += 1
    if n_missing_mainchain and not valid_res.any():
        # 1A1D: a CA-only trace leaves no residue with its mainchain
        raise ValueError(
            f"No residue remains: {n_missing_mainchain} polymer residues lack "
            "mainchain atoms (a CA-only or P-only trace model?) and were dropped. "
            "A PoseStack needs resolved backbones; rebuild them first or pass "
            "complete coordinates."
        )

    atom_res = _get_all_residue_positions(biotite_structure)
    _validate_filtered_covalent_partners(
        _template_array(biotite_structure),
        co,
        atom_res,
        valid_res,
        biotite_residues,
        cut_partners,
    )
    valid_atoms = valid_res[atom_res]

    # A kept residue whose neighbor was dropped has an unknown connection on
    # that side; the ends of the kept set are termini, so they are marked after
    # filtering
    lower = numpy.roll(valid_res, 1)[valid_res]
    upper = numpy.roll(valid_res, -1)[valid_res]
    if lower.size:
        lower[0] = True
        upper[-1] = True
    not_connected = numpy.invert(numpy.column_stack((lower, upper)))

    if isinstance(biotite_structure, biotite.structure.AtomArrayStack):
        biotite_structure = biotite_structure[:, valid_atoms]
    else:
        biotite_structure = biotite_structure[valid_atoms]

    return biotite_structure, not_connected


def _extract_residue_metadata(
    biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack,
    not_connected,
):
    biotite_residue_starts = biotite.structure.get_residue_starts(biotite_structure)

    # Residue labels are not chain identities. Biotite's get_chain_starts also
    # splits whenever res_id decreases, turning a reversed chain into one chain
    # per residue. Work at residue granularity and retain explicit symmetry IDs.
    # Author chain labels can be shared by a polymer and separate ligand
    # entities. That must not hide the polymer's terminal boundary.
    keys = ["chain_id"]
    if "label_entity_id" in biotite_structure.get_annotation_categories():
        keys.append("label_entity_id")
    if "sym_id" in biotite_structure.get_annotation_categories():
        keys.append("sym_id")
    boundaries = numpy.zeros(max(0, len(biotite_residue_starts) - 1), dtype=bool)
    for key in keys:
        values = biotite_structure.get_annotation(key)[biotite_residue_starts]
        boundaries |= values[1:] != values[:-1]
    biotite_chain_id_for_res = numpy.cumsum(numpy.r_[0, boundaries])[
        : len(biotite_residue_starts)
    ]

    if len(biotite_chain_id_for_res) > 1:
        res_is_disconnected_from_neighbor = (
            biotite_chain_id_for_res[1:] != biotite_chain_id_for_res[:-1]
        )
        not_connected[1:, 0] &= ~res_is_disconnected_from_neighbor
        not_connected[:-1, 1] &= ~res_is_disconnected_from_neighbor

    biotite_chain_labels = biotite_structure.chain_id[biotite_residue_starts]
    biotite_insertion_codes = biotite_structure.ins_code[biotite_residue_starts]
    biotite_residue_labels, biotite_residues = biotite.structure.get_residues(
        biotite_structure
    )
    return (
        biotite_chain_id_for_res,
        biotite_chain_labels,
        biotite_insertion_codes,
        biotite_residue_labels,
        biotite_residues,
    )


def _normalize_input_identifiers(biotite_structure, name3_aliases):
    """Resolve residue aliases and AtomWorks assembly instances without mutation.

    An aliased residue is read as the one it names, so nothing downstream --
    atom mapping, restype lookup, nonstandard-residue detection -- ever sees
    the input name. Atom names follow through the target's atom aliases.
    """
    from atomworks.io.utils.atom_array import chain_identifier

    chains = chain_identifier(biotite_structure)
    names = biotite_structure.res_name
    rename_residues = bool(name3_aliases) and any(
        name in name3_aliases for name in numpy.unique(names)
    )
    if not rename_residues and (
        chains is biotite_structure.chain_id
        or numpy.array_equal(chains, biotite_structure.chain_id)
    ):
        return biotite_structure
    renamed = biotite_structure.copy()
    renamed.chain_id = chains.copy()
    if rename_residues:
        renamed.res_name = numpy.array([name3_aliases.get(n, n) for n in names])
    return renamed


def _with_input_hydrogens(
    biotite_structure,
    ph,
    co,
    chemdb,
    find_metal_coordination,
    *,
    detect_metal_bonds,
    default_canonical_ordering,
):
    """The input with AtomWorks' protonation of residues lacking hydrogens (only
    ``co``'s if given), seeing detected metal bonds and ``chemdb`` backbone links."""
    biotite_structure = _with_stated_termini(biotite_structure, chemdb, co)
    _validate_connection_hydrogens(biotite_structure, chemdb)
    aliases = {a.name3: a.read_as for a in chemdb.name3_aliases}
    names = None
    if co is not None:
        names = set(co.restype_io_equiv_classes)
        names |= {alias for alias, name in aliases.items() if name in names}
    template = _template_array(biotite_structure)
    forms = _database_forms(chemdb)
    if PROTONATION_VARIANT in template.get_annotation_categories() or (
        EXPLICIT_TERMINI not in template.get_annotation_categories()
        and not _residues_lacking_hydrogens(biotite_structure, names, forms)[1].any()
    ):
        return biotite_structure
    metal_atom = _metal_atom_names(chemdb=chemdb)
    biotite_structure = _with_input_chemistry_normalized(
        _normalize_input_identifiers(biotite_structure, aliases), metal_atom
    )
    coordination = None
    res_name = _template_array(biotite_structure).res_name
    if find_metal_coordination and numpy.isin(res_name, list(metal_atom)).any():
        coordination = detect_metal_bonds(
            biotite_structure, co or default_canonical_ordering(), chemdb
        )
    backbone = {}
    for residue in chemdb.residues:
        connection = {c.name: c.atom for c in residue.connections}
        up, down = backbone.get(residue.io_equiv_class, (None, None))
        backbone[residue.io_equiv_class] = (
            up or connection.get("up"),
            down or connection.get("down"),
        )
    return _with_atomworks_hydrogens(
        biotite_structure,
        ph=ph,
        residue_names=names,
        coordination=coordination,
        backbone=backbone,
        forms=forms,
    )


@dataclass(frozen=True)
class ContextPreparation:
    structure: object
    parameter_database: object
    canonical_ordering: object
    fragment_definitions: object
    cut_covalent_partners: object
    use_default_context: bool


def prepare_structure_context(
    structure,
    *,
    parameter_database,
    default_database,
    prepare_ligands,
    ligand_ph,
    strict_atom_types,
    strict_ligands,
    ligand_params_files,
    chem_comp_types,
    ligand_seed,
    ligand_hydrogens,
    input_hydrogens,
    ligand_preparer,
):
    """Prepare chemistry once, retaining the caller's database and strict policy.

    The frontend supplies only default context and execution services. Supplied
    parameter files, normalization order, metal bond handling, fragment/cut
    metadata and ligand options are shared with the mature constructor.
    """
    from tmol_core.io.missing_atoms import validate_ligand_hydrogens

    validate_ligand_hydrogens(ligand_hydrogens)
    db = parameter_database
    if ligand_params_files and not prepare_ligands:
        from tmol_core.ligand import inject_params_files

        db = inject_params_files(
            db or default_database(),
            ligand_params_files,
            strict_atom_types=strict_atom_types,
        )
    chemical = (db or default_database()).chemical
    structure = _normalize_input_identifiers(
        structure, {alias.name3: alias.read_as for alias in chemical.name3_aliases}
    )
    structure = _with_input_chemistry_normalized(
        structure, _metal_atom_names(chemdb=chemical)
    )
    if prepare_ligands:
        structure = input_hydrogens(structure, ligand_ph, None, chemical, True)
    structure, coordinating_atoms = _without_metal_coordination_bonds(structure)
    if not prepare_ligands:
        return ContextPreparation(
            structure, db or default_database(), None, (), frozenset(), db is None
        )
    using_default = db is None
    db, ordering, fragments, cut_partners = ligand_preparer(
        structure,
        param_db=db or default_database(),
        ph=ligand_ph,
        strict_atom_types=strict_atom_types,
        params_files=ligand_params_files,
        strict_ligands=strict_ligands,
        return_fragment_definitions=True,
        return_cut_partners=True,
        chem_comp_types=chem_comp_types,
        seed=ligand_seed,
        coordinating_atoms=coordinating_atoms,
        input_hydrogen_names=ligand_hydrogens != "rebuild",
    )
    return ContextPreparation(
        structure,
        db,
        ordering,
        fragments,
        cut_partners,
        using_default and db is default_database() and not fragments,
    )


def detected_metal_bonds_for_canonical(structure, co, chemdb, cf):
    """Map shared metal/disulfide decisions back to original input atom indices."""
    from tmol_core.io.disulfides import prepare_disulfide_search, resolve_disulfides
    from tmol_core.io.metal_detection import declared_metal_sites, find_metal_geometries

    template = _template_array(structure)
    if not (cf.res_types >= 0).all():
        return None
    declared, required = declared_metal_sites(cf.metal_sites, cf.metal_coordination)
    plan = prepare_disulfide_search(
        co.cys_inds.cys_co_aa_inds,
        cf.res_types,
        cf.disulfides,
        True,
        co.cys_inds.sg_atoms_for_co_cys if co.cys_inds.sg_atom_for_co_cys < 0 else None,
    )
    _, disulfide_variants = resolve_disulfides(
        plan,
        cf.coords if plan.search else None,
        co.cys_inds.sg_atom_for_co_cys,
    )
    _, assignments = find_metal_geometries(
        co,
        chemdb,
        cf.res_types,
        cf.coords,
        excluded_donor_residues=disulfide_variants != 0,
        declared_sites=declared,
        find_additional=True,
        required_donors=required,
    )
    ins = template.ins_code
    index = {
        key: i
        for i, key in enumerate(
            zip(template.chain_id, template.res_id, ins, template.atom_name)
        )
    }
    metallic = numpy.isin(
        numpy.char.upper(template.element.astype(str)),
        [e.upper() for e in _ion_elements(_metal_atom_names(chemdb=chemdb))],
    )

    def atom_of(pose, res, atom):
        name = co.restypes_ordered_atom_names[
            co.restype_io_equiv_classes[int(cf.res_types[pose, res])]
        ][atom]
        return index.get(
            (
                cf.chain_labels[pose, res],
                int(cf.res_labels[pose, res]),
                str(cf.residue_insertion_codes[pose, res] or ""),
                name,
            )
        )

    pairs = []
    for (pose, metal), got in assignments:
        if pose != 0:
            continue
        chain, res_id = cf.chain_labels[0, metal], int(cf.res_labels[0, metal])
        metals = numpy.flatnonzero(
            metallic & (template.chain_id == chain) & (template.res_id == res_id)
        )
        for res, atom in got.donor_atoms:
            donor = atom_of(0, res, atom)
            if donor is None or not len(metals):
                continue
            dist = numpy.linalg.norm(
                template.coord[metals] - template.coord[donor], axis=-1
            )
            pairs.append((int(metals[numpy.argmin(dist)]), donor))
    return numpy.array(pairs, dtype=numpy.int64).reshape(-1, 2)
