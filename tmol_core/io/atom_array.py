"""Host AtomArray bond identity and provenance shared by framework readers.

This optional I/O module uses Biotite; importing the scientific core itself
requires no structure-reader or framework runtime.
"""

import numpy
import biotite.structure
from tmol_core.chemical import _metal_table

INFERRED_POLYMER_BOND = "_tmol_inferred_polymer_bond"


def template_array(structure):
    """The single AtomArray whose bond table describes every model."""
    if isinstance(structure, biotite.structure.AtomArrayStack):
        return structure[0]
    return structure


def is_metal(structure, table=None) -> numpy.ndarray:
    """Per atom, whether its element is one the metal table coordinates."""
    table = _metal_table() if table is None else table
    metals = [ion["element"].upper() for ion in table["ions"]]
    return numpy.isin(
        numpy.char.upper(numpy.asarray(structure.element, dtype=str)), metals
    )


def metal_coordination_bond_mask(structure):
    """Which bond rows are metal coordination: typed COORDINATION (CIF metalc), or from
    a metal to another residue (PDB CONECT)."""
    bonds = structure.bonds.as_array()
    if not len(bonds):
        return bonds, numpy.zeros(0, dtype=bool)
    template = template_array(structure)
    metal_atom = is_metal(template)
    residue = biotite.structure.get_residue_positions(
        template, numpy.arange(template.array_length())
    )
    crosses = residue[bonds[:, 0]] != residue[bonds[:, 1]]
    coordination = (bonds[:, 2] == biotite.structure.BondType.COORDINATION) | (
        crosses & (metal_atom[bonds[:, 0]] | metal_atom[bonds[:, 1]])
    )
    return bonds, coordination


def with_metal_coordination_typed(structure):
    """Type every metal coordination bond COORDINATION, however it was read."""
    if structure.bonds is None:
        return structure
    bonds, coordination = metal_coordination_bond_mask(structure)
    untyped = coordination & (bonds[:, 2] != biotite.structure.BondType.COORDINATION)
    if not untyped.any():
        return structure
    structure = structure.copy()
    bonds = bonds.copy()
    bonds[untyped, 2] = biotite.structure.BondType.COORDINATION
    structure.bonds = biotite.structure.BondList(
        template_array(structure).array_length(), bonds
    )
    return structure


def metal_coordination_from_biotite(
    array, atom_res_inds, valid_atom_mask, valid_atom_inds
):
    """(metal, -1, donor, donor atom) for each declared metal bond.

    The site is left for detection to choose.
    """
    if array.bonds is None:
        return numpy.zeros((0, 4), dtype=numpy.int64)
    bonds = array.bonds.as_array()
    declared = bonds[bonds[:, 2] == biotite.structure.BondType.COORDINATION]
    if not len(declared):
        return numpy.zeros((0, 4), dtype=numpy.int64)
    atom_canonical_ind = numpy.full(array.array_length(), -1, dtype=numpy.int64)
    atom_canonical_ind[valid_atom_mask] = valid_atom_inds
    metal_atom = is_metal(array)
    rows = []
    for atom1, atom2, _ in declared:
        if metal_atom[atom2] and not metal_atom[atom1]:
            atom1, atom2 = atom2, atom1
        if not metal_atom[atom1] or metal_atom[atom2]:
            continue
        metal, donor = int(atom_res_inds[atom1]), int(atom_res_inds[atom2])
        donor_atom = int(atom_canonical_ind[atom2])
        if metal == donor or donor_atom < 0:
            continue
        rows.append((metal, -1, donor, donor_atom))
    return numpy.array(sorted(set(rows)), dtype=numpy.int64).reshape(-1, 4)


def covalent_bonds_from_biotite(
    array, co, atom_res_inds, restype_for_res, valid_atom_mask, valid_atom_inds
):
    """Declared cross-residue bonds, including nonsequential polymer links.

    Disulfides use the dedicated variant-selection channel, preserving declared
    bonds even when the sulfur coordinates are unresolved or far apart.
    """
    if array.bonds is None:
        return numpy.zeros((0, 4), dtype=numpy.int64), None, numpy.zeros(0, dtype=bool)

    atom_canonical_ind = numpy.full(array.array_length(), -1, dtype=numpy.int64)
    atom_canonical_ind[valid_atom_mask] = valid_atom_inds

    cys_classes = frozenset(co.cys_inds.cys_co_aa_inds)
    sg_atoms = co.cys_inds.sg_by_class

    tags = (
        array.get_annotation(INFERRED_POLYMER_BOND)
        if INFERRED_POLYMER_BOND in array.get_annotation_categories()
        else numpy.full(array.array_length(), "", dtype=object)
    )
    found = []
    disulfides = []
    for atom1, atom2, order in array.bonds.as_array():
        # A metal's bonds are retyped COORDINATION upstream. They are not
        # covalent attachments, and declaring one asks for a residue type that
        # describes an ion bonded through a named connection. The sibling scan
        # below skips them for the same reason.
        if order == biotite.structure.BondType.COORDINATION:
            continue
        res1, res2 = int(atom_res_inds[atom1]), int(atom_res_inds[atom2])
        if res1 == res2:
            continue
        canonical1 = int(atom_canonical_ind[atom1])
        canonical2 = int(atom_canonical_ind[atom2])
        if canonical1 < 0 or canonical2 < 0:
            continue
        restype1, restype2 = restype_for_res[res1], restype_for_res[res2]
        if (
            restype1 in cys_classes
            and restype2 in cys_classes
            and canonical1 == sg_atoms[restype1]
            and canonical2 == sg_atoms[restype2]
        ):
            disulfides.append(tuple(sorted((res1, res2))))
            continue
        if res1 > res2:
            res1, canonical1, res2, canonical2 = res2, canonical2, res1, canonical1
        inferred = bool(tags[atom1]) and tags[atom1] == tags[atom2]
        found.append((res1, canonical1, res2, canonical2, inferred))

    found = numpy.array(sorted(set(found)), dtype=numpy.int64).reshape(-1, 5)
    return (
        found[:, :4],
        numpy.array(sorted(set(disulfides)), dtype=numpy.int64).reshape(-1, 2),
        found[:, 4].astype(bool),
    )
