"""A ligand donor that binds a metal as its conjugate base: its type and charges.

A sulfonamide, hydroxamate or thiol coordinates as its anion, so the bond that
declares the coordination also removes a hydrogen. The atom type that form takes
comes from the ligand typer, and its charges from MMFF94, each run on the
residue's own topology less that hydrogen.
"""

import logging
from types import SimpleNamespace

from tmol.database.chemical import (
    MAX_DEPROTONATED_BRIDGE,
    _metal_donor_connections,
    _metal_leaving_hydrogen,
)
from tmol.ligand._atom_typing import _assign_tmol_atom_types
from tmol.ligand._block_type_smiles import _mol_from_block_type
from tmol.ligand._terminus_patches import _mmff94_charges

logger = logging.getLogger(__name__)

# A charge shift below this is MMFF94 round-off, not a change worth a row.
_CHARGE_TOLERANCE = 1e-6


def _deprotonating_donors(restype, atom_types):
    """{donor atom: the hydrogen it gives up} for every such donor of a residue."""
    by_name = {t.name: t for t in atom_types}
    found = {}
    for atom in restype.atoms:
        hydrogen = _metal_leaving_hydrogen(restype, atom.name, by_name)
        if hydrogen is not None:
            found[atom.name] = hydrogen
    return found


def _molecule(restype, atom_types, omit=frozenset()):
    """The residue as a molecule, and the atom name behind each molecule atom."""
    mol, source = _mol_from_block_type(
        restype, SimpleNamespace(atom_types=tuple(atom_types)), omit=omit
    )
    return mol, [restype.atoms[i].name for i in source]


def _perceived_molecule(restype, atom_types, omit=frozenset()):
    """``_molecule``, or ``(None, None)`` if RDKit rejects the perceived bonds."""
    try:
        return _molecule(restype, atom_types, omit=omit)
    except Exception as error:
        logger.warning(
            "%s: RDKit rejected the residue's bond orders (%s); bound to a metal, "
            "its donors keep neutral charges",
            restype.name,
            error,
        )
        return None, None


def _metal_bound_type(restype, atom_types, atom, hydrogen):
    """The atom type ``atom`` takes once ``hydrogen`` has left it."""
    mol, names = _molecule(restype, atom_types, omit=frozenset((hydrogen,)))
    typed = {t.index: t.atom_type for t in _assign_tmol_atom_types(mol)}
    return typed[names.index(atom)]


def _metal_deprotonation_charges(restype, atom_types, base_charges):
    """``{variant residue name: {atom: charge}}`` for each conjugate-base donor.

    The MMFF94 shift between the residue and its conjugate base is added to the
    residue's own charges, so the residue gains one negative charge however its
    charges were made. Only atoms that shift get a row; the rest fall through to
    the base. A donor bridging several metals has the same charges under each
    count's patch name.
    """
    donors = _deprotonating_donors(restype, atom_types)
    if not donors:
        return {}
    neutral, names = _perceived_molecule(restype, atom_types)
    if neutral is None:
        return {}
    neutral_charges = _mmff94_charges(neutral, neutral.GetNumAtoms())
    if neutral_charges is None:
        logger.warning(
            "%s: no MMFF94 charges; bound to a metal, its donors keep neutral charges",
            restype.name,
        )
        return {}
    neutral_charge = dict(zip(names, neutral_charges))

    rows = {}
    for atom, hydrogen in donors.items():
        anion, kept = _perceived_molecule(
            restype, atom_types, omit=frozenset((hydrogen,))
        )
        if anion is None:
            continue
        anion_charges = _mmff94_charges(anion, anion.GetNumAtoms())
        if anion_charges is None:
            logger.warning(
                "%s %s: no MMFF94 charges for its conjugate base; bound to a metal "
                "it keeps neutral charges",
                restype.name,
                atom,
            )
            continue
        shifted = {
            name: base_charges[name] + charge - neutral_charge[name]
            for name, charge in zip(kept, anion_charges)
            if abs(charge - neutral_charge[name]) > _CHARGE_TOLERANCE
        }
        for n_metals in range(1, MAX_DEPROTONATED_BRIDGE + 1):
            patch = _metal_donor_connections(atom, n_metals)[-1]
            rows[f"{restype.name}:{patch}"] = shifted
    return rows
