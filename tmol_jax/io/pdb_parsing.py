"""PDB parsing through Biotite, shared with the AtomArray frontend."""

from io import StringIO
from os import PathLike
from pathlib import Path

import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
import numpy as np


def _records_from_biotite(atoms):
    n = len(atoms)
    return {
        "_atom_array": atoms,
        "record_name": np.where(atoms.hetero, "HETATM", "ATOM"),
        "atom_serial": np.arange(n, dtype=np.int32),
        "atom_name": np.asarray(atoms.atom_name),
        "alt_loc": np.asarray(getattr(atoms, "altloc_id", [""] * n)),
        "res_name": np.asarray(atoms.res_name),
        "chain_id": np.asarray(atoms.chain_id),
        "res_num": np.asarray(atoms.res_id, dtype=np.int32),
        "insert": np.asarray(atoms.ins_code),
        "coords": np.asarray(atoms.coord, dtype=np.float32),
        "occupancy": np.asarray(getattr(atoms, "occupancy", np.ones(n))),
        "b_factor": np.asarray(getattr(atoms, "b_factor", np.zeros(n))),
        "model": np.zeros(n, dtype=np.int32),
    }


def parse_pdb(pdb_input):
    """Read a path, PDB text, or lines into per-atom NumPy arrays.

    For a multi-model file, use the first model, matching TMol's PDB loader.
    Biotite selects the first alternate conformation for each residue.
    Malformed structures raise the parser's error rather than being repaired.
    """
    if isinstance(pdb_input, PathLike):
        text = Path(pdb_input).read_text()
    elif isinstance(pdb_input, str):
        text = (
            Path(pdb_input).read_text()
            if "\n" not in pdb_input and Path(pdb_input).is_file()
            else pdb_input
        )
    else:
        text = "\n".join(line.rstrip("\n") for line in pdb_input)
    pdb = PDBFile.read(StringIO(text))
    atoms = pdb.get_structure(extra_fields=["atom_id", "occupancy", "b_factor"])
    if isinstance(atoms, struc.AtomArrayStack):
        atoms = atoms[0]
    # Retain actual CONECT records. include_bonds=True also invents bonds from
    # residue names, losing the declared/inferred distinction needed for gaps.
    atoms.bonds = pdb._get_bonds(atoms.atom_id)
    return _records_from_biotite(atoms)


def pose_stack_from_pdb(
    pdb_input,
    *,
    canonical_ordering=None,
    packed_block_types=None,
    missing_density_distance_threshold=2.4,
    **kwargs,
):
    """Load a molecular PDB using its chemical ordering and residue palette."""
    from .canonical_ordering import (
        canonical_form_from_pdb,
        default_canonical_ordering,
        default_packed_block_types,
    )
    from .pose_stack_construction import pose_stack_from_canonical_form

    co = canonical_ordering or default_canonical_ordering()
    pbt = packed_block_types or default_packed_block_types()
    return pose_stack_from_canonical_form(
        dict(
            canonical_form_from_pdb(
                pdb_input,
                co,
                missing_density_distance_threshold=missing_density_distance_threshold,
            ),
            canonical_ordering=co,
        ),
        pbt,
        **kwargs,
    )
