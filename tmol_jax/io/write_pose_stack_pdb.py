"""Write a :class:`PoseStack` as a PDB file.

Upstream reference: ``tmol/io/write_pose_stack_pdb.py``. The torch
version builds a numpy structured array matching
``tmol.io.pdb_parsing.atom_record_dtype`` and hands it off to the
PDB-parsing utility for string formatting. We mirror that flow with a
plain ``list[dict]`` -- avoiding the structured-dtype boilerplate --
plus a PDB-string formatter that follows the fixed-width ATOM record
spec:

    ATOM  AAAAA NNNN RRR CSSSS   XXXXXXXXYYYYYYYYZZZZZZZZOOOOOOBBBBBB          EE

"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Union

import numpy as np

# Default values matching upstream constants.
DEFAULT_ATOM_OCCUPANCY = 1.0
DEFAULT_ATOM_B_FACTOR = 0.0

# Max number of chain labels we can auto-generate ("A".."Z", "AA".."ZZ").
_MAX_AUTO_CHAIN_LABELS = 26 * 27


# --------------------------------------------------------------------------- #
# PDB atom-name justification                                                  #
# --------------------------------------------------------------------------- #


def _format_atom_name(name: str) -> str:
    """Format an atom name into the 4-column PDB slot.

    PDBv3 uses right-justification for 4-character names (e.g. hydrogen
    "HD11") but a leading space for short names that start with a
    standard element letter. This matches upstream ``format_atomn``.
    """
    name = name.strip()
    if len(name) == 0:
        return "    "
    if name[0] in ("H", "C", "N", "O", "S") and len(name) < 4:
        return " {0:<3}".format(name)
    return "{0:<4}".format(name)


# --------------------------------------------------------------------------- #
# atom_records_from_pose_stack                                                 #
# --------------------------------------------------------------------------- #


def atom_records_from_pose_stack(
    pose_stack: Any,
    pose_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build a list of per-atom dicts from a PoseStack.

    Keys mirror the upstream ``atom_record_dtype`` fields: ``record_name``,
    ``modeli``, ``chaini``, ``resi``, ``atomi``, ``model``, ``chain``,
    ``resn``, ``atomn``, ``x``, ``y``, ``z``, ``location``, ``insert``,
    ``occupancy``, ``b``, ``element``.

    When ``pose_index`` is given, only atoms of that pose are emitted;
    otherwise we emit every pose and number them with ``modeli`` /
    ``model`` as MODEL records.
    """

    pbt = pose_stack.packed_block_types
    elements = {
        atom_type.name: atom_type.element for atom_type in pbt.chem_db.atom_types
    }
    bti = np.asarray(pose_stack.block_type_ind)
    n_poses, max_n_blocks = bti.shape
    coords = np.asarray(pose_stack.coords)
    block_coord_offset = np.asarray(pose_stack.block_coord_offset, dtype=np.int32)
    n_atoms_per_bt = np.asarray(pbt.n_atoms, dtype=np.int32)

    pdb_info = getattr(pose_stack, "pdb_info", None)
    pdb_chain_id = getattr(pdb_info, "chain_id", None) if pdb_info is not None else None
    pdb_res_num = getattr(pdb_info, "res_num", None) if pdb_info is not None else None
    pdb_res_name = getattr(pdb_info, "res_name", None) if pdb_info is not None else None
    pdb_atom_record_type = (
        getattr(pdb_info, "atom_record_type", None) if pdb_info is not None else None
    )
    if pdb_res_num is not None:
        pdb_res_num = np.asarray(pdb_res_num)

    # Per-pose chain label -> int remap so MODEL-1 "A" and MODEL-2 "A"
    # each get ``chaini == 0`` local to their pose.
    records: List[Dict[str, Any]] = []

    poses_to_emit = range(n_poses) if pose_index is None else [int(pose_index)]

    for p in poses_to_emit:
        chain_label_to_int: Dict[str, int] = {}
        atom_counter = 0
        real_res_counter = 0

        for r in range(max_n_blocks):
            bt_ind = int(bti[p, r])
            if bt_ind < 0:
                continue
            bt = pbt.active_block_types[bt_ind]
            n_atoms = int(n_atoms_per_bt[bt_ind])
            resn = str(getattr(bt, "name3", "") or getattr(bt, "name", ""))[:3]

            # Chain label
            if (
                pdb_chain_id is not None
                and p < len(pdb_chain_id)
                and r < len(pdb_chain_id[p])
                and pdb_chain_id[p][r]
            ):
                chain_str = str(pdb_chain_id[p][r])
            else:
                chain_str = "A"
            chain_int = chain_label_to_int.setdefault(
                chain_str, len(chain_label_to_int)
            )

            # Residue number
            if (
                pdb_res_num is not None
                and pdb_res_num.shape[0] > p
                and pdb_res_num.shape[1] > r
                and int(pdb_res_num[p, r]) >= 0
            ):
                resi = int(pdb_res_num[p, r])
            else:
                real_res_counter += 1
                resi = real_res_counter

            # Residue name (pdb_info may hold a more specific label).
            if (
                pdb_res_name is not None
                and p < len(pdb_res_name)
                and r < len(pdb_res_name[p])
                and pdb_res_name[p][r]
            ):
                resn = str(pdb_res_name[p][r])[:3]

            start = int(block_coord_offset[p, r])
            for a in range(n_atoms):
                atom = bt.atoms[a]
                atomn = atom.name
                element = elements[atom.atom_type]

                # Determine ATOM vs HETATM record type
                record_name = "ATOM  "
                if (
                    pdb_atom_record_type is not None
                    and p < len(pdb_atom_record_type)
                    and r < len(pdb_atom_record_type[p])
                    and a < len(pdb_atom_record_type[p][r])
                ):
                    rt = str(pdb_atom_record_type[p][r][a])
                    record_name = "HETATM" if rt.upper() == "HETATM" else "ATOM  "

                atom_counter += 1
                x, y, z = coords[p, start + a]

                records.append(
                    {
                        "record_name": record_name,
                        "modeli": p,
                        "chaini": chain_int,
                        "resi": resi,
                        "atomi": atom_counter,
                        "model": str(p + 1),
                        "chain": chain_str,
                        "resn": resn,
                        "atomn": atomn,
                        "x": float(x),
                        "y": float(y),
                        "z": float(z),
                        "location": " ",
                        "insert": " ",
                        "occupancy": DEFAULT_ATOM_OCCUPANCY,
                        "b": DEFAULT_ATOM_B_FACTOR,
                        "element": element,
                    }
                )

    return records


# --------------------------------------------------------------------------- #
# PDB string formatting                                                        #
# --------------------------------------------------------------------------- #


def _format_atom_line(r: Dict[str, Any]) -> str:
    """Format a single atom record following the PDB v3 ATOM spec."""

    record_name = r.get("record_name", "ATOM  ")
    if len(record_name) < 6:
        record_name = record_name.ljust(6)[:6]
    else:
        record_name = record_name[:6]

    atomi = int(r["atomi"]) % 100000  # serial field is 5 columns
    atomn = _format_atom_name(str(r.get("atomn", "")))
    location = (str(r.get("location", "")) or " ")[:1]
    resn = str(r.get("resn", "UNK"))[:3].rjust(3)
    chain = (str(r.get("chain", "A")) or "A")[:1]
    resi = int(r["resi"]) % 10000  # 4 columns
    insert = (str(r.get("insert", "")) or " ")[:1]
    x = float(r["x"])
    y = float(r["y"])
    z = float(r["z"])
    occupancy = float(r.get("occupancy", DEFAULT_ATOM_OCCUPANCY))
    b = float(r.get("b", DEFAULT_ATOM_B_FACTOR))
    element = str(r.get("element", "")).rjust(2)[:2]

    # Standard PDB v3 ATOM record (80 columns).
    return (
        f"{record_name}"
        f"{atomi:>5d}"
        f" "
        f"{atomn}"
        f"{location:1s}"
        f"{resn:3s}"
        f" "
        f"{chain:1s}"
        f"{resi:>4d}"
        f"{insert:1s}"
        f"   "
        f"{x:>8.3f}"
        f"{y:>8.3f}"
        f"{z:>8.3f}"
        f"{occupancy:>6.2f}"
        f"{b:>6.2f}"
        f"          "
        f"{element:>2s}"
        f"\n"
    )


def _pdb_string_from_records(
    records: List[Dict[str, Any]],
    multi_model: bool,
) -> str:
    """Stitch atom records into a PDB-compliant string.

    When ``multi_model`` is True, we emit one MODEL/ENDMDL block per
    distinct ``modeli``. TER records are emitted at chain boundaries.
    """

    lines: List[str] = []
    current_model: Optional[int] = None
    current_chain: Optional[int] = None

    def _start_model(m: int) -> None:
        if multi_model:
            lines.append(f"MODEL     {m + 1:>4d}\n")

    def _end_model() -> None:
        if multi_model:
            lines.append("TER\n")
            lines.append("ENDMDL\n")

    for rec in records:
        mi = int(rec["modeli"])
        ci = int(rec["chaini"])

        if current_model is None:
            current_model = mi
            current_chain = ci
            _start_model(mi)
        elif mi != current_model:
            _end_model()
            current_model = mi
            current_chain = ci
            _start_model(mi)
        elif ci != current_chain:
            lines.append("TER\n")
            current_chain = ci

        lines.append(_format_atom_line(rec))

    if current_model is not None:
        _end_model()

    lines.append("END\n")
    return "".join(lines)


# --------------------------------------------------------------------------- #
# Biotite optional path                                                        #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #


def write_pose_stack_pdb(
    pose_stack: Any,
    path_or_file: Union[str, Path, TextIO],
    pose_index: Optional[int] = 0,
) -> None:
    """Write a PoseStack (or a single pose of a stack) to a PDB file.

    Parameters
    ----------
    pose_stack :
        The PoseStack to serialise.
    path_or_file :
        Either a filesystem path or an open text file handle.
    pose_index :
        Pose index to emit. Defaults to ``0`` (first pose). Pass
        ``None`` to emit all poses as separate MODEL records.
    """

    records = atom_records_from_pose_stack(pose_stack, pose_index=pose_index)
    multi_model = pose_index is None

    pdb_str = _pdb_string_from_records(records, multi_model=multi_model)
    if hasattr(path_or_file, "write"):
        path_or_file.write(pdb_str)
    else:
        with open(path_or_file, "w") as fh:
            fh.write(pdb_str)


__all__ = [
    "atom_records_from_pose_stack",
    "write_pose_stack_pdb",
    "DEFAULT_ATOM_OCCUPANCY",
    "DEFAULT_ATOM_B_FACTOR",
]
