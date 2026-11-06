from types import SimpleNamespace as Record

import numpy as np
import pytest

from tmol_core.io.pdb_info import PDBInfo
from tmol_core.pose.concatenation import (
    concatenation_layout,
    concatenate_pdb_info,
    merged_block_types,
    widest_packed_block_types,
)


def test_chemical_extensions_preserve_identity_not_value_equality():
    first, added = Record(name="base"), Record(name="custom")
    old = Record(
        chem_db=Record(residues=[first]), n_types=1, active_block_types=[first]
    )
    grown = Record(
        chem_db=Record(residues=[first, added]),
        n_types=2,
        active_block_types=[added, first],
    )
    poses = [Record(packed_block_types=p) for p in (old, grown)]
    assert widest_packed_block_types(poses) is grown
    assert merged_block_types(poses) == [first, added]
    old.chem_db = Record(residues=[Record(name="base")])
    with pytest.raises(ValueError, match="build them from one context"):
        widest_packed_block_types(poses)


@pytest.mark.parametrize("annotations_present", [True, False])
def test_jagged_metadata_preserves_origins_annotations_and_padding(annotations_present):
    origin = {"original_atom": "ZN", "residue": 42}
    annotation_dtype = np.dtype([("sequence_index", np.int64)])
    poses = []
    for count, blocks, atoms in ((2, 1, 4), (1, 3, 7)):
        metadata = PDBInfo(
            residue_labels=np.full((count, blocks), 42),
            residue_insertion_codes=np.full((count, blocks), "A", object),
            chain_labels=np.full((count, blocks), "protein/ligand", object),
            atom_occupancy=np.full((count, atoms), 0.5, np.float32),
            atom_b_factor=np.full((count, atoms), 12, np.float32),
            metal_origins=(
                np.full((count, blocks), origin, object) if count == 2 else None
            ),
            residue_annotations=(
                np.ones((count, blocks), dtype=annotation_dtype)
                if annotations_present or count == 2
                else None
            ),
        )
        poses.append(
            Record(
                n_poses=count,
                max_n_blocks=blocks,
                coords=Record(shape=(count, atoms, 3)),
                pdb_info=metadata,
            )
        )
    layout = concatenation_layout(poses)
    assert (layout.n_poses, layout.max_n_blocks, layout.max_n_atoms) == (3, 3, 7)
    np.testing.assert_array_equal(layout.pose_offsets, [0, 2])
    combined = concatenate_pdb_info(poses, 3, layout.pose_offsets, 3)
    for pose, rows in zip(poses, (slice(0, 2), slice(2, 3))):
        for name in (
            "residue_labels",
            "residue_insertion_codes",
            "chain_labels",
            "atom_occupancy",
            "atom_b_factor",
        ):
            original = getattr(pose.pdb_info, name)
            np.testing.assert_array_equal(
                combined[name][rows, : original.shape[1]], original
            )
    assert combined["metal_origins"][0, 0] is origin
    assert combined["metal_origins"][2, 0] is None
    assert combined["residue_labels"][0, 1] == 0
    assert combined["atom_occupancy"][0, 4] == 1
    assert combined["atom_b_factor"][0, 4] == 0
    if annotations_present:
        assert combined["residue_annotations"].dtype == annotation_dtype
        assert combined["residue_annotations"][0, 1]["sequence_index"] == 0
    else:
        assert combined["residue_annotations"] is None
    combined["residue_labels"][0, 0] = 99
    assert poses[0].pdb_info.residue_labels[0, 0] == 42
