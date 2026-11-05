from types import SimpleNamespace

import numpy as np

from tmol_core.scoring.hbond import (
    HBondChemicalData,
    prepare_hbond_block,
    pack_hbond_blocks,
)


def test_donor_heavy_atom_survives_cross_tile_hydrogen_and_padding():
    # The bonded hydrogen is in tile zero; its heavy donor is beyond uint8.
    atom_types = ["C"] * 300
    atom_types[1] = "H"
    atom_types[299] = "O"
    residue = SimpleNamespace(
        name="wide_hydroxyl",
        n_atoms=len(atom_types),
        atoms=[
            SimpleNamespace(name=str(i), atom_type=t) for i, t in enumerate(atom_types)
        ],
    )
    chemistry = HBondChemicalData(
        ("C", "H", "O", None),
        np.array([0, 0, 2, 0], dtype=np.int32),
        np.array([0, 1, 0, 0], dtype=np.int32),
        np.array([-1, -1, 0, -1]),
        np.array([-1, -1, 0, -1]),
    )
    bonds = np.array([[1, 299], [299, 1]], dtype=np.int32)
    spans = np.zeros((300, 2), dtype=np.int32)
    spans[1] = (0, 1)
    spans[299] = (1, 2)
    block = prepare_hbond_block(residue, chemistry, bonds, spans)
    assert block.tile_donH_inds[0, 0] == 1
    assert block.tile_donH_hvy_inds[0, 0] == 299
    assert block.tile_acceptor_n_attached_H[9, 0] == 1
    assert block.tile_acc_inds[9, 0] == 11
    packed = pack_hbond_blocks([block], max_n_atoms=352)
    assert packed["tile_n_donH"].sum() == 1
    assert packed["tile_n_acc"].sum() == 1
    assert (packed["tile_donH_hvy_inds"][:, 10] == -1).all()
    assert not packed["is_hydrogen"][:, 300:].any()


def test_empty_hbond_palette_preserves_native_extents():
    arrays = pack_hbond_blocks(())
    assert arrays["tile_donH_inds"].shape == (0, 0, 32)
    assert arrays["is_hydrogen"].shape == (0, 0)
    assert all(a.dtype == np.int32 for a in arrays.values())
