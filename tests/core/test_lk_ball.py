from types import SimpleNamespace
import numpy as np

from tmol_core.scoring.lk_ball import prepare_lk_ball_block, pack_lk_ball_blocks
from tmol_core.chemical.bonds import pack_bond_topology


def test_lk_ball_tiles_preserve_all_heavy_atoms_and_polar_priority():
    # Cover a ligand-sized residue with two full tiles and a partial last tile.
    n_atoms = 67
    polar = {0, 1, 33, 34, 66}
    hydrogens = {2, 35}
    h = np.zeros(n_atoms, np.int32)
    h[list(hydrogens)] = 1
    hb = SimpleNamespace(
        don_hvy_inds=np.array([0, 33, 66]),
        acc_inds=np.array([1, 34]),
        is_hydrogen=h,
        tile_donH_inds=np.empty((3, 32), np.int32),
    )
    params = np.zeros((n_atoms, 9), np.float32)
    params[:, 0] = np.arange(n_atoms)
    record = prepare_lk_ball_block(
        SimpleNamespace(n_atoms=n_atoms), hb, np.arange(n_atoms), params
    )
    seen = []
    for tile in range(3):
        count = record.tile_n_occluder_atoms[tile]
        n_polar = record.tile_n_polar_atoms[tile]
        local = record.tile_pol_occ_inds[tile, :count]
        assert np.all((local >= 0) & (local < 32))
        atoms = local + tile * 32
        seen.extend(atoms)
        assert set(atoms[:n_polar]) <= polar
        assert not set(atoms[n_polar:]) & polar
        np.testing.assert_array_equal(
            record.tile_lk_ball_params[tile, :count, 0], atoms
        )
        assert np.all(record.tile_pol_occ_inds[tile, count:] == -1)
    assert len(seen) == len(set(seen))
    assert set(seen) == set(range(n_atoms)) - hydrogens


def test_empty_native_lk_and_bond_metadata_have_defined_shapes():
    lk = pack_lk_ball_blocks([])
    assert lk["tile_pol_occ_inds"].shape == (0, 0, 32)
    assert lk["tile_lk_ball_params"].shape == (0, 0, 32, 9)
    bonds = pack_bond_topology([])
    assert bonds["bond_separation"].shape == (0, 0, 0)
    assert bonds["all_bonds"].shape == (0, 0, 3)
    assert bonds["atom_all_bond_ranges"].shape == (0, 0, 2)
