"""LK-ball type, residue and padded topology preparation for native execution."""

from dataclasses import dataclass
from collections.abc import Mapping
import numpy
from .ljlk import lk_isotropic_coefficients
from .tiling import tile_subset_indices

LK_BALL_TYPE_FIELDS = (
    "lj_radius",
    "lk_dgfree",
    "lk_lambda",
    "lk_volume",
    "is_donor",
    "is_hydroxyl",
    "is_polarh",
    "is_acceptor",
    "is_carbon_lk",
)
LK_BALL_GLOBAL_FIELDS = (
    "lj_hbond_dis",
    "lj_hbond_OH_donor_dis",
    "lj_hbond_hdis",
    "lkb_water_dist",
    "max_dis",
)
WATER_GEN_GLOBAL_FIELDS = (
    "lkb_water_dist",
    "lkb_water_angle_sp2",
    "lkb_water_angle_sp3",
    "lkb_water_angle_ring",
)


def native_lk_ball_type_parameters(parameters):
    p = {
        name: numpy.asarray(parameters[name], dtype=numpy.float32)
        for name in LK_BALL_TYPE_FIELDS
    }
    coeff, inv_lambda2 = lk_isotropic_coefficients(p["lk_dgfree"], p["lk_lambda"])
    return numpy.stack(
        (
            p["lj_radius"],
            coeff,
            inv_lambda2,
            p["lk_volume"],
            p["is_donor"],
            p["is_hydroxyl"],
            p["is_polarh"],
            p["is_acceptor"],
            p["is_carbon_lk"],
        ),
        axis=1,
    )


def _globals(parameters, fields):
    get = (
        parameters.__getitem__
        if isinstance(parameters, Mapping)
        else lambda name: getattr(parameters, name)
    )
    return numpy.stack(
        [
            numpy.atleast_1d(numpy.asarray(get(name), dtype=numpy.float32))
            for name in fields
        ],
        axis=1,
    )


def native_lk_ball_global_parameters(parameters):
    return _globals(parameters, LK_BALL_GLOBAL_FIELDS)


def native_water_gen_global_parameters(parameters):
    """Water geometry fields; raw shared database angles are already radians."""
    return _globals(parameters, WATER_GEN_GLOBAL_FIELDS)


@dataclass(frozen=True)
class LKBallBlockData:
    tile_n_polar_atoms: numpy.ndarray
    tile_n_occluder_atoms: numpy.ndarray
    tile_pol_occ_inds: numpy.ndarray
    tile_lk_ball_params: numpy.ndarray


def prepare_lk_ball_block(
    block_type, hbbt_params, atom_type_indices, type_parameters, tile_size=32
):
    """Consume structural H-bond metadata; preserve polar-first tile ordering."""
    n_tiles = hbbt_params.tile_donH_inds.shape[0]

    atom_is_polar = numpy.full((block_type.n_atoms,), False, dtype=bool)
    atom_is_polar[hbbt_params.don_hvy_inds] = True
    atom_is_polar[hbbt_params.acc_inds] = True
    polar_inds = numpy.nonzero(atom_is_polar)[0].astype(numpy.int32)

    tiled_polar_orig_inds, tile_n_polar = tile_subset_indices(
        polar_inds, tile_size, True, block_type.n_atoms - 1
    )
    tiled_polar_orig_inds = tiled_polar_orig_inds.reshape(n_tiles, tile_size)

    is_tiled_polar = tiled_polar_orig_inds != -1
    tiled_polars = numpy.full((n_tiles, tile_size), -1, dtype=numpy.int32)
    tiled_polars[is_tiled_polar] = polar_inds

    # ASSUMPTION! either h or hvy; change when VRTS are added!
    # Grace: lk parameters for VRTs should not affect score even if included,
    # it would just be slightly inefficient
    atom_is_heavy = numpy.invert(hbbt_params.is_hydrogen == 1)

    # "apolar" here means "does not build waters"; e.g. proline's N would be
    # considered apolar.
    atom_is_heavy_apolar = numpy.logical_and(atom_is_heavy, numpy.invert(atom_is_polar))
    heavy_apolar_inds = numpy.nonzero(atom_is_heavy_apolar)[0].astype(numpy.int32)

    tiled_heavy_apolar_orig_inds, tile_n_apolar = tile_subset_indices(
        heavy_apolar_inds, tile_size, True, block_type.n_atoms - 1
    )
    tiled_heavy_apolar_orig_inds = tiled_heavy_apolar_orig_inds.reshape(
        n_tiles, tile_size
    )
    is_tiled_heavy_apolar = tiled_heavy_apolar_orig_inds != -1
    tiled_heavy_apolar = numpy.full((n_tiles, tile_size), -1, dtype=numpy.int32)
    tiled_heavy_apolar[is_tiled_heavy_apolar] = heavy_apolar_inds

    # now lets combine the heavy polar indices and the heavy non-polar indices
    tiled_pols_and_occs = numpy.copy(tiled_polars)
    tile_n_occ = numpy.copy(tile_n_polar)
    for i in range(n_tiles):
        tile_n_occ[i] = tile_n_polar[i] + tile_n_apolar[i]
        r = slice(tile_n_polar[i], tile_n_occ[i])
        tiled_pols_and_occs[i, r] = tiled_heavy_apolar[i, : tile_n_apolar[i]]
    is_pol_or_occ = tiled_pols_and_occs != -1

    # ok, now let's collect the properties of the atoms in this block
    # needed for LKBallTypeParams (see properties/params.hh)
    bt_lk_ball_at_params = type_parameters[atom_type_indices]
    tiled_bt_lk_ball_at_params = numpy.zeros(
        (n_tiles, tile_size, 9), dtype=numpy.float32
    )
    tiled_bt_lk_ball_at_params[is_pol_or_occ] = bt_lk_ball_at_params[
        tiled_pols_and_occs[is_pol_or_occ]
    ]

    # The kernel expects tile_pol_occ_inds to contain tile-local atom
    # indices (0..tile_size-1) — it reconstructs the full block-atom
    # index as `pol_start + pol_tile_ind`. Until here we have been
    # tracking full block-atom indices so the per-atom params lookup
    # above works; now convert to tile-local for the output tensor.
    tile_pol_occ_inds = numpy.where(
        tiled_pols_and_occs != -1,
        tiled_pols_and_occs % tile_size,
        -1,
    ).astype(numpy.int32)

    bt_lk_ball_params = LKBallBlockData(
        tile_n_polar_atoms=tile_n_polar,
        tile_n_occluder_atoms=tile_n_occ,
        tile_pol_occ_inds=tile_pol_occ_inds,
        tile_lk_ball_params=tiled_bt_lk_ball_at_params,
    )
    return bt_lk_ball_params


def pack_lk_ball_blocks(blocks, n_tiles=None, tile_size=32):
    """Pack arbitrary residue records; zero-padding and local-index sentinels match native kernels."""
    blocks = tuple(blocks)
    n_types = len(blocks)
    if n_tiles is None:
        n_tiles = max((b.tile_n_polar_atoms.shape[0] for b in blocks), default=0)
    tile_n_polar_atoms = numpy.full((n_types, n_tiles), 0, dtype=numpy.int32)
    tile_n_occluder_atoms = numpy.full((n_types, n_tiles), 0, dtype=numpy.int32)
    tile_pol_occ_inds = numpy.full((n_types, n_tiles, tile_size), -1, dtype=numpy.int32)
    tile_lk_ball_params = numpy.full(
        (n_types, n_tiles, tile_size, 9), 0, dtype=numpy.float32
    )

    for i, i_lkbp in enumerate(blocks):
        i_n_tiles = i_lkbp.tile_n_polar_atoms.shape[0]

        tile_n_polar_atoms[i, :i_n_tiles] = i_lkbp.tile_n_polar_atoms
        tile_n_occluder_atoms[i, :i_n_tiles] = i_lkbp.tile_n_occluder_atoms
        tile_pol_occ_inds[i, :i_n_tiles] = i_lkbp.tile_pol_occ_inds
        tile_lk_ball_params[i, :i_n_tiles] = i_lkbp.tile_lk_ball_params

    return dict(
        tile_n_polar_atoms=tile_n_polar_atoms,
        tile_n_occluder_atoms=tile_n_occluder_atoms,
        tile_pol_occ_inds=tile_pol_occ_inds,
        tile_lk_ball_params=tile_lk_ball_params,
    )
