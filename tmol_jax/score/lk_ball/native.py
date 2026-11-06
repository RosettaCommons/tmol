"""Native LK-ball operand layout from shared host chemistry and water metadata."""

import jax.numpy as jnp
import numpy as np

from tmol_core.chemical.bonds import pack_bond_topology
from tmol_core.scoring.atom_types import atom_type_parameters
from tmol_core.scoring.ljlk import LJLK_FIELDS, ljlk_type_parameters
from tmol_core.scoring.lk_ball import (
    native_lk_ball_global_parameters,
    native_lk_ball_type_parameters,
    native_water_gen_global_parameters,
    pack_lk_ball_blocks,
    prepare_lk_ball_block,
)
from tmol_jax.score._native import (
    coordinate_energy,
    pose_arguments,
    register,
    workspace_sizes,
)
from tmol_jax.score.nonbonded import (
    _pose_connections,
    _validate_native_connections,
    _validate_atom_type_order,
)
from tmol_jax.score.hbond.native import prepare_hbond_metadata


def prepare_lk_ball(pose, database):
    """Consume common H-bond records; all scientific preparation stays shared."""
    pbt = pose.packed_block_types
    hbond_blocks, hbond_packed = prepare_hbond_metadata(pbt, database)
    _validate_native_connections(pbt)
    types = pbt.active_block_types
    names, flags = atom_type_parameters(database.chemical.atom_types)
    _validate_atom_type_order(pbt, names)
    lj = ljlk_type_parameters(names, database.scoring.ljlk)
    for block in types:
        lj.validate_block_type(block)
    parameters = dict(zip(LJLK_FIELDS, lj.values.T))
    parameters.update(flags)
    parameters["is_carbon_lk"] = lj.is_carbon_lk
    native_types = native_lk_ball_type_parameters(parameters)
    indices = np.asarray(pbt.atom_types)
    lk = pack_lk_ball_blocks(
        [
            prepare_lk_ball_block(block, hb, indices[i, : block.n_atoms], native_types)
            for i, (block, hb) in enumerate(zip(types, hbond_blocks, strict=True))
        ]
    )
    bonds = pack_bond_topology(types, max_n_atoms=pbt.max_n_atoms)
    offsets, first, block_types, _ = pose_arguments(pose)
    poses, blocks = block_types.shape
    near, separation = _pose_connections(pose)
    globals_ = database.scoring.ljlk.global_parameters
    arrays = (
        offsets,
        np.repeat(np.arange(poses, dtype=np.int32), blocks),
        np.tile(np.arange(blocks, dtype=np.int32), poses),
        block_types,
        first,
        pose.inter_residue_connections,
        pbt.n_atoms,
        pbt.n_conn,
        pbt.conn_atoms,
        bonds["n_all_bonds"],
        bonds["all_bonds"],
        bonds["atom_all_bond_ranges"],
        *(
            hbond_packed[name]
            for name in (
                "is_hydrogen",
                "tile_n_donH",
                "tile_n_acc",
                "tile_donH_inds",
                "tile_donH_hvy_inds",
                "tile_which_donH_of_donH_hvy",
                "tile_acc_inds",
                "tile_acceptor_hybridization",
                "tile_acceptor_n_attached_H",
            )
        ),
        native_water_gen_global_parameters(globals_),
        np.asarray(globals_.lkb_water_tors_sp2, dtype=np.float32),
        np.asarray(globals_.lkb_water_tors_sp3, dtype=np.float32),
        np.asarray(globals_.lkb_water_tors_ring, dtype=np.float32),
        near,
        separation,
        bonds["bond_separation"],
        lk["tile_n_polar_atoms"],
        lk["tile_n_occluder_atoms"],
        lk["tile_pol_occ_inds"],
        lk["tile_lk_ball_params"],
        native_lk_ball_global_parameters(globals_),
    )
    register("tmol_lk_ball")
    return tuple(jnp.asarray(a) for a in arrays), float(globals_.max_dis)


def lk_ball_energy(coords, state, weights=None):
    parameters, reach = state
    target = "tmol_lk_ball" if weights is None else "tmol_lk_ball_weighted"
    if weights is not None:
        register(target)
        parameters += (jnp.asarray(weights, dtype=coords.dtype),)
    return coordinate_energy(
        target,
        coords,
        parameters,
        channels=4 if weights is None else 1,
        workspace_bytes=workspace_sizes(
            target,
            (coords.shape[0], parameters[3].shape[1], coords.shape[1]),
            coords.dtype.itemsize * 8,
        ),
        attributes={"reach": np.float64(reach)},
    )
