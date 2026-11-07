"""Array transport for shared hydrogen-bond chemistry and native physics."""

import jax
import numpy as np

from tmol_core.chemical.bonds import index_bonds, pack_bond_topology
from tmol_core.scoring.hbond import (
    HBondChemicalData,
    native_hbond_parameters,
    pack_hbond_blocks,
    prepare_hbond_block,
)
from tmol_jax.score.energy_term import prepared_palette
from tmol_jax.score._native import (
    coordinate_energy,
    pose_arguments,
    register,
    workspace_sizes,
)
from tmol_jax.score.nonbonded import _pose_connections, _validate_native_connections


def prepare_hbond_metadata(pbt, database):
    """Reuse canonical chemistry only for its live parameter/palette owner."""
    return prepared_palette(
        pbt, database, "hbond", lambda: _hbond_metadata(pbt, database)
    )


def prepare_bond_metadata(pbt):
    return prepared_palette(
        pbt,
        pbt.chem_db,
        "bonds",
        lambda: pack_bond_topology(pbt.active_block_types, max_n_atoms=pbt.max_n_atoms),
    )


def _hbond_metadata(pbt, database):
    chemistry = HBondChemicalData.from_database(
        database.chemical, database.scoring.hbond
    )
    blocks = []
    for block in pbt.active_block_types:
        rows = np.zeros((block.bond_indices.shape[0], 3), dtype=np.int32)
        rows[:, 1:] = block.bond_indices
        bonds = index_bonds(rows, minlength=block.n_atoms)
        blocks.append(
            prepare_hbond_block(
                block, chemistry, bonds["bonds"][0], bonds["bond_spans"][0]
            )
        )
    return blocks, pack_hbond_blocks(blocks, max_n_atoms=pbt.max_n_atoms)


def prepare_hbond(pose, database, *, rotamers=None):
    pbt = pose.packed_block_types
    _validate_native_connections(pbt)
    _, hb = prepare_hbond_metadata(pbt, database)
    bonds = prepare_bond_metadata(pbt)
    params = prepared_palette(
        pbt,
        database,
        "hbond_parameters",
        lambda: native_hbond_parameters(database.chemical, database.scoring.hbond),
    )
    offsets, first, types, _ = pose_arguments(pose)
    if rotamers is not None:
        offsets = rotamers.coord_offset_for_rot
        first = rotamers.rot_offset_for_block
        types = rotamers.first_rot_block_type
    poses, blocks = types.shape
    near, separation = _pose_connections(pose)
    arrays = (
        offsets,
        first,
        types,
        (
            np.tile(np.arange(blocks, dtype=np.int32), poses)
            if rotamers is None
            else rotamers.block_ind_for_rot
        ),
        (
            np.repeat(np.arange(poses, dtype=np.int32), blocks)
            if rotamers is None
            else rotamers.pose_for_rot
        ),
        pose.inter_residue_connections,
        near,
        separation,
        pbt.n_atoms,
        pbt.n_conn,
        pbt.conn_atoms,
        bonds["n_all_bonds"],
        bonds["all_bonds"],
        bonds["atom_all_bond_ranges"],
        hb["is_hydrogen"],
        bonds["bond_separation"],
        *(
            hb[name]
            for name in (
                "tile_n_donH",
                "tile_n_acc",
                "tile_donH_inds",
                "tile_acc_inds",
                "tile_donorH_type",
                "tile_acceptor_type",
                "tile_acceptor_hybridization",
            )
        ),
        params["pair_param_table"],
        # Preserve canonical double coefficients even when JAX x64 is disabled.
        np.ascontiguousarray(params["pair_poly_table"]).view(np.uint8),
        params["global_param_table"],
    )
    from tmol_jax._host import invoke

    # Connectivity and chemical types are fixed for a rendered scorer. Execute
    # the same native base planner as Torch once, then gather changing coordinates
    # on the execution device. -1 is unused; -2 is an active unresolved base.
    (indices,) = invoke(
        15 if rotamers is None else 16,
        (
            pose.coords.shape[0] * pose.coords.shape[1]
            if rotamers is None
            else rotamers.coords.shape[0]
        ),
        *(
            arrays[i]
            for i in (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 22)
        ),
        *((rotamers.block_type_ind_for_rot,) if rotamers is not None else ()),
    )
    register("tmol_hbond" if rotamers is None else "tmol_rotamer_hbond")
    device = pose.coords.device
    chemical_arrays = prepared_palette(
        pbt,
        database,
        "hbond",
        lambda: jax.device_put(arrays[8:], device=device),
        device=device,
    )
    topology = jax.device_put(
        (
            *arrays[:8],
            indices,
            *((rotamers.block_type_ind_for_rot,) if rotamers is not None else ()),
        ),
        device=device,
    )
    return (*topology[:8], *chemical_arrays, *topology[8:])


def hbond_energy(coords, parameters):
    return coordinate_energy(
        "tmol_hbond",
        coords,
        parameters,
        workspace_bytes=workspace_sizes(
            "tmol_hbond",
            (coords.shape[0], parameters[2].shape[1], coords.shape[1]),
            coords.dtype.itemsize * 8,
        ),
    )
