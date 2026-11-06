"""Array transport for shared hydrogen-bond chemistry and native physics."""

import jax.numpy as jnp
import numpy as np

from tmol_core.chemical.bonds import index_bonds, pack_bond_topology
from tmol_core.scoring.hbond import (
    HBondChemicalData,
    native_hbond_parameters,
    pack_hbond_blocks,
    prepare_hbond_block,
)
from tmol_jax.score._native import (
    coordinate_energy,
    pose_arguments,
    register,
    workspace_sizes,
)
from tmol_jax.score.nonbonded import _pose_connections, _validate_native_connections


def prepare_hbond_metadata(pbt, database):
    """Consume canonical residue preparation without framework annotations."""
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


def prepare_hbond(pose, database):
    pbt = pose.packed_block_types
    _validate_native_connections(pbt)
    _, hb = prepare_hbond_metadata(pbt, database)
    bonds = pack_bond_topology(pbt.active_block_types, max_n_atoms=pbt.max_n_atoms)
    params = native_hbond_parameters(database.chemical, database.scoring.hbond)
    offsets, first, types, _ = pose_arguments(pose)
    poses, blocks = types.shape
    near, separation = _pose_connections(pose)
    arrays = (
        offsets,
        first,
        types,
        np.tile(np.arange(blocks, dtype=np.int32), poses),
        np.repeat(np.arange(poses, dtype=np.int32), blocks),
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
        15,
        pose.coords.shape[0] * pose.coords.shape[1],
        *(
            arrays[i]
            for i in (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 22)
        ),
    )
    register("tmol_hbond")
    return tuple(jnp.asarray(a) for a in (*arrays, indices))


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
