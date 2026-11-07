"""Host execution protocol consumed by common topology preparation."""

from functools import partial
from tmol_jax._host import invoke

_OPERATIONS = {
    "get_kfo_indices_for_atoms": 0,
    "get_kfo_atom_parents": 1,
    "get_children": 2,
    "get_id_and_frame_xyz": 3,
    "calculate_ff_edge_delays": 4,
    "get_jump_atom_indices": 5,
    "get_block_parent_connectivity_from_toposort": 6,
    "get_scans2": 7,
    "create_minimizer_map": 8,
    "assemble_pose": 10,
    "inverse_kinematics": 11,
}


class NativePosePlanner:
    """Host execution protocol consumed by shared topology preparation."""

    def __getattr__(self, name):
        try:
            operation = _OPERATIONS[name]
        except KeyError as error:
            raise AttributeError(name) from error
        return partial(invoke, operation)
