"""Reject inconsistent physical indexing before native topology traversal."""

import numpy as np
import pytest

from tmol_core.kinematics.pose import validate_pose_kinematics_inputs


def metadata():
    connections = np.full((2, 2, 1, 2), -1, dtype=np.int32)
    connections[0, 0, 0] = (1, 0)
    connections[0, 1, 0] = (0, 0)
    return dict(
        block_type=np.array([[0, 1], [1, -1]], dtype=np.int32),
        block_coord_offset=np.array([[0, 3], [0, -1]], dtype=np.int32),
        inter_residue_connections=connections,
        ff_edges=np.array(
            [[[2, -1, 0, 0], [3, 0, 1, 0]], [[2, -1, 0, 0], [-1, -1, -1, -1]]],
            dtype=np.int32,
        ),
        n_atoms=np.array([3, 1], dtype=np.int32),
        n_conn=np.array([1, 1], dtype=np.int32),
        polymeric_conn=np.full((2, 2), -1, dtype=np.int32),
        max_n_pose_atoms=4,
    )


def test_chemical_edges_and_permuted_coordinate_storage():
    data = metadata()
    validate_pose_kinematics_inputs(**data)
    # An unconnected polymer end can retain its opposite port number.
    data["inter_residue_connections"][1, 0, 0, 1] = 0
    validate_pose_kinematics_inputs(**data)
    data["block_coord_offset"][0] = (1, 0)
    validate_pose_kinematics_inputs(**data)


def test_polymer_fold_edge_requires_physical_connection():
    data = metadata()
    data["polymeric_conn"][:] = 0
    data["ff_edges"][0, 1, 0] = 0
    validate_pose_kinematics_inputs(**data)
    data["inter_residue_connections"][0] = -1
    with pytest.raises(ValueError, match="chain break"):
        validate_pose_kinematics_inputs(**data)


@pytest.mark.parametrize("dtype,value", [(np.uint32, 2**31), (np.int64, -(2**40))])
def test_wide_integer_metadata_cannot_wrap_native_indices(dtype, value):
    data = metadata()
    data["n_atoms"] = np.array([value, 1], dtype=dtype)
    with pytest.raises(ValueError, match="native int32"):
        validate_pose_kinematics_inputs(**data)


@pytest.mark.parametrize(
    "field,index,value,message",
    [
        ("block_type", (0, 1), 2, "palette"),
        ("block_coord_offset", (0, 1), 2, "overlap"),
        ("block_coord_offset", (0, 1), 4, "capacity"),
        ("inter_residue_connections", (0, 1, 0, 0), 1, "reciprocal"),
        ("ff_edges", (0, 1, 3), 1, "invalid source port"),
    ],
)
def test_cross_array_errors(field, index, value, message):
    data = metadata()
    data[field][index] = value
    with pytest.raises(ValueError, match=message):
        validate_pose_kinematics_inputs(**data)
