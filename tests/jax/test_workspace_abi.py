"""Exported native layout queries reject invalid extents without truncation."""

import ctypes

import pytest

from tmol_jax.score._native import libraries


@pytest.mark.parametrize(
    "target,dimensions",
    (
        ("dunbrack", (3, 7)),
        ("nonbonded", (2, 3)),
        ("nonbonded_weighted", (2, 3)),
        ("nonbonded_precise", (2, 3, 17)),
        ("nonbonded_weighted_precise", (2, 3, 17)),
        ("ljlk", (2, 3)),
        ("ljlk_precise", (2, 3, 17)),
        ("elec", (2, 3)),
        ("inverse_kinematics", (7,)),
        ("lk_ball", (2, 3, 17)),
        ("lk_ball_weighted", (2, 3, 17)),
        ("lk_ball_precise", (2, 3, 17)),
        ("lk_ball_weighted_precise", (2, 3, 17)),
        ("hbond", (2, 3, 17)),
        ("rotamer_lk_ball", (2, 3, 17)),
        ("rotamer_hbond", (2, 3, 17)),
        ("graph_prefix", (7,)),
    ),
)
def test_workspace_query_precision_and_invalid_extents(target, dimensions):
    query = getattr(libraries()["cpu"], "tmol_" + target + "_workspace_bytes")
    query.argtypes = [ctypes.c_int64] * len(dimensions) + [ctypes.c_int] * 2
    query.restype = ctypes.c_uint64
    invalid = (1 << 64) - 1
    for derivative in (0, 1):
        for bits in (32, 64):
            size = query(*dimensions, bits, derivative)
            if target == "graph_prefix" and bits == 32:
                assert size == invalid
                continue
            assert 0 < size < invalid
            for axis in range(len(dimensions)):
                for bad in (-1, 1 << 62):
                    broken = list(dimensions)
                    broken[axis] = bad
                    assert query(*broken, bits, derivative) == invalid
        for bits in (0, 16, 128):
            assert query(*dimensions, bits, derivative) == invalid
    if target != "graph_prefix":
        assert query(*dimensions, 64, 0) >= query(*dimensions, 32, 0)


@pytest.mark.parametrize(
    "target", ("tmol_nonbonded", "tmol_nonbonded_weighted", "tmol_ljlk")
)
def test_wide_nonbonded_workspace_only_changes_float_derivatives(target):
    from tmol_jax.score._native import workspace_sizes

    for bits in (32, 64):
        old = workspace_sizes(target, (2, 3), bits)
        precise = workspace_sizes(target + "_precise", (2, 3, 17), bits)
        assert old[0] == precise[0]
        if bits == 64:
            assert old == precise
        else:
            assert precise[1] > precise[0]
    # Reject layouts whose derivative element dispatch would narrow to int32.
    with pytest.raises(ValueError, match="workspace layout"):
        workspace_sizes(target + "_precise", (1 << 20, 1, 1 << 20), 32)


def test_cartesian_accumulator_workspace_precision_and_overflow():
    from tmol_jax.score._native import workspace_sizes

    target = "tmol_cartbonded_precise"
    energy, derivative = workspace_sizes(target, (2, 17), 32)
    assert derivative > energy > 0
    assert workspace_sizes(target, (2, 17), 64) == (0, 0)
    for dimensions in ((-1, 17), (1 << 62, 17), (1 << 20, 1 << 20)):
        with pytest.raises(ValueError, match="workspace layout"):
            workspace_sizes(target, dimensions, 32)
