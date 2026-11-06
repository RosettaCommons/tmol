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
        ("ljlk", (2, 3)),
        ("elec", (2, 3)),
        ("inverse_kinematics", (7,)),
        ("lk_ball", (2, 3, 17)),
        ("lk_ball_weighted", (2, 3, 17)),
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
