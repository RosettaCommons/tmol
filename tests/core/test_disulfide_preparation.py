from types import SimpleNamespace

import numpy as np

from tmol_core.scoring.disulfide import (
    DisulfideParameter,
    native_disulfide_parameters,
    pack_disulfide_connections,
)


def test_disulfide_port_identity_and_mixed_chirality_rows():
    def block(chirality):
        return SimpleNamespace(
            connection_to_cidx={"other": 0, "dslf": 299},
            connections=tuple(range(300)),
            properties=SimpleNamespace(
                polymer=SimpleNamespace(sidechain_chirality=chirality)
            ),
        )

    connections, mirrored = pack_disulfide_connections([block("l"), block("d")])
    assert np.argwhere(connections).tolist() == [[0, 299], [1, 299]]
    values = np.arange(1, 33, dtype=np.float32)
    rows = native_disulfide_parameters(values, mirrored)
    assert rows.shape == (2, 33)
    assert rows.dtype == np.float32
    columns = {parameter.attribute: i for i, parameter in enumerate(DisulfideParameter)}
    assert rows[0, -1] == 1 and rows[1, -1] == -1
    for name in ("dss_mu1", "dss_mu2", "dcs_mu1", "dcs_mu2", "dcs_mu3"):
        i = columns[name]
        assert rows[0, i] == -rows[1, i]
    for name in ("dss_mixed_mu1", "dss_mixed_mu2", "d_location", "wt_len"):
        i = columns[name]
        assert rows[0, i] == rows[1, i]


def test_empty_disulfide_palette_has_valid_parameter_extent():
    connections, mirrored = pack_disulfide_connections(())
    assert connections.shape == (0, 0)
    assert native_disulfide_parameters(np.zeros(32), mirrored).shape == (0, 33)


def test_disulfide_search_preserves_declared_pairs_and_reuses_metadata():
    from tmol_core.io.disulfides import prepare_disulfide_search, resolve_disulfides

    types = np.array([[3, 8, 3, 8], [-1, -1, -1, -1]], dtype=np.int32)
    plan = prepare_disulfide_search((3, 8), types, [[0, 0, 2]])
    initial = plan.variants.copy()
    for dtype in (np.float32, np.float64):
        coords = np.zeros((2, 4, 1, 3), dtype=dtype)
        coords[0, :, 0, 0] = [0, 1.8, 2.2, 4]
        observed = coords.copy()
        for _ in range(2):
            edges, variants = resolve_disulfides(plan, coords, 0)
            # Declared 0--2 wins over the shorter 0--1 distance. The remaining
            # mixed-class pair is inferred, while padding has no variant.
            np.testing.assert_array_equal(edges, [[0, 0, 2], [0, 1, 3]])
            np.testing.assert_array_equal(variants, [[1, 1, 1, 1], [0, 0, 0, 0]])
            assert edges.dtype == np.int64 and variants.dtype == np.int32
        np.testing.assert_array_equal(coords, observed)
        np.testing.assert_array_equal(plan.variants, initial)


def test_disulfide_search_can_avoid_coordinates_and_rejects_repeated_endpoints():
    import pytest
    from tmol_core.io.disulfides import prepare_disulfide_search, resolve_disulfides

    types = np.array([[3, 8, 3, 8]], dtype=np.int32)
    plan = prepare_disulfide_search((3, 8), types, [[0, 0, 2]], False)
    assert not plan.search
    edges, variants = resolve_disulfides(plan, None, 0)
    np.testing.assert_array_equal(edges, [[0, 0, 2]])
    np.testing.assert_array_equal(variants, [[1, 0, 1, 0]])
    with pytest.raises(ValueError, match="only one declared disulfide"):
        prepare_disulfide_search((3, 8), types, [[0, 0, 2], [0, 1, 2]])
