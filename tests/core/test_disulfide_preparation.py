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
