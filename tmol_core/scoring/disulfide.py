"""Disulfide connection identities and chirality-specific native parameters."""

from enum import Enum
from collections.abc import Mapping
import numpy


class DisulfideParameter(Enum):
    """Describe the C++ parameter struct layout and D-residue transformation."""

    D_LOCATION = ("d_location", False)
    D_SCALE = ("d_scale", False)
    D_SHAPE = ("d_shape", False)
    A_LOG_A = ("a_logA", False)
    A_KAPPA = ("a_kappa", False)
    A_MU = ("a_mu", False)
    DSS_LOG_A1 = ("dss_logA1", False)
    DSS_KAPPA1 = ("dss_kappa1", False)
    DSS_MU1 = ("dss_mu1", True)
    DSS_LOG_A2 = ("dss_logA2", False)
    DSS_KAPPA2 = ("dss_kappa2", False)
    DSS_MU2 = ("dss_mu2", True)
    DCS_LOG_A1 = ("dcs_logA1", False)
    DCS_MU1 = ("dcs_mu1", True)
    DCS_KAPPA1 = ("dcs_kappa1", False)
    DCS_LOG_A2 = ("dcs_logA2", False)
    DCS_MU2 = ("dcs_mu2", True)
    DCS_KAPPA2 = ("dcs_kappa2", False)
    DCS_LOG_A3 = ("dcs_logA3", False)
    DCS_MU3 = ("dcs_mu3", True)
    DCS_KAPPA3 = ("dcs_kappa3", False)
    WT_DIH_SS = ("wt_dih_ss", False)
    WT_DIH_CS = ("wt_dih_cs", False)
    WT_ANG = ("wt_ang", False)
    WT_LEN = ("wt_len", False)
    SHIFT = ("shift", False)
    DSS_MIXED_LOG_A1 = ("dss_mixed_logA1", False)
    DSS_MIXED_KAPPA1 = ("dss_mixed_kappa1", False)
    DSS_MIXED_MU1 = ("dss_mixed_mu1", False)
    DSS_MIXED_LOG_A2 = ("dss_mixed_logA2", False)
    DSS_MIXED_KAPPA2 = ("dss_mixed_kappa2", False)
    DSS_MIXED_MU2 = ("dss_mixed_mu2", False)
    CHIRALITY = (None, True)

    def __init__(self, attribute: str | None, negated_for_d: bool):
        self.attribute = attribute
        self.negated_for_d = negated_for_d


MIRRORED_PARAMETER_SIGNS = numpy.asarray(
    [-1.0 if parameter.negated_for_d else 1.0 for parameter in DisulfideParameter],
    dtype=numpy.float32,
)


def disulfide_connections(block_type):
    connections = numpy.array([], dtype=numpy.int32)
    if "dslf" in block_type.connection_to_cidx:
        connections = numpy.append(connections, [block_type.connection_to_cidx["dslf"]])
    return connections


def pack_disulfide_connections(block_types, max_n_conn=None, block_connections=None):
    blocks = tuple(block_types)
    if max_n_conn is None:
        max_n_conn = max((len(b.connections) for b in blocks), default=0)
    connections = numpy.zeros((len(blocks), max_n_conn), dtype=bool)
    if block_connections is None:
        block_connections = map(disulfide_connections, blocks)
    for i, indices in enumerate(block_connections):
        for connection in indices:
            connections[i, connection] = True
    mirrored = numpy.asarray(
        [b.properties.polymer.sidechain_chirality == "d" for b in blocks], dtype=bool
    )
    return connections, mirrored


def disulfide_parameter_signs(is_mirrored):
    """Common stereochemical transformation; frontends may multiply on device."""
    mirrored = numpy.asarray(is_mirrored, dtype=bool)
    signs = numpy.ones((mirrored.size, len(DisulfideParameter)), dtype=numpy.float32)
    signs[mirrored] = MIRRORED_PARAMETER_SIGNS
    return signs


def native_disulfide_parameters(parameters, is_mirrored):
    """Return one canonical 33-field float32 row per residue type.

    Input is a database global-parameter record, a mapping, or the 32 stored
    parameter values in DisulfideParameter order (chirality is derived here).
    Every disulfide half keeps its own stereochemistry, including mixed L/D.
    """
    fields = tuple(p for p in DisulfideParameter if p.attribute is not None)
    if isinstance(parameters, numpy.ndarray):
        values = numpy.asarray(parameters, dtype=numpy.float32).reshape(-1)
        if values.size != len(fields):
            raise ValueError("Disulfide requires 32 global parameter values")
    else:
        get = (
            parameters.__getitem__
            if isinstance(parameters, Mapping)
            else lambda name: getattr(parameters, name)
        )
        values = numpy.asarray(
            [get(p.attribute) for p in fields], dtype=numpy.float32
        ).reshape(-1)
    row = numpy.concatenate((values, numpy.ones(1, dtype=numpy.float32)))
    return row * disulfide_parameter_signs(is_mirrored)
