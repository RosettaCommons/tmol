"""Canonical input preserves declared cycles, molecular ports and invalidity."""

from contextlib import nullcontext

import numpy as np
import pytest

from .molecules import DOMAIN_CASES


def _run(b, operation, values, **options):
    if b.framework == "jax":
        from tmol_jax.io import _native

        function = getattr(_native, "canonical_" + operation)
        args = tuple(b.module.device_put(x, b.device) for x in values)

        def run(*xs):
            return function(*xs, **options)

        result = b.module.jit(run)(*args)
        mapped = b.module.jit(b.module.vmap(run))(
            *(b.module.numpy.stack((x, x)) for x in args)
        )
        for x, y in zip(b.module.tree.leaves(result), b.module.tree.leaves(mapped)):
            np.testing.assert_array_equal(b.numpy(x), b.numpy(y[0]))
        return b.module.tree.map(b.numpy, result)
    from tmol.io.details.compiled import _compiled

    name = "wiring" if operation == "connections" else operation
    function = getattr(_compiled, "_canonical_" + name)
    result = function(
        [b.module.as_tensor(x, device=b.device) for x in values],
        next(iter(options.values())),
    )
    return tuple(map(b.numpy, result)) if isinstance(result, tuple) else b.numpy(result)


@pytest.mark.parametrize(
    "case", ("d_peptide", "connected_noncanonical", "modified_dna")
)
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_cycle_identity_and_port_reciprocity(backend, case, dtype):
    from tmol_core.io.canonical_ordering import CanonicalOrdering

    b = backend.molecule(DOMAIN_CASES[case], terms=("ref",))
    co = CanonicalOrdering.from_chemdb(b.parameter_database.chemical)
    pbt = b.pose.packed_block_types
    residue = b.pose.block_type(0, 0)
    equiv = co.restype_io_equiv_classes.index(residue.io_equiv_class)
    kinds = np.full((2, 3), equiv, np.int32)
    chains = np.zeros((2, 3), np.int32)
    up = np.asarray(co.polymer_conn_inds.up_atom_for_co_restype, np.int32)
    down = np.asarray(co.polymer_conn_inds.down_atom_for_co_restype, np.int32)
    assert up[equiv] >= 0 and down[equiv] >= 0
    coordinates = np.zeros((2, 3, co.max_n_canonical_atoms, 3), dtype)
    coordinates[:, 2, up[equiv], 0] = 1.25
    # The same atoms cannot imply a cycle through an intervening nonpolymer.
    kinds[1, 1] = co.restype_io_equiv_classes.index("ZN")
    empty = np.empty((0, 3), np.int32)
    precision = (
        b.module.enable_x64(dtype == np.float64)
        if b.framework == "jax"
        else nullcontext()
    )
    with precision:
        values = [kinds, chains, up, down, coordinates, empty]
        partners = _run(b, "cycles", values, cutoff=1.5)
        np.testing.assert_array_equal(partners, [[-1, -1, 0], [-1, -1, -1]])
        declared = np.array([[0, 2, 0]], np.int32)
        np.testing.assert_array_equal(
            _run(b, "cycles", values[:-1] + [declared], cutoff=1.5),
            -np.ones((2, 3), np.int32),
        )
        np.testing.assert_array_equal(
            _run(b, "cycles", values, cutoff=1.0), -np.ones((2, 3), np.int32)
        )
        # A real residue's native port indices determine the graph; atom order
        # and polymer chemistry may differ between peptide and nucleotide cases.
        t = list(pbt.active_block_types).index(residue)
        types = np.array([[t, t, t]], np.int32)
        polymeric = np.ones_like(types, bool)
        first = np.array([[True, False, False]])
        last = np.array([[False, False, True]])
        up_ports = np.array(
            [r.up_connection_ind for r in pbt.active_block_types], np.int32
        )
        down_ports = np.array(
            [r.down_connection_ind for r in pbt.active_block_types], np.int32
        )
        disulfide_ports = np.array(
            [r.connection_to_cidx.get("dslf", -1) for r in pbt.active_block_types],
            np.int32,
        )
        values = [
            types,
            polymeric,
            first,
            last,
            np.zeros((1, 3, 2), bool),
            up_ports,
            down_ports,
            disulfide_ports,
            empty,
            declared,
            np.empty((0, 5), np.int32),
        ]
        graph, status = _run(b, "connections", values, max_connections=pbt.max_n_conn)
        np.testing.assert_array_equal(status, 0)
        for r in range(3):
            np.testing.assert_array_equal(
                graph[0, r, up_ports[t]], [(r + 1) % 3, down_ports[t]]
            )
            np.testing.assert_array_equal(
                graph[0, r, down_ports[t]], [(r - 1) % 3, up_ports[t]]
            )
        # An invalid explicit endpoint must remain a failed construction.
        values[-1] = np.array([[0, 0, pbt.max_n_conn, 2, 0]], np.int32)
        _, status = _run(b, "connections", values, max_connections=pbt.max_n_conn)
        assert status[0] != 0
