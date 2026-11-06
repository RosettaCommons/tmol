"""Canonical input retains chemistry, atom identity, topology and physical scores."""

import numpy as np
import pytest

from .molecules import DOMAIN_CASES, BACKBONE_CASES, MolecularCase
from .metal_molecules import METAL_CASES

CASES = {
    key: DOMAIN_CASES[key]
    for key in (
        "ligand",
        "noncanonical",
        "connected_noncanonical",
        "d_peptide",
        "dna",
        "rna",
        "modified_dna",
        "mixed_batch",
    )
}
CASES["cyclic"] = BACKBONE_CASES["cyclic"]
CASES.update({"metal_" + key: value for key, value in METAL_CASES.items()})
CASES.update(
    {
        "protonation_" + name: MolecularCase((), ((name,),), ((1,),))
        for name in ("CYS_DEP", "DCYS_DEP", "TYR_DEP", "DTYR_DEP", "HIS_POS")
    }
)


def canonical_input(case, database):
    from tmol_core.io.canonical_ordering import CanonicalOrdering
    from tmol_core.chemical.restypes import refined_from_raw

    co = CanonicalOrdering.from_chemdb(database.chemical)
    raw = {r.name: r for r in database.chemical.residues}
    rows = case.blocks(database)
    shape = (len(rows), max(map(len, rows)))
    coords = np.full((*shape, co.max_n_canonical_atoms, 3), np.nan, np.float32)
    types = np.full(shape, -1, np.int32)
    chains = np.full(shape, -1, np.int32)
    gaps = np.zeros((*shape, 2), bool)
    for p, row in enumerate(rows):
        start = 0
        for chain, length in enumerate(case.chain_lengths[p]):
            chains[p, start : start + length] = chain
            gaps[p, start, 0] = gaps[p, start + length - 1, 1] = True
            start += length
        for r, (name, names, xyz) in enumerate(row):
            residue = refined_from_raw(raw[name])
            equiv = residue.io_equiv_class
            types[p, r] = co.restype_io_equiv_classes.index(equiv)
            indices = [co.restypes_atom_index_mapping[equiv][a] for a in names]
            coords[p, r, indices] = xyz
    cycles = np.array(
        [
            (p, a, c)
            for p, edges in enumerate(case.extra_connections())
            for a, b, c, d in edges
            if b == "up" and d == "down"
        ],
        np.int32,
    ).reshape(-1, 3)
    from tmol_core.chemical import GEOMETRY_NAMES, _site_connections

    metal_sites, coordination = [], []
    for p, row in enumerate(rows):
        for r, (name, _, _) in enumerate(row):
            residue = refined_from_raw(raw[name])
            if residue.metal_sites:
                metal_sites.append(
                    (p, r, GEOMETRY_NAMES.index(residue.metal_sites[0].geometry))
                )
        for metal, port, donor, donor_port in case.extra_connections()[p]:
            if port in ("up", "down"):
                continue
            metal_type = refined_from_raw(raw[row[metal][0]])
            donor_type = refined_from_raw(raw[row[donor][0]])
            if port not in _site_connections(metal_type):
                continue
            atom = donor_type.connections[
                donor_type.connection_to_cidx[donor_port]
            ].atom
            coordination.append(
                (
                    p,
                    metal,
                    _site_connections(metal_type).index(port),
                    donor,
                    co.restypes_atom_index_mapping[donor_type.io_equiv_class][atom],
                )
            )
    metal_sites = np.array(metal_sites, np.int32).reshape(-1, 3)
    coordination = np.array(coordination, np.int32).reshape(-1, 5)
    if shape[1] > 1:
        # Internal padding must not change polymer adjacency or declared cycles.
        coords = np.insert(coords, 1, np.nan, axis=1)
        types = np.insert(types, 1, -1, axis=1)
        chains = np.insert(chains, 1, -1, axis=1)
        gaps = np.insert(gaps, 1, False, axis=1)
        cycles[:, 1:] += cycles[:, 1:] >= 1
        metal_sites[:, 1] += metal_sites[:, 1] >= 1
        coordination[:, [1, 3]] += coordination[:, [1, 3]] >= 1
    return co, dict(
        coords=coords,
        res_types=types,
        chain_id=chains,
        res_gap=gaps,
        cyclic_bonds=cycles,
        metal_sites=metal_sites,
        metal_coordination=coordination,
        find_additional_cyclic_closures=False,
    )


def physical_edges(b):
    pose = b.pose
    types = b.numpy(pose.block_type_ind)
    connections = b.numpy(pose.inter_residue_connections)
    result = set()
    for p, r, port in zip(*np.nonzero(connections[..., 0] >= 0)):
        other, other_port = connections[p, r, port]
        if r < other:
            first = pose.packed_block_types.active_block_types[types[p, r]]
            second = pose.packed_block_types.active_block_types[types[p, other]]
            result.add(
                (
                    p,
                    r,
                    first.connections[port].atom,
                    other,
                    second.connections[other_port].atom,
                )
            )
    return result


@pytest.mark.parametrize("case", tuple(CASES))
def test_canonical_input_retains_molecular_state(backend, case):
    terms = ("gen_torsions", "cart_lengths", "fa_elec")
    if case.startswith("metal_"):
        terms += ("metal_coordination",)
    b = backend.molecule(CASES[case], terms=terms)
    expected_coords = b.coords().copy()
    expected_edges = physical_edges(b)
    expected_names = [
        [b.pose.block_type(p, r).name for r in range(len(row))]
        for p, row in enumerate(CASES[case].sequences)
    ]
    expected_channels, expected_gradient = b.channel_values_and_gradients(
        expected_coords
    )
    co, cf = canonical_input(CASES[case], b.parameter_database)
    rebuild_canonical(b, co, cf, unpatched_metals=case.startswith("metal_"))
    np.testing.assert_array_equal(b.coords(), expected_coords)
    assert physical_edges(b) == expected_edges
    assert [
        [b.pose.block_type(p, r).name for r in range(len(row))]
        for p, row in enumerate(CASES[case].sequences)
    ] == expected_names
    channels, gradient = b.channel_values_and_gradients(b.coords())
    if b.requested_device == "cpu":
        np.testing.assert_array_equal(channels, expected_channels)
        np.testing.assert_array_equal(gradient, expected_gradient)
    else:
        # Native CUDA electrostatics uses atomic reductions: unchanged repeated
        # evaluations also differ. Retain the shared molecular scoring bound;
        # coordinates, atom/type identities and topology above stay exact.
        np.testing.assert_allclose(channels, expected_channels, rtol=3e-5, atol=3e-5)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=3e-5, atol=3e-5)
    assert np.isfinite(gradient).all() and np.linalg.norm(gradient) > 0
    assert np.any(channels != 0)
    if case in ("ligand", "connected_noncanonical", "mixed_batch"):
        assert np.max(np.abs(channels[b.channel_names().index("gen_torsions")])) > 1e-3


def rebuild_canonical(b, co, cf, *, unpatched_metals=False):
    if b.framework == "jax":
        from tmol_jax.io import pose_stack_from_canonical_form
        from tmol_jax.pose.packed_block_types import PackedBlockTypes

        chemistry = b.parameter_database.chemical
        if unpatched_metals:
            from tmol_jax.database import ParameterDatabase

            chemistry = ParameterDatabase.get_default().chemical
            assert not any("metal_OD1" in r.name for r in chemistry.residues)
        with b.module.default_device(b.device):
            b.pose = pose_stack_from_canonical_form(
                dict(cf, canonical_ordering=co),
                PackedBlockTypes.default(chemistry),
                compact_padding=True,
            )
        b.pose = b.pose.to(b.device)
    else:
        from tmol.io._pose_stack_construction import _pose_stack_from_canonical_form
        from tmol.io._canonical_ordering import CanonicalOrdering

        co = CanonicalOrdering.from_chemdb(b.parameter_database.chemical)
        pbt = b.pose.packed_block_types
        if unpatched_metals:
            from tmol import ParameterDatabase, PackedBlockTypes
            from tmol.chemical._restypes import ResidueTypeSet

            chemistry = ParameterDatabase.get_default().chemical
            restypes = ResidueTypeSet.from_database(chemistry)
            pbt = PackedBlockTypes.from_restype_list(
                chemistry, restypes, restypes.residue_types, b.device
            )
        b.pose = _pose_stack_from_canonical_form(
            co,
            pbt,
            b.module.as_tensor(cf["chain_id"], device=b.device),
            b.module.as_tensor(cf["res_types"], device=b.device),
            b.module.as_tensor(cf["coords"], device=b.device),
            None,
            None,
            None,
            res_not_connected=b.module.as_tensor(cf["res_gap"], device=b.device),
            cyclic_bonds=b.module.as_tensor(
                cf["cyclic_bonds"].astype(np.int64), device=b.device
            ),
            find_additional_cyclic_closures=False,
            ligand_hydrogens="input",
            protonation_variants=(
                None
                if cf.get("protonation_variants") is None
                else b.module.as_tensor(
                    cf["protonation_variants"].astype(np.int64), device=b.device
                )
            ),
            metal_sites=b.module.as_tensor(
                cf["metal_sites"].astype(np.int64), device=b.device
            ),
            metal_coordination=b.module.as_tensor(
                cf["metal_coordination"].astype(np.int64), device=b.device
            ),
        )
        b._render_torch()
