"""Molecular input preserves chemistry, atom identity, graph edges and precision."""

import dataclasses

import jax
import numpy as np
import pytest

from tmol_jax.database import ParameterDatabase
from tmol_jax.io import ResidueBlock, pose_stack_from_residue_blocks


@pytest.fixture(scope="module")
def database():
    return ParameterDatabase.get_default()


def block(database, name, dtype=np.float32):
    residue = next(r for r in database.chemical.residues if r.name == name)
    # Labeled, distinct coordinates make atom-order loss observable independently
    # of the ideal-coordinate implementation. These are not scoring fixtures.
    xyz = np.arange(3 * len(residue.atoms), dtype=dtype).reshape(-1, 3) / 7
    return ResidueBlock(name, [a.name for a in residue.atoms][::-1], xyz[::-1])


def test_mixed_batch_preserves_exact_types_atom_order_and_context(database):
    inputs = [
        [block(database, name) for name in ("DALA", "HOH")],
        [block(database, "GLY")],
    ]
    pose, context = pose_stack_from_residue_blocks(
        inputs, parameter_database=database, return_context=True
    )
    assert context.parameter_database is database
    assert pose.packed_block_types.chem_db is database.chemical
    assert [
        [pose.block_type(p, b).name for b in range(len(row))]
        for p, row in enumerate(inputs)
    ] == [["DALA", "HOH"], ["GLY"]]
    for p, row in enumerate(inputs):
        expected = np.concatenate([r.coords[::-1] for r in row])
        np.testing.assert_array_equal(pose.coords[p, : len(expected)], expected)
        assert np.asarray(pose.atom_is_real[p]).sum() == len(expected)
    assert np.all(np.asarray(pose.inter_residue_connections) == -1)
    reordered = pose_stack_from_residue_blocks(inputs[::-1], context=context)
    assert reordered.packed_block_types is context.packed_block_types
    assert reordered.block_type(0, 0).name == "GLY"


def test_explicit_cycle_has_symmetric_connections_and_graph_distance(database):
    inputs = [[block(database, "GLY") for _ in range(3)]]
    edges = [[(0, "up", 1, "down"), (1, "up", 2, "down"), (2, "up", 0, "down")]]
    pose = pose_stack_from_residue_blocks(
        inputs, parameter_database=database, connections=edges
    )
    rt = pose.block_type(0, 0)
    up, down = rt.connection_to_cidx["up"], rt.connection_to_cidx["down"]
    conn, sep = map(
        np.asarray, (pose.inter_residue_connections, pose.inter_block_bondsep)
    )
    for left, _, right, _ in edges[0]:
        np.testing.assert_array_equal(conn[0, left, up], [right, down])
        np.testing.assert_array_equal(conn[0, right, down], [left, up])
        assert sep[0, left, right, up, down] == 1
    # GLY down/up ports lie on N and C: N--CA--C is exactly two bonds.
    assert sep[0, 0, 0, down, up] == 2
    # N0--C2--CA2--N2 takes three bonds across the closing edge.
    assert sep[0, 0, 2, down, down] == 3


@pytest.mark.parametrize(
    "edges",
    [
        [(0, "missing", 1, "down")],
        [(0, "up", 3, "down")],
        [(0, "up", 0, "down")],
        [(0, "up", 1, "down"), (0, "up", 2, "down")],
    ],
)
def test_invalid_connections_raise(database, edges):
    with pytest.raises(ValueError):
        pose_stack_from_residue_blocks(
            [[block(database, "GLY")] * 3],
            parameter_database=database,
            connections=[edges],
        )


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "unknown"])
def test_atom_and_residue_identity_cannot_be_silently_coerced(database, change):
    b = block(database, "DALA")
    if change == "missing":
        b = dataclasses.replace(b, atom_names=b.atom_names[:-1], coords=b.coords[:-1])
    elif change == "extra":
        b = dataclasses.replace(
            b,
            atom_names=[*b.atom_names, "UNKNOWN"],
            coords=np.concatenate([b.coords, b.coords[:1]]),
        )
    elif change == "duplicate":
        b = dataclasses.replace(b, atom_names=[b.atom_names[1], *b.atom_names[1:]])
    else:
        b = dataclasses.replace(b, residue_type="UNKNOWN")
    with pytest.raises(ValueError):
        pose_stack_from_residue_blocks([[b]], parameter_database=database)


def test_float64_coordinates_are_preserved_or_rejected_explicitly(database):
    inputs = [[block(database, "DALA", np.float64)]]
    with jax.enable_x64(True):
        pose = pose_stack_from_residue_blocks(inputs, parameter_database=database)
        assert pose.coords.dtype == np.float64
        np.testing.assert_array_equal(pose.coords[0], inputs[0][0].coords[::-1])
    with jax.enable_x64(False), pytest.raises(ValueError, match="float64"):
        pose_stack_from_residue_blocks(inputs, parameter_database=database)


@pytest.mark.parametrize(
    "case_name",
    ("ligand", "connected_noncanonical", "d_peptide", "modified_dna", "mixed_batch"),
)
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_explicit_missing_leaf_input_preserves_chemistry_and_observations(
    case_name, dtype
):
    from tests.scientific.execution import Execution
    from tests.scientific.molecules import DOMAIN_CASES
    from tmol_core.io.missing_atoms import prepare_leaf_atom_metadata

    case = DOMAIN_CASES[case_name]
    terms = {
        "ligand": "gen_torsions",
        "connected_noncanonical": "cart_lengths",
        "d_peptide": "fa_elec",
        "modified_dna": "na_torsion",
        "mixed_batch": "gen_torsions",
    }
    lane = "jax-cuda" if jax.devices()[0].platform == "gpu" else "jax-cpu"
    b = Execution(lane).molecule(case, terms=(terms[case_name],))
    with jax.enable_x64(dtype == np.float64), jax.default_device(b.device):
        complete = [
            [ResidueBlock(*record) for record in row]
            for row in case.blocks(b.parameter_database, dtype)
        ]
        expected, context = pose_stack_from_residue_blocks(
            complete,
            parameter_database=b.parameter_database,
            connections=case.connections(),
            return_context=True,
        )
        pbt = context.packed_block_types
        info = prepare_leaf_atom_metadata(
            pbt.active_block_types, np.asarray(pbt.atom_is_hydrogen), pbt.max_n_atoms
        )
        incomplete, observed, targets = [], np.asarray(expected.atom_is_real).copy(), []
        for p, row in enumerate(complete):
            prepared = []
            for r, block in enumerate(row):
                t = int(expected.block_type_ind[p, r])
                n = len(block.atom_names)
                eligible = np.flatnonzero(
                    info.is_leaf_atom[t, :n]
                    & np.asarray(pbt.atom_is_hydrogen[t, :n])
                    & np.all(info.icoors.anc_uaids[t, :n, :, 0] >= 0, axis=1)
                )
                keep = np.ones(n, bool)
                if eligible.size:
                    atom = int(eligible[0])
                    keep[atom] = False
                    offset = int(expected.block_coord_offset[p, r])
                    observed[p, offset + atom] = False
                    targets.append(
                        (
                            p,
                            offset + atom,
                            offset + info.icoors.anc_uaids[t, atom, 0, 0],
                            info.icoors.geom[t, atom, 2],
                        )
                    )
                prepared.append(
                    ResidueBlock(
                        block.residue_type,
                        np.asarray(block.atom_names)[keep][::-1],
                        block.coords[keep][::-1],
                    )
                )
            incomplete.append(prepared)
        assert targets, f"{case_name} must exercise missing molecular hydrogens"
        with pytest.raises(ValueError, match="atoms must match"):
            pose_stack_from_residue_blocks(
                incomplete, context=context, connections=case.connections()
            )
        actual, returned = pose_stack_from_residue_blocks(
            incomplete,
            context=context,
            connections=case.connections(),
            return_context=True,
            build_missing_leaf_atoms=True,
        )
        assert returned is context and actual.packed_block_types is pbt
        assert actual.coords.dtype == dtype
        np.testing.assert_array_equal(
            actual.coords[observed], expected.coords[observed]
        )
        np.testing.assert_array_equal(actual.block_type_ind, expected.block_type_ind)
        np.testing.assert_array_equal(
            actual.inter_residue_connections, expected.inter_residue_connections
        )
        for p, atom, parent, distance in targets:
            np.testing.assert_allclose(
                np.linalg.norm(
                    np.asarray(actual.coords[p, atom] - actual.coords[p, parent])
                ),
                distance,
                rtol=2e-6 if dtype == np.float32 else 2e-12,
                atol=2e-6 if dtype == np.float32 else 2e-12,
            )
        b.pose = actual.to(b.device)
        energy, gradient = b.value_and_grad(b.coords())
        assert np.isfinite(energy).all() and np.isfinite(gradient).all()
        assert np.any(energy != 0) and np.linalg.norm(gradient) > 0
        # Opting into leaf reconstruction must not hide a missing core atom.
        first = incomplete[0][0]
        nonleaf = np.flatnonzero(
            ~info.is_leaf_atom[0, : len(complete[0][0].atom_names)]
        )[0]
        atom = complete[0][0].atom_names[nonleaf]
        keep = np.asarray(first.atom_names) != atom
        invalid = [
            [
                dataclasses.replace(
                    first,
                    atom_names=np.asarray(first.atom_names)[keep],
                    coords=first.coords[keep],
                ),
                *incomplete[0][1:],
            ],
            *incomplete[1:],
        ]
        with pytest.raises(ValueError, match="missing non-leaf"):
            pose_stack_from_residue_blocks(
                invalid,
                context=context,
                connections=case.connections(),
                build_missing_leaf_atoms=True,
            )
