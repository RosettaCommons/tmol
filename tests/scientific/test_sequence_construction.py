"""Shared sequence grammar yields real extended molecules and parameter context."""

from importlib import import_module

import numpy as np
import pytest

from .molecules import CASES, BACKBONE_CASES


def names(pose, backend):
    return [
        [pose.packed_block_types.active_block_types[int(k)].name for k in row if k >= 0]
        for row in backend.numpy(pose.block_type_ind)
    ]


@pytest.mark.parametrize(
    "sequence,expected",
    [
        (["AKG", "A"], [["ALA:nterm", "LYS", "GLY:cterm"], ["ALA:cterm:nterm"]]),
        (
            "AX[DALA]G:ac",
            [["ALA:nterm", "DALA", "GLY:cterm", "DA:na5prime", "DC:na3prime"]],
        ),
        ("a[RA]c[RC]g[RG]", [["RA:na5prime", "RC", "RG:na3prime"]]),
        ("X[HIS_D]GX[HYP]", [["HIS_D:nterm", "GLY", "HYP"]]),
        ("AX[HYP]V:X[LG1]", [["ALA:nterm", "HYP", "VAL:cterm", "LG1"]]),
    ],
    ids=["protein-ragged", "d-peptide-dna", "rna", "custom-variant", "custom-ligand"],
)
def test_sequence_identity_context_torsions_and_full_scoring(
    backend, sequence, expected
):
    b = backend
    api = import_module("tmol_jax" if b.framework == "jax" else "tmol")
    query = import_module(api.__name__ + ".pose")
    db = None
    if sequence == "X[HIS_D]GX[HYP]":
        db = CASES["noncanonical"].database(api.ParameterDatabase.get_default())
    if sequence == "AX[HYP]V:X[LG1]":
        db = BACKBONE_CASES["mixed"].database(api.ParameterDatabase.get_default())
    pose, context = api.extended_pose_stack_from_sequences(
        sequence, device=b.device, param_db=db, return_context=True
    )
    assert names(pose, b) == expected
    assert pose.coords.device == b.device
    assert pose.chain_id.device == b.device
    np.testing.assert_array_equal(
        b.numpy(pose.chain_id) < 0, b.numpy(pose.block_type_ind) < 0
    )
    assert np.isfinite(b.numpy(pose.coords)).all()
    if db is not None:
        assert context.parameter_database is db
    copied = api.extended_pose_stack_from_sequences(
        sequence, device=b.device, context=context
    )
    assert copied.packed_block_types.chem_db is context.parameter_database.chemical
    assert names(copied, b) == expected
    np.testing.assert_array_equal(b.numpy(copied.coords), b.numpy(pose.coords))
    # Independent known extended peptide/DNA/RNA targets, rather than reading
    # the implementation's target table back into the assertion.
    for p, row in enumerate(expected):
        for block, rt in enumerate(row):
            measured = query.get_named_torsions(pose, p, block)
            targets = (
                {"beta": 136.0}
                if rt.split(":")[0] in ("DA", "DC")
                else (
                    {"beta": 178.0}
                    if rt.split(":")[0] in ("RA", "RC", "RG")
                    else {"psi": 135.0}
                )
            )
            for name, target in targets.items():
                if name in measured and np.isfinite(measured[name]):
                    assert (measured[name] - target + 180) % 360 - 180 == pytest.approx(
                        0, abs=1e-2
                    )
    b.pose = pose
    b.parameter_database = context.parameter_database
    if b.framework == "jax":
        b.score_function = api.beta2016_score_function(context.parameter_database)
    else:
        b.score_function = api.beta2016_score_function(
            b.device, param_db=context.parameter_database
        )
        b._render_torch()
    values, gradients = b.channel_values_and_gradients(b.coords())
    assert values.shape[0] == 25
    assert np.isfinite(values).all() and np.isfinite(gradients).all()
    assert np.linalg.norm(gradients) > 0
    with pytest.raises(ValueError, match="either context"):
        api.extended_pose_stack_from_sequences(
            sequence,
            device=b.device,
            context=context,
            param_db=context.parameter_database,
        )


def test_sequence_smiles_context_reuse_and_connection_labels(backend):
    b = backend
    api = import_module("tmol_jax" if b.framework == "jax" else "tmol")
    pose, context = api.extended_pose_stack_from_sequences(
        "X(CCO):AG", device=b.device, return_context=True
    )
    assert names(pose, b)[0] == [context.ligand_names["CCO"], "ALA:nterm", "GLY:cterm"]
    assert context.parameter_database is not api.ParameterDatabase.get_default()
    assert b.numpy(api.get_ligand_mask(pose)).tolist() == [[True, False, False]]
    repeated = api.extended_pose_stack_from_sequences(
        "X(CCO):AG", device=b.device, context=context
    )
    np.testing.assert_array_equal(b.numpy(pose.coords), b.numpy(repeated.coords))
    assert np.isfinite(b.numpy(pose.coords)).all()
    sulfur = api.extended_pose_stack_from_sequences(
        "X[CYD--dslf-pair]AGX[CYD--dslf-pair]", device=b.device, termini=False
    )
    c0 = sulfur.block_type(0, 0).connection_to_cidx["dslf"]
    c3 = sulfur.block_type(0, 3).connection_to_cidx["dslf"]
    conn = b.numpy(sulfur.inter_residue_connections)
    np.testing.assert_array_equal(conn[0, 0, c0], [3, c3])
    np.testing.assert_array_equal(conn[0, 3, c3], [0, c0])
    with pytest.raises(ValueError, match="unknown"):
        api.extended_pose_stack_from_sequences("X[NOT_A_RESIDUE]", device=b.device)
    with pytest.raises(ValueError, match="Failed to find connection"):
        api.extended_pose_stack_from_sequences(
            "X[CYD--missing-pair]AX[CYD--dslf-pair]", device=b.device, termini=False
        )
