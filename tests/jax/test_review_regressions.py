"""Cache ownership, short topologies, metadata and interpolation contracts."""

from pathlib import Path

import attrs
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tmol_jax import beta2016_score_function, pose_stack_from_pdb
from tmol_jax.io import (
    canonical_form_from_pdb,
    default_packed_block_types,
    pose_stack_from_canonical_form,
)
from tmol_jax.io.write_pose_stack_pdb import atom_records_from_pose_stack
from tmol_jax.numeric.bspline import bspline_coefficients, bspline_interpolate
from tmol_jax.optimization.line_search import _cubic_interp, strong_wolfe_line_search
from tmol_jax.score import ScoreFunction, ScoreType
from tmol_jax.score.cartbonded.cartbonded_energy_term import CartBondedEnergyTerm

PDB = Path(__file__).parents[2] / "tmol/tests/data/pdb/1ubq.pdb"


def peptide(n):
    return (
        "\n".join(
            line
            for line in PDB.read_text().splitlines()
            if line.startswith("ATOM") and 1 <= int(line[22:26]) <= n
        )
        + "\nEND\n"
    )


@pytest.mark.parametrize("n", [1, 2])
def test_short_palette_scores_and_differentiates(n):
    pose = pose_stack_from_pdb(peptide(n))
    slots = pose.packed_block_types.max_n_conn
    assert pose.inter_residue_connections.shape[2] == slots
    assert pose.inter_block_bondsep.shape[-2:] == (slots, slots)
    pbt = pose.packed_block_types
    for atoms, count in zip(np.asarray(pbt.conn_atoms), np.asarray(pbt.n_conn)):
        assert np.all(atoms[count:] == -1)
    energy, gradient = beta2016_score_function().jitted_value_and_grad(pose)(
        pose.coords
    )
    assert np.isfinite(energy).all() and np.isfinite(gradient).all()


def test_elec_database_switch_preserves_rendered_scorers():
    pose = pose_stack_from_pdb(peptide(3))
    db = beta2016_score_function().param_db
    zero_elec = attrs.evolve(
        db.scoring.elec,
        atom_charge_parameters=tuple(
            attrs.evolve(p, charge=0.0) for p in db.scoring.elec.atom_charge_parameters
        ),
    )
    zero_db = attrs.evolve(db, scoring=attrs.evolve(db.scoring, elec=zero_elec))

    def render(database):
        sf = ScoreFunction(database)
        sf.set_weight(ScoreType.fa_elec, 1.0)
        return jax.jit(sf.render_whole_pose_scoring_module(pose))

    standard = render(db)
    charges = pose.packed_block_types.elec_partial_charge
    expected_charges = np.asarray(charges).copy()
    expected = np.asarray(standard(pose.coords))
    assert np.any(expected != 0)
    zero = render(zero_db)
    np.testing.assert_array_equal(pose.packed_block_types.elec_partial_charge, 0.0)
    np.testing.assert_array_equal(charges, expected_charges)
    np.testing.assert_array_equal(zero(pose.coords), 0.0)
    # Native float32 tile sums use unordered atomic adds (three ULPs observed
    # over 600 repeated evaluations). Check cache data exactly above; allow
    # only reduction roundoff in scores, well below the zero-charge change.
    np.testing.assert_allclose(standard(pose.coords), expected, rtol=1e-6, atol=0)
    np.testing.assert_allclose(render(db)(pose.coords), expected, rtol=1e-6, atol=0)
    np.testing.assert_array_equal(
        pose.packed_block_types.elec_partial_charge, expected_charges
    )
    np.testing.assert_array_equal(zero(pose.coords), 0.0)


def test_cart_scores_follow_molecular_identity_across_owner_lifetimes():
    from tmol_jax import ResidueBlock, pose_stack_from_residue_blocks

    db = beta2016_score_function().param_db
    full = default_packed_block_types()
    term = CartBondedEnergyTerm(db)
    scores = []
    for name in ("ILE", "LEU", "ILE", "LEU"):
        bt = next(rt for rt in full.active_block_types if rt.name == name)
        x = bt.ideal_coords[bt.at_to_icoor_ind].astype(np.float32)
        x += np.random.default_rng(814).normal(0, 0.05, x.shape)
        pose = pose_stack_from_residue_blocks(
            [[ResidueBlock(name, tuple(a.name for a in bt.atoms), x)]],
            parameter_database=db,
        )
        state = term.setup_for_pose_stack(pose)
        scores.append(jnp.stack(list(term(pose, state).values())))
    np.testing.assert_array_equal(scores[0], scores[2])
    np.testing.assert_array_equal(scores[1], scores[3])
    assert not np.allclose(scores[0], scores[1])


@pytest.fixture
def heterogeneous_pose():
    canonical = canonical_form_from_pdb(PDB)
    canonical = {
        key: np.concatenate([value[:, :1], value[:, 1:2]], axis=0)
        for key, value in canonical.items()
    }
    canonical["chain_labels"][:] = [["A"], ["B"]]
    canonical["res_num"][:] = [[11], [22]]
    return pose_stack_from_canonical_form(canonical, default_packed_block_types())


@pytest.mark.parametrize(
    "index, expected",
    [
        (1, [1]),
        (-1, [1]),
        (np.int64(-2), [0]),
        (slice(None, None, -1), [1, 0]),
        ([1, 0, 1], [1, 0, 1]),
        (np.array([False, True]), [1]),
        (slice(0, 0), []),
    ],
)
def test_pose_slice_preserves_pdb_identity(heterogeneous_pose, index, expected):
    pose = heterogeneous_pose
    sliced = pose[index]
    assert sliced.n_poses == len(expected)
    for i, source in enumerate(expected):

        def identity_records(stack, pose_index):
            return [
                {
                    key: value
                    for key, value in record.items()
                    if key not in ("model", "modeli")
                }
                for record in atom_records_from_pose_stack(stack, pose_index=pose_index)
            ]

        assert identity_records(sliced, i) == identity_records(pose, source)
        np.testing.assert_array_equal(sliced.coords[i], pose.coords[source])
    assert pose.replace(pdb_info=None)[index].pdb_info is None


@pytest.mark.parametrize("index", [-3, 2])
def test_pose_integer_bounds(heterogeneous_pose, index):
    with pytest.raises(IndexError):
        heterogeneous_pose[index]


@pytest.mark.parametrize("shape", [(2, 2), (4, 4), (8, 9), (36, 36)])
def test_periodic_spline_reconstructs_grid(shape):
    with jax.enable_x64(True):
        values = jnp.asarray(np.random.default_rng(712).normal(size=shape))
        points = jnp.asarray(
            np.stack(
                np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1
            )
        )
        restored = bspline_interpolate(bspline_coefficients(values), points)
        np.testing.assert_allclose(restored, values, rtol=0, atol=3e-14)


def test_cubic_line_search_quadratic():
    assert _cubic_interp(0.0, 4.0, -4.0, 3.0, 1.0, 2.0) == pytest.approx(2.0)
    assert _cubic_interp(3.0, 1.0, 2.0, 0.0, 4.0, -4.0) == pytest.approx(2.0)
    calls = []

    def value_and_grad(x):
        calls.append(float(x[0]))
        return ((x - 2) ** 2).sum(), 2 * (x - 2)

    x = jnp.zeros(1)
    alpha, value, gradient = strong_wolfe_line_search(
        value_and_grad,
        x,
        jnp.ones(1),
        jnp.asarray(4.0),
        jnp.asarray([-4.0]),
        alpha0=5.0,
        c2=0.1,
    )
    assert alpha == pytest.approx(2.0, abs=1e-6)
    assert len(calls) == 2
    assert float(value) < 1e-12
    np.testing.assert_allclose(gradient, 0, atol=1e-6)


@pytest.mark.parametrize(
    "field", ["chain_id", "res_num", "res_name", "atom_record_type"]
)
def test_pose_slice_preserves_missing_metadata_fields(heterogeneous_pose, field):
    pose = heterogeneous_pose.replace(
        pdb_info=attrs.evolve(heterogeneous_pose.pdb_info, **{field: None})
    )
    assert getattr(pose[-1].pdb_info, field) is None
    blank = attrs.evolve(
        pose.pdb_info, chain_id=None, res_num=None, res_name=None, atom_record_type=None
    )
    assert pose.replace(pdb_info=blank)[-1].n_poses == 1


def test_elec_adapter_preserves_restype_overrides_and_compound_patches():
    from types import SimpleNamespace

    from tmol_core.scoring.elec import partial_charges_for_block

    block = SimpleNamespace(name="LIG:nterm:conj_x:other", atoms=("N", "H", "V"))
    table = {
        "LIG": {
            "N": {"": 1.0, "nterm:conj_x": 2.0, "nterm:conj_x:other": 3.0},
            "H": {"": 4.0},
        }
    }
    with pytest.raises(KeyError, match="V not found"):
        partial_charges_for_block(
            block, table, overrides=getattr(block, "elec_partial_charges", None)
        )
    table["LIG"]["V"] = {"": 0.0}
    np.testing.assert_array_equal(
        partial_charges_for_block(
            block, table, overrides=getattr(block, "elec_partial_charges", None)
        ),
        [3, 4, 0],
    )
    block.elec_partial_charges = {"N": -0.5, "H": 0.5}
    with pytest.raises(KeyError, match="V not found"):
        partial_charges_for_block(
            block, table, overrides=getattr(block, "elec_partial_charges", None)
        )
    block.elec_partial_charges["V"] = 0.0
    np.testing.assert_array_equal(
        partial_charges_for_block(
            block, table, overrides=getattr(block, "elec_partial_charges", None)
        ),
        [-0.5, 0.5, 0],
    )
    block.elec_partial_charges["V"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        partial_charges_for_block(
            block, table, overrides=getattr(block, "elec_partial_charges", None)
        )
