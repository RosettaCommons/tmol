"""Shared Cartesian-bonded physics on real molecular topologies."""

import numpy as np
import pytest

from .test_score_contract import check_coordinate_transforms

from .molecules import BACKBONE_CASES, DOMAIN_CASES, MolecularCase

CARTESIAN_CASES = {
    **DOMAIN_CASES,
    "hydroxyls": MolecularCase((), (("SER", "THR", "TYR"),), ((3,),)),
}

TERMS = (
    "cart_lengths",
    "cart_angles",
    "cart_torsions",
    "cart_impropers",
    "cart_hxltorsions",
)


@pytest.mark.parametrize("name", CARTESIAN_CASES)
def test_cartesian_bonded_channels_and_coordinate_derivatives(backend, name, tmp_path):
    backend.molecule(CARTESIAN_CASES[name], terms=TERMS)
    x = backend.coords()
    values, gradients = backend.channel_values_and_gradients(x)
    assert np.isfinite(values).all() and np.isfinite(gradients).all()
    assert np.abs(values).sum() > 1e-3
    assert np.linalg.norm(gradients) > 1e-3
    # All five channels are checked, including genuinely absent interactions.
    direction = np.random.default_rng(195).normal(size=x.shape).astype(x.dtype)
    direction /= np.linalg.norm(direction)
    step = 5e-3
    numeric = (
        backend.scores(x + step * direction, sum_terms=False, apply_weights=False)
        - backend.scores(x - step * direction, sum_terms=False, apply_weights=False)
    ) / (2 * step)
    analytic = (gradients * direction).sum(axis=(2, 3))
    np.testing.assert_allclose(analytic, numeric, rtol=3e-3, atol=3e-3)
    np.testing.assert_allclose(gradients.sum(axis=2), 0, atol=1e-3)

    if name == "mixed_batch":
        check_coordinate_transforms(
            backend,
            CARTESIAN_CASES[name],
            TERMS,
            empty_term="cart_lengths",
            empty_case=BACKBONE_CASES["mixed"],
        )

        if backend.framework == "jax":
            _check_peptide_improper_geometry(tmp_path)


@pytest.mark.parametrize("name", ("ligand", "connected_noncanonical"))
def test_cartesian_spring_context_reuse_preserves_values_and_gradients(backend, name):
    import attrs

    backend.molecule(DOMAIN_CASES[name], terms=("cart_lengths",))
    pose = backend.pose
    palette = pose.packed_block_types
    coords = backend.coords()
    expected = backend.value_and_grad(coords)
    assert np.max(np.abs(expected[0])) > 1e-3
    assert np.linalg.norm(expected[1]) > 1e-3
    database = backend.parameter_database
    cart = database.scoring.cartbonded

    def springs(record):
        return attrs.evolve(
            record,
            length_parameters=tuple(
                attrs.evolve(group, K=3.25 * group.K)
                for group in record.length_parameters
            ),
        )

    changed = type(cart).from_cartres_dict(
        {name: springs(record) for name, record in cart.residue_params.items()},
        connection_params=tuple(map(springs, cart.connection_params)),
    )
    scaled = attrs.evolve(
        database, scoring=attrs.evolve(database.scoring, cartbonded=changed)
    )
    for context, factor in (
        (database, 1),
        (scaled, 3.25),
        (database, 1),
        (scaled, 3.25),
    ):
        backend.use_parameter_database(context)
        value, gradient = backend.value_and_grad(coords)
        for actual, reference in zip((value, gradient), expected):
            np.testing.assert_allclose(actual, factor * reference, rtol=3e-6, atol=3e-5)
        assert backend.pose is pose and backend.pose.packed_block_types is palette
        np.testing.assert_array_equal(backend.coords(), coords)

    if backend.framework == "jax" and name == "ligand":
        _check_cartesian_owner_identity()


def _check_peptide_improper_geometry(tmp_path):
    from tmol_jax import pose_stack_from_pdb, beta2016_score_function
    from .test_canonical_molecular_input import peptide_pdb

    """Planarity impropers across each peptide bond follow plane-normal geometry."""
    import jax.numpy as jnp
    from tmol_jax.score import ScoreType

    path = tmp_path / "tripeptide.pdb"
    path.write_text(peptide_pdb(3))
    pose = pose_stack_from_pdb(path)
    term = next(
        t
        for t in beta2016_score_function().terms
        if type(t).__name__ == "CartBondedEnergyTerm"
    )
    state = term.setup_for_pose_stack(pose)
    # Twist the peptide groups off planarity, where energies vanish anyway.
    coords = pose.coords + 0.17 * jnp.sin(jnp.arange(pose.coords.size)).reshape(
        pose.coords.shape
    ).astype(pose.coords.dtype)
    disconnected = pose.replace(
        inter_residue_connections=jnp.full_like(pose.inter_residue_connections, -1)
    )
    intra_only = term.setup_for_pose_stack(disconnected)
    inter = (
        term(pose, state, coords)[ScoreType.cart_impropers]
        - term(disconnected, intra_only, coords)[ScoreType.cart_impropers]
    )

    pbt = pose.packed_block_types
    x = np.asarray(coords, dtype=np.float64)[0]

    def xyz(block, name):
        block_type = pbt.active_block_types[int(pose.block_type_ind[0, block])]
        return x[int(pose.block_coord_offset[0, block]) + block_type.atom_to_idx[name]]

    def improper(k2, a, b, c, d):
        first, second = np.cross(b - a, c - b), np.cross(c - b, d - c)
        cosine = first @ second / np.linalg.norm(first) / np.linalg.norm(second)
        return 2 * k2 * (1 - cosine**2)

    expected = 0.0
    for left in (0, 1):
        c, n = xyz(left, "C"), xyz(left + 1, "N")
        expected += improper(9.667, xyz(left, "CA"), n, c, xyz(left, "O"))
        expected += improper(10.458, xyz(left + 1, "CA"), c, n, xyz(left + 1, "H"))
    assert expected > 0.01
    np.testing.assert_allclose(inter[0], expected, rtol=2e-4)


def _check_cartesian_owner_identity():
    import jax.numpy as jnp
    from tmol_jax import beta2016_score_function
    from tmol_jax.io import default_packed_block_types
    from tmol_jax.score.cartbonded.cartbonded_energy_term import CartBondedEnergyTerm

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
