"""Retained score channels and rendered snapshots obey the mature public API."""

from importlib import import_module

import numpy as np
import pytest

from .molecules import BACKBONE_CASES, CASES


def evaluate(b, scorer, *, summed=False):
    if b.framework == "jax":
        import jax

        if summed:
            value, grad = jax.jit(jax.value_and_grad(lambda x: scorer(x).sum()))(
                b.pose.coords
            )
        else:
            value = scorer(b.pose.coords)
            grad = jax.grad(lambda x: scorer(x).sum())(b.pose.coords)
    else:
        x = b.pose.coords.detach().clone().requires_grad_(True)
        value = scorer(x)
        if summed:
            value = value.sum()
        grad = b.module.autograd.grad(value.sum(), x)[0]
    return b.numpy(value), b.numpy(grad)


@pytest.mark.parametrize(
    "weighted,channels",
    (
        ("fa_ljatr", ("fa_ljatr", "fa_ljrep", "fa_lk")),
        ("omega", ("rama", "omega")),
        ("gen_torsions", ("gen_torsions",)),
        ("ligand_charges", ()),
        ("mixed_charges", ()),
        ("mixed_fused_charges", ()),
        ("quadratic", ()),
        ("constraint_callbacks", ()),
    ),
)
def test_retained_zero_channels_removal_and_old_snapshot_are_consistent(
    backend, weighted, channels, monkeypatch
):
    if weighted.endswith("_charges"):
        _check_charge_snapshots(backend, weighted)
        return
    if weighted in ("quadratic", "constraint_callbacks"):
        if backend.framework != "jax":
            pytest.skip("JAX rendered-program capability")
        if weighted == "quadratic":
            _check_quadratic_snapshot(monkeypatch)
        else:
            _check_constraint_snapshots(backend)
        return
    if weighted == "gen_torsions":
        _check_generic_database_snapshot(backend)
        return
    b = backend.molecule(BACKBONE_CASES["mixed"], terms=(weighted,))
    sf = b.score_function
    st = import_module("tmol_jax" if b.framework == "jax" else "tmol").ScoreType
    channel = st[weighted]
    assert tuple(x.name for x in sf.all_score_types()) == channels
    assert len(sf.two_body_terms()) == 1
    assert not sf.one_body_terms() and not sf.multi_body_terms()
    assert sf.pre_work_initialization(b.pose) is None
    raw, gradients = b.channel_values_and_gradients(b.coords())
    assert raw.shape == (len(channels), b.pose.n_poses)
    assert np.linalg.norm(raw[0]) > 1e-4
    weights = np.array([name == weighted for name in channels], np.float32)
    np.testing.assert_array_equal(b.numpy(sf.weights_tensor()), weights)
    old = sf.render_whole_pose_scoring_module(b.pose)
    values, gradient = evaluate(b, old)
    index = channels.index(weighted)
    np.testing.assert_allclose(values, raw[index], rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(gradient, gradients[index], rtol=3e-5, atol=3e-5)
    sf.remove_term_for_score_type(channel)
    assert not sf.all_terms() and not sf.all_score_types()
    assert b.numpy(sf.weights_tensor()).size == 0
    assert not sf.score_type_covered_by_contained_term(channel)
    sf.set_weight(channel, -2)
    assert sf.pre_work_initialization(b.pose) is None
    fresh_values, fresh_gradient = evaluate(
        b, sf.render_whole_pose_scoring_module(b.pose)
    )
    np.testing.assert_allclose(fresh_values, -2 * values, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(fresh_gradient, -2 * gradient, rtol=3e-5, atol=3e-5)
    old_values, old_gradient = evaluate(b, old)
    for repeated, reference in ((old_values, values), (old_gradient, gradient)):
        # Independent unchanged-scorer CUDA controls show float32 atomic-sum
        # variation before mutation, with unchanged float64-oracle error after
        # mutation. Bound only this fixed-fixture repeat check at two epsilons
        # of its value/force scale; keep CPU snapshots bit-exact.
        if b.requested_device == "cuda":
            atol = 2 * np.finfo(reference.dtype).eps * np.max(np.abs(reference))
            np.testing.assert_allclose(repeated, reference, rtol=0, atol=atol)
        else:
            np.testing.assert_array_equal(repeated, reference)
    sf.set_weight(channel, 0)
    assert not sf.all_terms() and not sf.all_score_types()


def test_public_term_order_retains_native_channel_groups(backend):
    b = backend.molecule(
        BACKBONE_CASES["mixed"], terms=("gen_torsions", "omega", "fa_ljatr")
    )
    sf = b.score_function
    terms, channels = sf.get_sorted_terms(reversed(sf.all_terms()))
    assert terms == sf.all_terms()
    assert tuple(st.name for st in channels) == (
        "fa_ljatr",
        "fa_ljrep",
        "fa_lk",
        "rama",
        "omega",
        "gen_torsions",
    )


def _check_generic_database_snapshot(b):
    import attrs

    b.molecule(CASES["ligand"])
    old = b.score_function.render_whole_pose_scoring_module(b.pose)
    db = b.parameter_database
    gen = db.scoring.genbonded
    changed = attrs.evolve(
        gen,
        torsions=tuple(
            attrs.evolve(
                p,
                **{
                    name: 2 * getattr(p, name)
                    for name in ("k1", "k2", "k3", "k4", "offset")
                },
            )
            for p in gen.torsions
        ),
        impropers=tuple(attrs.evolve(p, k=2 * p.k) for p in gen.impropers),
    )
    value, gradient = evaluate(b, old, summed=True)
    value = value.sum()
    assert value > 0 and np.linalg.norm(gradient) > 0
    b.use_parameter_database(
        attrs.evolve(db, scoring=attrs.evolve(db.scoring, genbonded=changed))
    )
    new = b.score_function.render_whole_pose_scoring_module(b.pose)
    new_value, new_gradient = evaluate(b, new, summed=True)
    np.testing.assert_allclose(new_value.sum(), 2 * value, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(new_gradient, 2 * gradient, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(b.numpy(old(b.pose.coords)).sum(), value)


def _quadratic_scorer(monkeypatch):
    from types import SimpleNamespace
    from tmol_jax.score import score_function as scoring
    from tmol_jax.score import ScoreFunction, ScoreType
    from tmol_jax.score.energy_term import EnergyTerm

    class QuadraticTerm(EnergyTerm):
        def __init__(self, database):
            self.scale = 1.0

        def score_types_provided(self):
            return (ScoreType.fa_ljatr,)

        def setup_for_pose_stack(self, pose, block_types):
            return 1.0

        def __call__(self, pose, state, coords_override=None):
            return {
                ScoreType.fa_ljatr: self.scale
                * state
                * (coords_override**2).sum(axis=(1, 2))
            }

    original = scoring.importlib.import_module
    monkeypatch.setattr(
        scoring.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(QuadraticTerm=QuadraticTerm)
            if name == "tmol_jax.score.fake.fake_energy_term"
            else original(name)
        ),
    )
    monkeypatch.setitem(
        scoring._TERM_FOR_TYPE, ScoreType.fa_ljatr, ("fake", "QuadraticTerm")
    )
    score = ScoreFunction(None)
    score.set_weight(ScoreType.fa_ljatr, 2.0)
    return score


def _check_quadratic_snapshot(monkeypatch):
    import jax
    import jax.numpy as jnp
    from types import SimpleNamespace
    from tmol_jax import ScoreType

    scorer = _quadratic_scorer(monkeypatch)
    pose = SimpleNamespace(coords=jnp.ones((2, 3, 3)), packed_block_types=None)
    old = jax.jit(scorer.render_whole_pose_scoring_module(pose))
    scorer.set_weight(ScoreType.fa_ljatr, 3.0)
    scorer.terms[0].scale = 2.0
    new = scorer.jitted(pose)
    # Even first tracing after mutation preserves the original render snapshot.
    np.testing.assert_allclose(old(pose.coords), [18, 18])
    np.testing.assert_allclose(new(pose.coords), [54, 54])
    per_term = scorer.render_whole_pose_scoring_module(
        pose, sum_terms=False, apply_weights=False
    )
    np.testing.assert_allclose(per_term(pose.coords), [[18, 18]])
    energy, grad = scorer.jitted_value_and_grad(pose)(pose.coords)
    np.testing.assert_allclose(energy, [54, 54])
    np.testing.assert_allclose(grad, np.full((2, 3, 3), 12.0))

    scorer = _quadratic_scorer(monkeypatch)
    pose = SimpleNamespace(coords=jnp.ones((1, 2, 3)), packed_block_types=None)
    scorer.set_weight(ScoreType.fa_ljatr, 0)
    fn = scorer.jitted(pose)
    np.testing.assert_array_equal(fn(pose.coords), [0])
    np.testing.assert_array_equal(jax.grad(lambda c: fn(c).sum())(pose.coords), 0)
    with pytest.raises(ValueError, match="shape"):
        fn(jnp.ones((1, 3, 3)))
    with pytest.raises(ValueError, match="finite"):
        scorer.set_weight(ScoreType.fa_ljatr, float("nan"))


def _check_constraint_snapshots(backend):
    import jax
    import jax.numpy as jnp
    from tmol_jax import ScoreFunction, ScoreType

    from tmol_jax.pose.constraint_set import ConstraintSet

    from tests.scientific.molecules import MolecularCase

    pose = backend.molecule(
        MolecularCase((), (("GLY",), ("GLY",)), ((1,), (1,))), terms=()
    ).pose
    sf = ScoreFunction(None)
    sf.set_weight(ScoreType.constraint, 1.0)

    def squared_xyz(atoms, params):
        return params[:, 0] * (atoms[:, 0] ** 2).sum(axis=-1)

    def linear_x(atoms, params):
        return params[:, 0] * atoms[:, 0, 0]

    old = []
    for atom, scale, callback in (
        (0, 2.0, squared_xyz),
        (2, 3.0, squared_xyz),
        (2, 5.0, linear_x),
    ):
        cs = ConstraintSet.create_empty(
            device=pose.coords.device, n_poses=2
        ).add_constraints(
            callback,
            jnp.array([[[0, 0, atom]]], jnp.int32),
            jnp.array([[scale]], jnp.float32),
        )
        import attrs

        constrained = attrs.evolve(pose, constraint_set=cs)
        fn = sf.jitted_value_and_grad(constrained)
        expected_gradient = np.zeros(pose.coords.shape, np.float32)
        if callback is squared_xyz:
            expected_energy = scale * np.square(np.asarray(pose.coords)[0, atom]).sum()
            expected_gradient[0, atom] = 2 * scale * np.asarray(pose.coords)[0, atom]
        else:
            expected_energy = scale * float(pose.coords[0, atom, 0])
            expected_gradient[0, atom, 0] = scale
        old.append((fn, [expected_energy, 0.0], expected_gradient))
    # First execution after every mutation must retain each rendered snapshot.
    for fn, energy, gradient in old:
        actual = fn(pose.coords)
        np.testing.assert_allclose(actual[0], energy)
        np.testing.assert_allclose(actual[1], gradient)
        batched = jax.jit(jax.vmap(fn))(jnp.stack([pose.coords, pose.coords]))
        np.testing.assert_allclose(batched[0], [energy, energy])
        np.testing.assert_allclose(batched[1], [gradient, gradient])


def _check_charge_snapshots(b, case):
    import attrs
    from tmol_core.scoring.elec import partial_charge_table, partial_charges_for_block
    from .molecules import DOMAIN_CASES

    ligand, fused = case == "ligand_charges", case == "mixed_fused_charges"
    b.molecule(
        CASES["ligand"] if ligand else DOMAIN_CASES["mixed_batch"], terms=("fa_elec",)
    )
    db, pose = b.parameter_database, b.pose
    frontend = import_module("tmol_jax" if b.framework == "jax" else "tmol")
    ScoreType = frontend.ScoreType

    def render(database, term=ScoreType.fa_elec):
        sf = (
            frontend.ScoreFunction(database)
            if b.framework == "jax"
            else frontend.ScoreFunction(database, b.device)
        )
        sf.set_weight(term, 1)
        if fused:
            sf.set_weight(ScoreType.fa_ljrep, 1)
        if b.framework == "jax" and not ligand and not fused:
            return sf.jitted_value_and_grad(pose)
        if b.framework == "jax":
            rendered = sf.render_whole_pose_scoring_module(pose, sum_terms=not fused)
        else:
            b.score_function = sf
            b._render_torch()
            rendered = b.scorer
        index = list(sf.all_score_types()).index(term)

        def objective(x):
            values = (
                rendered(x)
                if b.framework == "jax"
                else rendered(x, sum_terms=not fused)
            )
            return (values[index] if fused else values).sum()

        if b.framework == "jax":
            evaluate = b.module.value_and_grad(objective)
            return evaluate if ligand else b.module.jit(evaluate)

        def evaluate(x):
            x = x.detach().clone().requires_grad_(True)
            value = objective(x)
            return value.detach(), b.module.autograd.grad(value, x)[0].detach()

        return evaluate

    original = render(db)
    if not ligand:
        expected = tuple(b.numpy(x) for x in original(pose.coords))
        assert all(np.linalg.norm(x) > 1e-3 for x in expected)
    factor, tolerance = (2, 3e-6) if ligand else (1.5, 3e-5)
    changed = attrs.evolve(
        db,
        scoring=attrs.evolve(
            db.scoring,
            elec=attrs.evolve(
                db.scoring.elec,
                atom_charge_parameters=tuple(
                    attrs.evolve(row, charge=factor * row.charge)
                    for row in db.scoring.elec.atom_charge_parameters
                ),
            ),
        ),
    )
    altered = render(changed)
    if ligand:
        expected = tuple(b.numpy(x) for x in original(pose.coords))
        assert abs(expected[0]) > 1e-3 and np.linalg.norm(expected[1]) > 1e-3

    def check(function, scale):
        for actual, reference in zip(function(pose.coords), expected):
            np.testing.assert_allclose(
                b.numpy(actual), scale * reference, rtol=tolerance, atol=tolerance
            )

    check(altered, factor**2)
    if ligand:
        # Keep the original value-only retained-scorer check at its tighter bound.
        np.testing.assert_allclose(
            b.numpy(original(pose.coords)[0]), expected[0], rtol=3e-6, atol=3e-6
        )
        missing_type = pose.block_type(0, 0).atoms[0].atom_type
        incomplete = attrs.evolve(
            db,
            scoring=attrs.evolve(
                db.scoring,
                ljlk=attrs.evolve(
                    db.scoring.ljlk,
                    atom_type_parameters=tuple(
                        p
                        for p in db.scoring.ljlk.atom_type_parameters
                        if p.name != missing_type
                    ),
                ),
            ),
        )
        with pytest.raises(ValueError, match="missing or non-finite LJLK parameters"):
            render(incomplete, ScoreType.fa_ljrep)
        return
    check(render(db), 1)
    if b.framework != "jax":
        return
    # Mutable frontend block annotations are a JAX preparation capability.
    table = partial_charge_table(db.scoring.elec)
    types = pose.packed_block_types.active_block_types
    previous = [getattr(bt, "elec_partial_charges", None) for bt in types]
    try:
        for bt, override in zip(types, previous):
            charges = partial_charges_for_block(bt, table, overrides=override)
            bt.elec_partial_charges = {
                atom.name: float(2 * charge) for atom, charge in zip(bt.atoms, charges)
            }
        doubled = render(db)
        check(doubled, 4)
        for bt in types:
            for atom in bt.elec_partial_charges:
                bt.elec_partial_charges[atom] *= 2
        check(render(db), 16)
        check(original, 1)
        check(altered, 2.25)
        check(doubled, 4)
    finally:
        for bt, override in zip(types, previous):
            if override is None:
                del bt.elec_partial_charges
            else:
                bt.elec_partial_charges = override
    check(render(db), 1)
    if b.framework == "jax" and not fused:
        _check_charge_cache_ownership()


def _check_charge_cache_ownership():
    import attrs
    import jax
    from tmol_jax import (
        beta2016_score_function,
        pose_stack_from_pdb,
        ScoreFunction,
        ScoreType,
    )
    from .test_canonical_molecular_input import peptide_pdb

    pose = pose_stack_from_pdb(peptide_pdb(3))
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
