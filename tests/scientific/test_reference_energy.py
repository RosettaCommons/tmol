"""Reference energies preserve chemical identity and explicit D overrides."""

import attrs
import numpy as np
import pytest

from .molecules import CASES, MolecularCase


@pytest.mark.parametrize("explicit_d", [False, True, "snapshots", "owners"])
def test_reference_identity_and_parameter_context(backend, explicit_d):
    if isinstance(explicit_d, str):
        if backend.framework != "jax":
            pytest.skip("JAX rendered-program capability")
        if explicit_d == "snapshots":
            _check_reference_snapshot(backend)
        else:
            _check_reference_owners(backend)
        return
    case = MolecularCase(
        CASES["mixed_batch"].parameter_files,
        (("DALA:nterm", "DLEU:cterm", "HYP", "LG1"), ("ALA", "LEU")),
        ((2, 1, 1), (2,)),
    )
    backend.molecule(case, terms=("ref",))
    original_database = backend.parameter_database
    # Deliberately omit DALA: it must inherit ALA. An explicit DLEU value wins.
    weights = {"ALA": 2.0, "LEU": -3.0, "HYP": 1.25, "LG1": -0.75}
    if explicit_d:
        weights["DLEU"] = 5.0
    database = attrs.evolve(
        original_database,
        scoring=attrs.evolve(
            original_database.scoring,
            ref=attrs.evolve(original_database.scoring.ref, weights=weights),
        ),
    )
    backend.molecule(case, terms=("ref",), parameter_database=database)
    xyz = backend.coords()
    expected = np.array([7.5 if explicit_d else -0.5, -1.0])
    value, gradient = backend.value_and_grad(xyz, coordinate_independent=True)
    np.testing.assert_array_equal(value, expected)
    np.testing.assert_array_equal(gradient, np.zeros_like(xyz))
    np.testing.assert_array_equal(backend.scores(xyz + 7.0), expected)
    # Reusing a palette with a different parameter context must not retain it.
    changed = attrs.evolve(
        database,
        scoring=attrs.evolve(
            database.scoring,
            ref=attrs.evolve(database.scoring.ref, weights={"ALA": 4.0}),
        ),
    )
    backend.molecule(case, terms=("ref",), parameter_database=changed)
    changed_value, changed_gradient = backend.value_and_grad(
        xyz, coordinate_independent=True
    )
    np.testing.assert_array_equal(changed_value, [4.0, 4.0])
    np.testing.assert_array_equal(changed_gradient, np.zeros_like(xyz))
    backend.set_options({"ref_weights": {"ALA": 6.0, "DLEU": 0.0}})
    np.testing.assert_array_equal(backend.scores(xyz), [6.0, 6.0])
    backend.set_options({})
    np.testing.assert_array_equal(backend.scores(xyz), [4.0, 4.0])


def test_reference_sum_retains_small_custom_weights(backend):
    case = MolecularCase(
        CASES["mixed_batch"].parameter_files,
        (("DALA:nterm", "DLEU:cterm", "HYP", "LG1"),),
        ((2, 1, 1),),
    )
    backend.molecule(case, terms=("ref",))
    # The exactly representable total is 1.125. Summing in float32 can lose
    # the unit contribution between the large opposite-sign constants.
    backend.set_options(
        {"ref_weights": {"ALA": 1e8, "LEU": 1.0, "HYP": -1e8, "LG1": 0.125}}
    )
    value, gradient = backend.value_and_grad(
        backend.coords(), coordinate_independent=True
    )
    np.testing.assert_array_equal(value, [1.125])
    np.testing.assert_array_equal(gradient, np.zeros_like(backend.coords()))


def _check_reference_snapshot(backend):
    import jax
    import jax.numpy as jnp

    from tests.scientific.dunbrack_molecules import DUNBRACK_CASES

    backend.molecule(DUNBRACK_CASES["d_peptide"], terms=("ref",))
    sf, pose = backend.score_function, backend.pose
    weights = {"ALA": 1.0, "LEU": 2.0, "ARG": 3.0, "PHE": 4.0}
    sf.set_options({"ref_weights": weights})
    old = sf.render_whole_pose_scoring_module(pose)
    clone = sf.clone()
    weights["ALA"] = 99.0
    sf.set_option("ref_weights", {"DALA": 8.0})
    new = sf.render_whole_pose_scoring_module(pose)
    with jax.enable_x64():
        coords = pose.coords.astype(jnp.float64)
        assert jax.jit(old)(coords).dtype == coords.dtype
        np.testing.assert_array_equal(jax.jit(old)(coords), [10.0])
        np.testing.assert_array_equal(clone.jitted(pose)(coords), [10.0])
        np.testing.assert_array_equal(jax.jit(new)(coords), [8.0])
        values, tangent = jax.jit(lambda x: jax.jvp(old, (x,), (jnp.ones_like(x),)))(
            coords
        )
        np.testing.assert_array_equal(tangent, [0.0])
        np.testing.assert_array_equal(
            jax.jit(jax.vmap(old))(jnp.stack([coords, coords + 3])), [[10.0], [10.0]]
        )
    with pytest.raises(NotImplementedError, match="unknown_option"):
        sf.set_option("unknown_option", True)


def _check_reference_owners(backend):
    import jax
    import jax.numpy as jnp
    from tmol_jax import ScoreFunction, ScoreType

    import attrs
    import gc
    import weakref

    from tests.scientific.dunbrack_molecules import DUNBRACK_CASES

    backend.molecule(DUNBRACK_CASES["d_peptide"], terms=())
    pose, original = backend.pose, backend.parameter_database
    rendered, owners = [], []
    for scale in (2.0, 7.0):
        database = attrs.evolve(
            original,
            scoring=attrs.evolve(
                original.scoring,
                ref=attrs.evolve(
                    original.scoring.ref,
                    weights={"ALA": scale, "LEU": scale, "ARG": scale, "PHE": scale},
                ),
            ),
        )
        owners.append(weakref.ref(database))
        sf = ScoreFunction(database)
        sf.set_weight(ScoreType.ref, 1.0)
        rendered.append(sf.render_whole_pose_scoring_module(pose))
    del database, sf
    gc.collect()
    assert all(owner() is None for owner in owners)
    # First execution after both owners expire must still use each snapshot.
    for scorer, expected in zip(rendered, (8.0, 28.0)):
        values, gradient = scorer.value_and_grad()(pose.coords)
        np.testing.assert_array_equal(values, [expected])
        np.testing.assert_array_equal(gradient, 0)
        np.testing.assert_array_equal(
            jax.jit(jax.vmap(scorer))(jnp.stack([pose.coords, pose.coords])),
            [[expected], [expected]],
        )
