"""Fixed sparse identities support compiled native coordinate reverse rules."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.scientific.execution import Execution
from tests.scientific.molecules import DOMAIN_CASES
from tests.scientific.hbond_molecules import HBOND_CASES
from tests.scientific.dunbrack_molecules import DUNBRACK_CASES
from tests.scientific.rotamer_execution import RotamerExecution
from tmol_jax.pack.scoring import sparse_nonbonded_values, nonbonded_candidate_pairs


@pytest.mark.parametrize("platform", ("cpu", "cuda"))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize(
    "family", ("ljlk", "elec", "dunbrack", "lk_ball", "genbonded", "hbond")
)
def test_sparse_values_reverse_batching_and_precision(platform, dtype, family):
    b = Execution("jax-" + platform)
    with jax.enable_x64(dtype == np.float64):
        cases = DUNBRACK_CASES if family == "dunbrack" else DOMAIN_CASES
        if family == "hbond":
            cases = HBOND_CASES
        b.molecule(cases["mixed_batch"], terms=("ref",))
        execute = RotamerExecution(b, family=family)
        coords = jax.device_put(
            np.asarray(execute.rotamers.coords, dtype=dtype), b.device
        )
        if family not in ("dunbrack", "genbonded"):
            indices = nonbonded_candidate_pairs(
                execute.rotamers,
                b.pose.packed_block_types,
                reach=execute.state[3],
                coords=coords,
            )
        before = np.asarray(coords).copy()

        def total(x):
            if family == "genbonded":
                from tmol_jax.pack.bonded_scoring import sparse_genbonded_scores

                values = sparse_genbonded_scores(x, execute.state)[0]
            elif family == "dunbrack":
                from tmol_jax.pack.dunbrack_scoring import sparse_dunbrack_scores

                values = sparse_dunbrack_scores(x, execute.state)[0]
            elif family in ("lk_ball", "hbond"):
                from tmol_jax.pack.lk_ball_scoring import sparse_lk_ball_values
                from tmol_jax.pack.hbond_scoring import sparse_hbond_values

                evaluate = (
                    sparse_lk_ball_values
                    if family == "lk_ball"
                    else sparse_hbond_values
                )
                values = evaluate(x, execute.state, indices)
            else:
                values = sparse_nonbonded_values(x, execute.state, indices)
            return jnp.sum(
                values
                * jnp.asarray([0.7, -0.3, 0.6, -0.2][: values.shape[0]], dtype)[:, None]
            )

        value, gradient = jax.jit(jax.value_and_grad(total))(coords)
        batch = jnp.stack(
            (
                coords,
                coords + jax.device_put(np.asarray([2.0, -3.0, 4.0], dtype), b.device),
            )
        )
        batched_value, batched_gradient = jax.jit(jax.vmap(jax.value_and_grad(total)))(
            batch
        )
        tolerance = 2e-4 if dtype == np.float32 else 2e-11
        np.testing.assert_allclose(
            batched_value, [value, value], rtol=tolerance, atol=tolerance
        )
        np.testing.assert_allclose(
            batched_gradient,
            np.stack((gradient, gradient)),
            rtol=tolerance,
            atol=tolerance,
        )
        assert value.dtype == gradient.dtype == dtype
        assert value.device == gradient.device == b.device
        np.testing.assert_array_equal(coords, before)
        with pytest.raises(TypeError, match="forward-mode"):
            jax.jvp(total, (coords,), (jnp.ones_like(coords),))


@pytest.mark.parametrize("family", ("ljlk", "lk_ball"))
def test_sparse_preparation_preserves_pose_and_parameter_context(family):
    import attrs
    from tmol_jax.pack.scoring import prepare_rotamer_nonbonded
    from tmol_jax.pack.lk_ball_scoring import prepare_rotamer_lk_ball

    b = Execution("jax-cpu")
    b.molecule(DOMAIN_CASES["mixed_batch"], terms=("ref",))
    e = RotamerExecution(b, family=family)
    prepare = prepare_rotamer_nonbonded if family == "ljlk" else prepare_rotamer_lk_ball
    pose, db = b.pose, b.parameter_database
    prepare(pose.replace(coords=pose.coords + 1), e.rotamers, db)
    with pytest.raises(ValueError, match="different pose topology"):
        prepare(
            pose.replace(block_type_ind=pose.block_type_ind.at[0, 0].set(-1)),
            e.rotamers,
            db,
        )
    chemical = attrs.evolve(db.chemical, atom_types=db.chemical.atom_types[::-1])
    with pytest.raises(ValueError, match="atom-type order"):
        prepare(pose, e.rotamers, attrs.evolve(db, chemical=chemical))
    # Static provenance survives ordinary pytree transport.
    transported = jax.jit(lambda r: r)(e.rotamers)
    prepare(pose, transported, db)


@pytest.mark.parametrize("platform", ("cpu", "cuda"))
@pytest.mark.parametrize("coupled", (False, True))
def test_connected_dispatch_preserves_lockstep_identities(platform, coupled):
    from tmol_jax.pack.bonded_scoring import _connected_stage
    from tmol_jax.pack.graph import prefix_counts, load_balance

    device = Execution("jax-" + platform).device
    with jax.default_device(device), jax.enable_x64(False):
        groups = [[0, 0]] if coupled else [[-1, -1]]
        args = tuple(
            jnp.asarray(a, jnp.int32)
            for a in (
                [0, 0, 0, 1, 1, 1],
                [0] * 6,
                [0] * 6,
                [[3, 3]],
                groups,
                [[[[1, 0]], [[0, 0]]]],
            )
        )
        counts = _connected_stage(
            args, operation=0, outputs=jax.ShapeDtypeStruct((12, 8), np.uint8)
        )
        offsets = prefix_counts(counts)
        total = int(np.asarray(offsets[-1]).view(np.int64)[0])
        owners = load_balance(offsets[:-1], total=total)
        actual = _connected_stage(
            (*args, jnp.asarray([[0, 3]], jnp.int32), offsets[:-1], owners),
            operation=1,
            outputs=jax.ShapeDtypeStruct((3, total), np.int32),
        )
        expected = []
        for rot in range(6):
            if rot < 3:
                partners = (rot + 3,) if coupled else range(3, 6)
                expected.extend((0, rot, other) for other in partners)
            expected.append((0, rot, rot))
        np.testing.assert_array_equal(actual, np.asarray(expected).T)
        assert actual.device == device


@pytest.mark.parametrize(
    "family", ("ljlk", "elec", "lk_ball", "dunbrack", "genbonded", "hbond")
)
def test_sparse_preparation_respects_cpu_pose_on_gpu_host(family):
    from tmol_jax.pack.scoring import prepare_rotamer_nonbonded
    from tmol_jax.pack.lk_ball_scoring import prepare_rotamer_lk_ball
    from tmol_jax.pack.dunbrack_scoring import prepare_rotamer_dunbrack
    from tmol_jax.pack.bonded_scoring import prepare_rotamer_genbonded
    from tmol_jax.pack.hbond_scoring import prepare_rotamer_hbond

    try:
        gpu = jax.devices("cuda")[0]
    except RuntimeError:
        pytest.skip("requires a GPU host with an independently selected CPU pose")
    b = Execution("jax-cpu")
    cases = DUNBRACK_CASES if family == "dunbrack" else DOMAIN_CASES
    if family == "hbond":
        cases = HBOND_CASES
    b.molecule(cases["mixed_batch"], terms=("ref",))
    e = RotamerExecution(b, family=family)
    prepare = {
        "lk_ball": prepare_rotamer_lk_ball,
        "dunbrack": prepare_rotamer_dunbrack,
        "genbonded": prepare_rotamer_genbonded,
        "hbond": prepare_rotamer_hbond,
    }.get(family, prepare_rotamer_nonbonded)
    with jax.default_device(gpu):
        state = prepare(
            b.pose,
            e.rotamers,
            b.parameter_database,
            **({"electrostatic": True} if family == "elec" else {}),
        )
    arrays = [x for x in jax.tree.leaves(state) if isinstance(x, jax.Array)]
    assert arrays and all(x.device == b.device for x in arrays)
