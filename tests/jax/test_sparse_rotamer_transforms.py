"""Fixed sparse identities support compiled native coordinate reverse rules."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.scientific.execution import Execution
from tests.scientific.molecules import DOMAIN_CASES
from tests.scientific.rotamer_execution import RotamerExecution
from tmol_jax.pack.scoring import sparse_nonbonded_values, nonbonded_candidate_pairs


@pytest.mark.parametrize("platform", ("cpu", "cuda"))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("electrostatic", (False, True))
def test_sparse_values_reverse_batching_and_precision(platform, dtype, electrostatic):
    b = Execution("jax-" + platform)
    with jax.enable_x64(dtype == np.float64):
        b.molecule(DOMAIN_CASES["mixed_batch"], terms=("ref",))
        execute = RotamerExecution(b, electrostatic=electrostatic)
        coords = jax.device_put(
            np.asarray(execute.rotamers.coords, dtype=dtype), b.device
        )
        indices = nonbonded_candidate_pairs(
            execute.rotamers,
            b.pose.packed_block_types,
            reach=execute.state[3],
            coords=coords,
        )
        before = np.asarray(coords).copy()

        def total(x):
            values = sparse_nonbonded_values(x, execute.state, indices)
            return jnp.sum(
                values
                * jnp.asarray([0.7, -0.3, 0.6][: values.shape[0]], dtype)[:, None]
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


def test_sparse_preparation_preserves_pose_and_parameter_context():
    import attrs
    from tmol_jax.pack.scoring import prepare_rotamer_nonbonded

    b = Execution("jax-cpu")
    b.molecule(DOMAIN_CASES["mixed_batch"], terms=("ref",))
    e = RotamerExecution(b)
    pose, db = b.pose, b.parameter_database
    prepare_rotamer_nonbonded(pose.replace(coords=pose.coords + 1), e.rotamers, db)
    with pytest.raises(ValueError, match="different pose topology"):
        prepare_rotamer_nonbonded(
            pose.replace(block_type_ind=pose.block_type_ind.at[0, 0].set(-1)),
            e.rotamers,
            db,
        )
    chemical = attrs.evolve(db.chemical, atom_types=db.chemical.atom_types[::-1])
    with pytest.raises(ValueError, match="atom-type order"):
        prepare_rotamer_nonbonded(pose, e.rotamers, attrs.evolve(db, chemical=chemical))
    # Static provenance survives ordinary pytree transport.
    transported = jax.jit(lambda r: r)(e.rotamers)
    prepare_rotamer_nonbonded(pose, transported, db)
