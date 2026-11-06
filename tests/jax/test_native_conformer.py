"""Native transport preserves retained DOFs, precision and functional arrays."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tmol_jax.pack._native import copy_conformer_dofs, assign_conformer_chi


@pytest.mark.parametrize("platform", ("cpu", "gpu"))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_conformer_dof_transport_preserves_unselected_state(dtype, platform):
    try:
        device = jax.devices(platform)[0]
    except RuntimeError:
        pytest.skip(f"{platform} device unavailable")
    with jax.enable_x64(), jax.default_device(device):
        source = np.arange(36, dtype=dtype).reshape(4, 9)
        source[2, 4] = np.nan
        initial = jnp.full((4, 9), -2, dtype=dtype)

        def integer(value):
            return jnp.asarray(value, jnp.int32)

        # A retained-atom map changes row order. An unresolved input component
        # must leave the initialized target component intact, as must the root.
        metadata = (
            integer([0]),
            integer([0]),
            integer([0]),
            integer([0]),
            integer([0]),
            integer([[0, 1, 2]]),
            integer([[[[2, 0, 1]]]]),
            integer([[0, 1, 2]]),
        )

        def copy(dofs):
            return copy_conformer_dofs(dofs, jnp.asarray(source), *metadata, sampler=0)

        copied = copy(initial)
        expected = np.vstack(([-2] * 9, source[2], source[3], source[1])).astype(dtype)
        expected[1, 4] = -2
        np.testing.assert_array_equal(copied, expected)
        np.testing.assert_array_equal(initial, np.full((4, 9), -2, dtype=dtype))
        np.testing.assert_array_equal(
            jax.jit(jax.vmap(copy))(jnp.stack((initial, initial))),
            np.stack((expected, expected)),
        )
        chi = jnp.asarray([[0.123456789, -1.2]], dtype=dtype)

        def assign(dofs):
            return assign_conformer_chi(
                dofs,
                chi,
                integer([0]),
                integer([0]),
                integer([0]),
                integer([[2, -1]]),
                integer([[0, 1, 2]]),
                jnp.asarray([[0.0, 0.0, 0.2]], dtype=dtype),
            )

        expected[3, 3] = np.asarray(chi)[0, 0] - np.asarray(0.2, dtype=dtype)
        assigned = assign(copied)
        np.testing.assert_array_equal(assigned, expected)
        np.testing.assert_array_equal(
            jax.jit(jax.vmap(assign))(jnp.stack((copied, copied))),
            np.stack((expected, expected)),
        )
        assert copied.device == assigned.device == device
        assert copied.dtype == assigned.dtype == dtype
        # A skipped chi leaves all its components untouched, and assignment
        # does not mutate the previously returned copy operation's result.
        np.testing.assert_array_equal(copied[3], source[1])
