"""Invalid third-party optimizer results cannot alter molecular array contracts."""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.scientific.execution import Execution
from tests.scientific.molecules import CASES
from tests.scientific.test_custom_optimizer import gradient_step
from tests.jax.test_compact_minimization import device_name
from tmol_jax import run_cart_min


@pytest.mark.parametrize(
    "invalid", ("shape", "dtype", "finite", "initial_energy", "energy_shape")
)
def test_invalid_optimizer_result_rejects_without_mutating_pose(invalid):
    backend = Execution(device_name()).molecule(CASES["ligand"], terms=("ref",))
    initial = backend.coords().copy()

    def optimizer(*args, **kwargs):
        result = gradient_step(*args, lr=0.1, **kwargs)
        if invalid == "shape":
            result["grad"] = result["grad"][:-1]
        elif invalid == "dtype":
            result["x"] = result["x"].astype(jnp.float16)
        elif invalid == "initial_energy":
            result["initial_fun"] = jnp.nan
        elif invalid == "energy_shape":
            result["fun"] = jnp.zeros((2,))
        else:
            result["grad"] = result["grad"].at[0].set(jnp.nan)
        return result

    with pytest.raises((ValueError, FloatingPointError), match="optimizer"):
        run_cart_min(backend.pose, backend.score_function, optimizer=optimizer)
    np.testing.assert_array_equal(backend.coords(), initial)
