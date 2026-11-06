"""Transformation contracts for prepared tensor-only molecular inputs."""

import jax
import numpy as np
import pytest

from tests.jax.test_atom37_transforms import execution
from tests.scientific.test_atom37_tensor_input import tensor_fixture, bind_tensor


@pytest.mark.parametrize("name", ("ligand", "hydroxyproline"))
def test_tensor_input_jit_vmap_reuses_static_topology(name):
    b, io, source, topology, context = tensor_fixture(execution(), name)
    x, bind = bind_tensor(b, io, source, topology, context)
    shifts = jax.device_put(np.array([[0, 0, 0], [2, 3, -1]], np.float32), b.device)
    coords = jax.jit(jax.vmap(lambda shift: bind(x + shift).coords))(shifts)
    np.testing.assert_allclose(
        coords[1] - coords[0],
        np.broadcast_to(np.asarray(shifts[1]), coords[0].shape),
        atol=2e-5,
        rtol=2e-5,
    )
    invalid = x.at[0, 0, 0, 0].set(np.inf)
    assert np.isnan(np.asarray(jax.jit(lambda y: bind(y).coords)(invalid))).all()
    with pytest.raises(ValueError, match="shape"):
        bind(x[:, :-1])
    # No routing for a named ligand is a hard error, not ideal-coordinate input.
    if name == "ligand":
        with pytest.raises(ValueError, match="routes no atom"):
            io.prepare_atom37_tensor_builder(
                topology["res_types"], topology["chain_id"], context
            )
