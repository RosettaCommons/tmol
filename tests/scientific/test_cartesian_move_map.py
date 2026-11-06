"""Public Cartesian selection preserves fixed atoms in real molecular objectives."""

from importlib import import_module

import numpy as np
import pytest

from .molecules import DOMAIN_CASES


@pytest.mark.parametrize("name", ("ligand", "mixed_batch"))
def test_cartesian_move_map_and_public_coordinate_mask(backend, name):
    b = backend.molecule(DOMAIN_CASES[name], terms=())
    b.use_beta2016()
    api = import_module("tmol_jax" if b.framework == "jax" else "tmol")
    assert api.CartesianMoveMap().coord_mask is None
    before = b.coords().copy()
    mask = np.zeros(before.shape[:2], bool)
    mask[:, 1::3] = True
    device_mask = (
        b.module.device_put(mask, b.device)
        if b.framework == "jax"
        else b.module.as_tensor(mask, device=b.device)
    )
    movement = api.CartesianMoveMap(coord_mask=device_mask)
    assert movement.coord_mask is device_mask
    options = dict(optimizer_kwargs={"max_iter": 3, "fixed_iterations": True})
    if b.framework == "jax":
        options["optimizer"] = "armijo"
    out = api.run_cart_min(
        b.pose, b.score_function, coord_mask=movement.coord_mask, **options
    )
    after = b.numpy(out.coords)
    np.testing.assert_array_equal(b.coords(), before)
    np.testing.assert_array_equal(after[~mask], before[~mask])
    assert np.isfinite(after).all() and np.linalg.norm(after - before) > 1e-4
    assert b.scores(after).sum() < b.scores(before).sum() - 1e-3
