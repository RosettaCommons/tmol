"""Checkpoint continuation retains optimizer history and fixed molecular atoms."""

import copy
from contextlib import nullcontext

import numpy as np
import pytest

from .molecules import DOMAIN_CASES


def checkpoint_solutions(
    b, initial, objective, ids, mask=None, *, verify_host_roundtrip=False
):
    settings = dict(history_size=2, fixed_iterations=True, atol=0, rtol=0, gradtol=0)
    if b.framework == "jax":
        from tmol_jax.optimization import armijo_minimize

        x = b.module.device_put(initial, b.device)
        segments = b.module.device_put(ids, b.device)
        first = armijo_minimize(
            objective, x, segment_ids=segments, grad_mask=mask, max_iter=3, **settings
        )
        frozen = copy.deepcopy(first["checkpoint"])
        resumed = armijo_minimize(objective, first["x"], checkpoint=frozen, max_iter=3)
        if verify_host_roundtrip:
            host = b.module.tree.map(
                lambda v: np.asarray(v) if isinstance(v, b.module.Array) else v,
                frozen,
            )
            copied = armijo_minimize(objective, first["x"], checkpoint=host, max_iter=3)
            np.testing.assert_array_equal(copied["x"], resumed["x"])
            assert copied["x"].dtype == np.float64 and copied["x"].device == b.device
            leaves, structure = b.module.tree.flatten(copied["state"])
            reference, expected_structure = b.module.tree.flatten(resumed["state"])
            assert structure == expected_structure
            for value, expected in zip(leaves, reference):
                if isinstance(expected, b.module.Array):
                    assert isinstance(value, b.module.Array)
                    assert value.device == b.device and value.dtype == expected.dtype
            # Valid shapes still reject changed execution context before scoring.
            for altered, options, message in (
                (first["x"] + 0.1, {}, "checkpoint x"),
                (first["x"], {"segment_ids": np.zeros_like(ids)}, "segment_ids"),
                (first["x"], {"grad_mask": np.arange(x.size) != 1}, "grad_mask"),
                (first["x"], {"history_size": 1}, "history capacity"),
            ):
                with pytest.raises(ValueError, match=message):
                    armijo_minimize(objective, altered, checkpoint=host, **options)
        whole = armijo_minimize(
            objective, x, segment_ids=segments, grad_mask=mask, max_iter=6, **settings
        )
        assert resumed["n_iter"] == whole["n_iter"] == 6
        assert frozen["state"]["n_iter"] == 3
        return b.numpy(resumed["x"]), b.numpy(whole["x"])
    from tmol.optimization._lbfgs_armijo import LBFGS_Armijo

    segments = b.module.as_tensor(ids, device=b.device)
    mask = None if mask is None else b.module.as_tensor(mask, device=b.device)

    def make(values, iterations):
        x = b.module.tensor(
            values, device=b.device, dtype=b.module.float64, requires_grad=True
        )
        optimizer = LBFGS_Armijo(
            [x], segment_ids=segments, max_iter=iterations, **settings
        )
        return x, optimizer

    def step(x, optimizer):
        def closure():
            optimizer.zero_grad()
            values = objective(x)
            values.sum().backward()
            if mask is not None:
                x.grad *= mask
            return values

        optimizer.step(closure)

    first_x, first = make(initial, 3)
    step(first_x, first)
    checkpoint = copy.deepcopy(first.state_dict())
    result, resumed = make(b.numpy(first_x), 3)
    resumed.load_state_dict(checkpoint)
    step(result, resumed)
    expected, whole = make(initial, 6)
    step(expected, whole)
    assert resumed.state[result]["n_iter"] == whole.state[expected]["n_iter"] == 6
    return b.numpy(result), b.numpy(expected)


def test_segmented_armijo_checkpoint_matches_uninterrupted_history(backend):
    b = backend
    with b.module.enable_x64() if b.framework == "jax" else nullcontext():
        initial = np.array([2.0, -3.0, 1.0, 4.0, -1.0], np.float64)
        ids = np.array([0, 1, 0, 0, 1], np.int32)
        stack = b.module.numpy.stack if b.framework == "jax" else b.module.stack

        def objective(x):
            values = (x - 0.3) ** 4 + 0.2 * (x + 1.7) ** 2
            return stack((values[0] + values[2] + values[3], values[1] + values[4]))

        result, expected = checkpoint_solutions(
            b, initial, objective, ids, verify_host_roundtrip=True
        )
        np.testing.assert_allclose(result, expected, rtol=2e-12, atol=2e-12)
        assert np.linalg.norm(result - initial) > 1

        def potential(x):
            return np.sum((x - 0.3) ** 4 + 0.2 * (x + 1.7) ** 2)

        assert potential(result) < potential(initial)


@pytest.mark.parametrize("name", ("ligand", "mixed_batch"))
def test_molecular_checkpoint_retains_full_objective_history_and_frozen_atoms(
    backend, name
):
    b = backend.molecule(DOMAIN_CASES[name], terms=())
    b.use_beta2016()
    with b.module.enable_x64() if b.framework == "jax" else nullcontext():
        initial = b.coords().astype(np.float64)
        score = b.score_function.render_whole_pose_scoring_module(b.pose)
        ids = np.broadcast_to(
            np.arange(initial.shape[0])[:, None, None], initial.shape
        ).astype(np.int32)
        mask = np.ones(initial.shape, bool)
        mask[:, ::3] = False

        def objective(x):
            return score(x.reshape(initial.shape))

        result, expected = checkpoint_solutions(
            b, initial.reshape(-1), objective, ids.reshape(-1), mask.reshape(-1)
        )
        np.testing.assert_allclose(result, expected, rtol=2e-10, atol=2e-10)
        np.testing.assert_array_equal(
            result[~mask.reshape(-1)], initial.reshape(-1)[~mask.reshape(-1)]
        )
        assert (
            np.isfinite(result).all()
            and np.linalg.norm(result - initial.reshape(-1)) > 1e-4
        )
        assert (
            b.scores(result.reshape(initial.shape)).sum()
            < b.scores(initial).sum() - 1e-3
        )
