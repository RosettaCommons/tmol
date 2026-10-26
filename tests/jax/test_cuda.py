"""Optional native CUDA adapter: loading rules and agreement with portable JAX."""

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tmol_jax import beta2016_score_function, pose_stack_from_pdb
from tmol_jax.score import _cuda as cuda_loader
from tmol_jax.score.elec import cuda as elec_cuda
from tmol_jax.score.hbond import cuda as hbond_cuda
from tmol_jax.score.ljlk import cuda
from tmol_jax.score.lk_ball import cuda as lkball_cuda

PDB = Path(__file__).parents[2] / "tmol/tests/data/pdb/1ubq.pdb"
NATIVE = jax.default_backend() == "gpu" and all(
    module.available() for module in (cuda, elec_cuda, hbond_cuda, lkball_cuda)
)
needs_native = pytest.mark.skipif(not NATIVE, reason="native CUDA adapter unavailable")


def test_plugin_is_optional(monkeypatch):
    def missing(_name):
        raise ImportError

    cuda_loader.load_library.cache_clear()
    monkeypatch.setattr(cuda_loader.importlib, "import_module", missing)
    assert cuda_loader.load_library() is None
    cuda_loader.load_library.cache_clear()


def test_plugin_abi_mismatch_is_explicit(monkeypatch):
    class Marker:
        def __call__(self):
            return cuda_loader.ABI_VERSION + 1

    library = SimpleNamespace(tmol_jax_cuda_abi_version=Marker())
    plugin = SimpleNamespace(library_path=lambda: "/unused/plugin.so")
    cuda_loader.load_library.cache_clear()
    monkeypatch.setattr(cuda_loader.importlib, "import_module", lambda _: plugin)
    monkeypatch.setattr(cuda_loader.ctypes.cdll, "LoadLibrary", lambda _: library)
    with pytest.raises(RuntimeError, match="ABI.*does not match"):
        cuda_loader.load_library()
    cuda_loader.load_library.cache_clear()


@needs_native
def test_native_pair_kernels_match_portable_jax(monkeypatch):
    """Per-channel energies and the weighted gradient agree on a real protein."""
    pose = pose_stack_from_pdb(PDB)

    def evaluate():
        sf = beta2016_score_function()
        channels = sf.render_whole_pose_scoring_module(
            pose, sum_terms=False, apply_weights=False
        )
        total = sf.render_whole_pose_scoring_module(pose)
        grad = jax.grad(lambda x: total(x).sum())
        return np.asarray(channels(pose.coords)), np.asarray(grad(pose.coords))

    native_energy, native_gradient = evaluate()
    cuda_loader.register_ffi_targets.cache_clear()
    monkeypatch.setattr(cuda_loader, "load_library", lambda: None)
    portable_energy, portable_gradient = evaluate()
    monkeypatch.undo()
    cuda_loader.register_ffi_targets.cache_clear()

    np.testing.assert_allclose(native_energy, portable_energy, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(native_gradient, portable_gradient, rtol=1e-3, atol=2e-2)


@needs_native
def test_fused_tile_bounds_recompute_after_large_coordinate_move():
    """Tile pruning is rebuilt from every coordinate argument, not frozen."""
    n_atoms = 2048
    values = jnp.ones((1, n_atoms), dtype=jnp.float32)
    real = jnp.zeros((1, n_atoms), dtype=jnp.bool_).at[0, (0, n_atoms - 1)].set(True)
    paths = jnp.full((1, n_atoms, n_atoms), 6, dtype=jnp.uint8)
    parameters = SimpleNamespace(
        lj_radius=values * 1.5,
        lj_wdepth=values,
        lk_dgfree=values * -0.1,
        lk_lambda=values * 3.5,
        lk_volume=values * 12.0,
        is_donor=values < 0,
        is_hydroxyl=values < 0,
        is_polarh=values < 0,
        is_acceptor=values < 0,
        is_carbon_lk=values < 0,
    )
    ljlk_state = SimpleNamespace(
        parameters=parameters,
        global_parameters=SimpleNamespace(
            max_dis=jnp.float32(6),
            lj_dlin_sigma_factor=jnp.float32(0.6),
            lj_hbond_dis=jnp.float32(3),
            lj_hbond_OH_donor_dis=jnp.float32(2.8),
            lj_hbond_hdis=jnp.float32(1.9),
        ),
        bonded_path_length=paths,
        atom_is_real=real,
    )
    elec_parameters = SimpleNamespace(
        D=jnp.float32(80),
        D0=jnp.float32(6),
        S=jnp.float32(0.4),
        min_dis=jnp.float32(1.5),
        max_dis=jnp.float32(5.5),
    )
    elec_state = SimpleNamespace(
        charges=values.at[0, n_atoms - 1].set(-1),
        bonded_path_length=paths,
        atom_is_real=real,
        cuda_global_parameters=elec_cuda._pack_global_parameters(elec_parameters),
    )
    far = jnp.zeros((1, n_atoms, 3), dtype=jnp.float32)
    far = far.at[0, n_atoms - 1, 0].set(20)
    near = far.at[0, n_atoms - 1, 0].set(4)
    weights = jnp.asarray([0.7, -0.2, 1.3, 0.5], dtype=jnp.float32)

    def separate(coordinates):
        ljlk = cuda.score(ljlk_state, coordinates)
        elec = elec_cuda.score(elec_state, coordinates)
        return jnp.concatenate((ljlk, elec[:, None]), axis=1)

    def fused(coordinates):
        return cuda.score_fused(ljlk_state, elec_state, coordinates)

    separate_vg = jax.jit(jax.value_and_grad(lambda x: (separate(x) * weights).sum()))
    fused_vg = jax.jit(jax.value_and_grad(lambda x: (fused(x) * weights).sum()))
    for coordinates in (far, near):
        separate_value, separate_gradient = separate_vg(coordinates)
        fused_value, fused_gradient = fused_vg(coordinates)
        np.testing.assert_allclose(fused_value, separate_value, atol=1e-5, rtol=1e-6)
        np.testing.assert_allclose(
            fused_gradient, separate_gradient, atol=1e-5, rtol=1e-5
        )
    assert fused_vg(far)[0] == 0
    assert fused_vg(near)[0] != 0
