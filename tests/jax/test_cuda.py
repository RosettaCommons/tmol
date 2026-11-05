"""Native ABI failures and explicit CPU placement inside a CUDA process."""

from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from tmol_jax import beta2016_score_function, pose_stack_from_pdb
from tmol_jax.score import _native

PDB = Path(__file__).parents[2] / "tmol/tests/data/pdb/1ubq.pdb"


def test_missing_native_package_is_explicit(monkeypatch):
    def missing(_name):
        raise ImportError

    _native.libraries.cache_clear()
    monkeypatch.setattr(_native, "import_module", missing)
    try:
        assert _native.libraries() == {}
        with pytest.raises(RuntimeError, match="requires tmol-jax-native"):
            _native.register("test_missing_native")
    finally:
        _native.libraries.cache_clear()


def test_native_abi_mismatch_is_explicit(monkeypatch):
    class Marker:
        def __call__(self):
            return _native.ABI_VERSION + 1

    library = SimpleNamespace(tmol_jax_native_abi_version=Marker())
    plugin = SimpleNamespace(library_path=lambda _: "/unused/plugin.so")
    _native.libraries.cache_clear()
    monkeypatch.setattr(_native, "import_module", lambda _: plugin)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda _: library)
    try:
        with pytest.raises(RuntimeError, match="ABI mismatch"):
            _native.libraries()
    finally:
        _native.libraries.cache_clear()


@pytest.fixture(scope="module")
def short_native_pose():
    text = (
        "\n".join(
            line
            for line in PDB.read_text().splitlines()
            if line.startswith("ATOM") and 1 <= int(line[22:26]) <= 3
        )
        + "\nEND\n"
    )
    return pose_stack_from_pdb(text)


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="CUDA device unavailable")
def test_fused_scorer_targets_cpu_in_gpu_process(short_native_pose):
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        pose = short_native_pose.to(cpu)
        sf = beta2016_score_function()
        score = sf.render_whole_pose_scoring_module(pose)
    for evaluate in (score, jax.jit(score)):
        value = evaluate(pose.coords)
        assert value.device.platform == "cpu"
        assert np.isfinite(value).all()
