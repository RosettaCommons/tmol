"""Frontend initialization, native package ABI and device workspace contracts."""

import ctypes
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import textwrap

import numpy as np
import pytest


@pytest.fixture
def native_runtime(backend):
    if backend.framework != "jax":
        pytest.skip("JAX plugin initialization and FFI ABI contract")
    from tmol_jax.score import _native

    return _native


def test_public_import_is_lazy_and_resolves_existing_modules(native_runtime):
    # Preserve both import orders in separate interpreters: lazy discovery and
    # importing the frontend after an already configured array framework.
    for code in (
        """
            import sys
            import tmol_jax

            assert 'jax' not in sys.modules
            assert 'torch' not in sys.modules
            assert 'tmol_jax_native' not in sys.modules
            assert set(tmol_jax.__all__) <= set(dir(tmol_jax))
            from tmol_jax import ScoreType, PoseStack, io
            from tmol_jax.score import ScoreType as direct_score_type
            from tmol_jax.pose import PoseStack as direct_pose_stack

            assert ScoreType is direct_score_type
            assert PoseStack is direct_pose_stack
            assert io is sys.modules['tmol_jax.io']
            assert tmol_jax.PoseStack is PoseStack
            assert 'torch' not in sys.modules
            try:
                tmol_jax.no_such_public_api
            except AttributeError:
                pass
            else:
                raise AssertionError('unknown public name must raise')
        """,
        """
            import sys
            import jax
            initial_x64 = jax.config.x64_enabled
            sys.path.insert(0, sys.argv[1])
            import tmol_jax
            from tmol_jax.score import ScoreFunction
            assert "torch" not in sys.modules
            assert jax.config.x64_enabled == initial_x64
        """,
    ):
        subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent(code),
                str(Path(native_runtime.__file__).parents[2]),
            ],
            check=True,
            capture_output=True,
            text=True,
        )


@pytest.mark.parametrize("failure", ("missing", "abi"))
def test_native_abi_mismatch_is_explicit(native_runtime, monkeypatch, failure):
    def missing(_name):
        raise ImportError

    class Marker:
        def __call__(self):
            return native_runtime.ABI_VERSION + 1

    library = SimpleNamespace(tmol_jax_native_abi_version=Marker())
    plugin = SimpleNamespace(library_path=lambda _: "/unused/plugin.so")
    native_runtime.libraries.cache_clear()
    monkeypatch.setattr(
        native_runtime,
        "import_module",
        missing if failure == "missing" else lambda _: plugin,
    )
    monkeypatch.setattr(native_runtime.ctypes, "CDLL", lambda _: library)
    try:
        if failure == "missing":
            assert native_runtime.libraries() == {}
            with pytest.raises(RuntimeError, match="requires tmol-jax-native"):
                native_runtime.register("test_missing_native")
        else:
            with pytest.raises(RuntimeError, match="ABI mismatch"):
                native_runtime.libraries()
    finally:
        native_runtime.libraries.cache_clear()


def test_fused_scorer_targets_cpu_in_gpu_process(backend, native_runtime):
    if backend.requested_device != "cuda":
        pytest.skip("CUDA process placement contract")
    jax = backend.module
    from tmol_jax import beta2016_score_function, pose_stack_from_pdb

    pdb = Path(__file__).parents[2] / "tmol/tests/data/pdb/1ubq.pdb"
    text = (
        "\n".join(
            line
            for line in pdb.read_text().splitlines()
            if line.startswith("ATOM") and 1 <= int(line[22:26]) <= 3
        )
        + "\nEND\n"
    )
    with jax.default_device(backend.device):
        short_native_pose = pose_stack_from_pdb(text)
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        pose = short_native_pose.to(cpu)
        sf = beta2016_score_function()
        score = sf.render_whole_pose_scoring_module(pose)
    for evaluate in (score, jax.jit(score)):
        value = evaluate(pose.coords)
        assert value.device.platform == "cpu"
        assert np.isfinite(value).all()


def test_cuda_memory_query_and_failure_restore_device(backend, native_runtime):
    from tmol_jax.pack.batching import cuda_memory_info
    from .molecules import DOMAIN_CASES

    jax = backend.module
    for invalid in ("cuda", 0, jax.devices("cpu")[0]):
        with pytest.raises(ValueError, match="local JAX GPU device"):
            cuda_memory_info(invalid)
    if backend.requested_device != "cuda":
        return

    # The adapter links the CUDA runtime privately. Observe the actual current
    # device through the independent public driver ABI instead of its symbols.
    devices = jax.local_devices(backend="gpu")
    free, total = cuda_memory_info(devices[0])
    assert 0 < free <= total
    driver = ctypes.CDLL("libcuda.so.1")
    get_device = driver.cuCtxGetDevice
    get_device.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_device.restype = ctypes.c_int
    previous = ctypes.c_int()
    assert get_device(ctypes.byref(previous)) == 0
    for device in devices:
        free, total = cuda_memory_info(device)
        assert 0 < free <= total
    current = ctypes.c_int()
    assert get_device(ctypes.byref(current)) == 0 and current.value == previous.value
    values = jax.device_put(np.arange(7, dtype=np.float32), devices[0])
    execute = jax.jit(lambda x: 2 * x + 1)
    expected = np.asarray(execute(values)).copy()
    molecule = backend.molecule(DOMAIN_CASES["ligand"])
    native = molecule.score_function.jitted_value_and_grad(molecule.pose)
    native_reference = tuple(np.asarray(x) for x in native(molecule.pose.coords))
    assert np.linalg.norm(native_reference[1]) > 0
    for invalid in (-1, np.iinfo(np.int32).max):
        free, total = ctypes.c_uint64(17), ctypes.c_uint64(29)
        assert (
            native_runtime.libraries()["cuda"].tmol_cuda_memory_info(
                int(invalid), ctypes.byref(free), ctypes.byref(total)
            )
            != 0
        )
        assert (free.value, total.value) == (17, 29)
        assert (
            get_device(ctypes.byref(current)) == 0 and current.value == previous.value
        )
        # A rejected host query must leave subsequent real device work usable,
        # including when an XLA executable is already cached.
        np.testing.assert_array_equal(execute(values), expected)
        for actual, reference in zip(native(molecule.pose.coords), native_reference):
            np.testing.assert_allclose(actual, reference, rtol=3e-6, atol=3e-5)


@pytest.mark.parametrize(
    "target,dimensions",
    (
        ("dunbrack", (3, 7)),
        ("dunbrack_energy", (3, 7, 2)),
        ("nonbonded", (2, 3)),
        ("nonbonded_weighted", (2, 3)),
        ("nonbonded_precise", (2, 3, 17)),
        ("nonbonded_weighted_precise", (2, 3, 17)),
        ("ljlk", (2, 3)),
        ("ljlk_precise", (2, 3, 17)),
        ("elec", (2, 3)),
        ("inverse_kinematics", (7,)),
        ("lk_ball", (2, 3, 17)),
        ("lk_ball_weighted", (2, 3, 17)),
        ("lk_ball_precise", (2, 3, 17)),
        ("lk_ball_weighted_precise", (2, 3, 17)),
        ("hbond", (2, 3, 17)),
        ("rotamer_lk_ball", (2, 3, 17)),
        ("rotamer_hbond", (2, 3, 17)),
        ("graph_prefix", (7,)),
    ),
)
def test_workspace_query_precision_and_invalid_extents(
    native_runtime, target, dimensions
):
    query = getattr(
        native_runtime.libraries()["cpu"], "tmol_" + target + "_workspace_bytes"
    )
    query.argtypes = [ctypes.c_int64] * len(dimensions) + [ctypes.c_int] * 2
    query.restype = ctypes.c_uint64
    invalid = (1 << 64) - 1
    for derivative in (0, 1):
        for bits in (32, 64):
            size = query(*dimensions, bits, derivative)
            if target == "graph_prefix" and bits == 32:
                assert size == invalid
                continue
            assert 0 < size < invalid
            for axis in range(len(dimensions)):
                for bad in (-1, 1 << 62):
                    broken = list(dimensions)
                    broken[axis] = bad
                    assert query(*broken, bits, derivative) == invalid
        for bits in (0, 16, 128):
            assert query(*dimensions, bits, derivative) == invalid
    if target != "graph_prefix":
        assert query(*dimensions, 64, 0) >= query(*dimensions, 32, 0)


@pytest.mark.parametrize(
    "target", ("tmol_nonbonded", "tmol_nonbonded_weighted", "tmol_ljlk")
)
def test_wide_nonbonded_workspace_only_changes_float_derivatives(
    native_runtime, target
):
    workspace_sizes = native_runtime.workspace_sizes

    for bits in (32, 64):
        old = workspace_sizes(target, (2, 3), bits)
        precise = workspace_sizes(target + "_precise", (2, 3, 17), bits)
        assert old[0] == precise[0]
        if bits == 64:
            assert old == precise
        else:
            assert precise[1] > precise[0]
    # Reject layouts whose derivative element dispatch would narrow to int32.
    with pytest.raises(ValueError, match="workspace layout"):
        workspace_sizes(target + "_precise", (1 << 20, 1, 1 << 20), 32)


def test_cartesian_accumulator_workspace_precision_and_overflow(native_runtime):
    workspace_sizes = native_runtime.workspace_sizes

    target = "tmol_cartbonded_precise"
    energy, derivative = workspace_sizes(target, (2, 17), 32)
    assert derivative > energy > 0
    assert workspace_sizes(target, (2, 17), 64) == (0, 0)
    for dimensions in ((-1, 17), (1 << 62, 17), (1 << 20, 1 << 20)):
        with pytest.raises(ValueError, match="workspace layout"):
            workspace_sizes(target, dimensions, 32)


def test_pose_energy_accumulation_workspace_is_bounded_and_precision_specific(
    native_runtime,
):
    workspace_sizes = native_runtime.workspace_sizes

    target = "tmol_backbone_torsion"
    assert workspace_sizes(target, (2,), 32) == (32, 32)
    assert workspace_sizes(target, (2,), 64) == (0, 0)
    assert workspace_sizes(target, (0,), 32) == (0, 0)
    for target, dimensions in (
        ("tmol_backbone_torsion", (-1,)),
        ("tmol_backbone_torsion", (1 << 30,)),
        ("tmol_dunbrack_energy", (1, 7, 1 << 30)),
    ):
        with pytest.raises(ValueError, match="workspace layout"):
            workspace_sizes(target, dimensions, 32)
