"""The native free-memory query preserves the selected CUDA device."""

import ctypes
import os

import jax
import numpy as np
import pytest

from tmol_jax.pack.batching import cuda_memory_info


def test_packing_memory_requires_a_local_cuda_device():
    for invalid in ("cuda", 0, jax.devices("cpu")[0]):
        with pytest.raises(ValueError, match="local JAX GPU device"):
            cuda_memory_info(invalid)


@pytest.mark.skipif(
    "jax-cuda" not in os.environ.get("TMOL_TEST_BACKENDS", ""),
    reason="requires CUDA test lane",
)
def test_cuda_memory_query_and_failure_restore_device():
    from tmol_jax.score._native import libraries
    from tests.scientific.execution import Execution
    from tests.scientific.molecules import DOMAIN_CASES

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
    molecule = Execution("jax-cuda").molecule(DOMAIN_CASES["ligand"])
    native = molecule.score_function.jitted_value_and_grad(molecule.pose)
    native_reference = tuple(np.asarray(x) for x in native(molecule.pose.coords))
    assert np.linalg.norm(native_reference[1]) > 0
    for invalid in (-1, np.iinfo(np.int32).max):
        free, total = ctypes.c_uint64(17), ctypes.c_uint64(29)
        assert (
            libraries()["cuda"].tmol_cuda_memory_info(
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
