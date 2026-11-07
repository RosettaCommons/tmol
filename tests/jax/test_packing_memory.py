"""The native free-memory query preserves the selected CUDA device."""

import ctypes
import os

import jax
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
    free, total = ctypes.c_uint64(), ctypes.c_uint64()
    assert (
        libraries()["cuda"].tmol_cuda_memory_info(
            -1, ctypes.byref(free), ctypes.byref(total)
        )
        != 0
    )
    assert get_device(ctypes.byref(current)) == 0 and current.value == previous.value
