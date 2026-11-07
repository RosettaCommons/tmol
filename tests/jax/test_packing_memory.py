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

    library = libraries()["cuda"]
    get_device = library.cudaGetDevice
    get_device.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_device.restype = ctypes.c_int
    previous = ctypes.c_int()
    assert get_device(ctypes.byref(previous)) == 0
    for device in jax.local_devices(backend="gpu"):
        free, total = cuda_memory_info(device)
        assert 0 < free <= total
    current = ctypes.c_int()
    assert get_device(ctypes.byref(current)) == 0 and current.value == previous.value
    free, total = ctypes.c_uint64(), ctypes.c_uint64()
    assert (
        library.tmol_cuda_memory_info(-1, ctypes.byref(free), ctypes.byref(total)) != 0
    )
    assert get_device(ctypes.byref(current)) == 0 and current.value == previous.value
