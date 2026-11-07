"""Pose/task buffer slicing for the common memory-bounded packing policy."""

import copy
import ctypes
import os

from tmol_core.pack.streaming import PACKER_TASK_POSE_TENSORS, packing_pose_chunk_size
from .task import SetPackerTask


def cuda_memory_info(device):
    """Query the selected local CUDA device without changing the caller's device."""
    import jax
    from tmol_jax.score._native import libraries

    if not isinstance(device, jax.Device) or device.platform != "gpu":
        raise ValueError("CUDA memory queries require a local JAX GPU device")
    if device not in jax.local_devices(backend="gpu"):
        raise ValueError("CUDA memory queries require a local JAX GPU device")
    library = libraries().get("cuda")
    query = getattr(library, "tmol_cuda_memory_info", None)
    if query is None:
        raise RuntimeError(
            "CUDA packing memory query requires matching tmol-jax-native"
        )
    query.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    query.restype = ctypes.c_int
    free, total = ctypes.c_uint64(), ctypes.c_uint64()
    status = query(device.local_hardware_id, ctypes.byref(free), ctypes.byref(total))
    if status:
        raise RuntimeError(f"CUDA packing memory query failed with status {status}")
    return free.value, total.value


def max_poses_per_chunk(pose):
    configured = os.environ.get("TMOL_PACK_MAX_POSES_PER_CHUNK")
    free = None
    if configured is None and pose.device.platform == "gpu" and pose.n_poses > 1:
        free, _ = cuda_memory_info(pose.device)
    return packing_pose_chunk_size(pose.max_n_blocks, free, configured)


def slice_task(task, pose, first, last):
    """Keep task choices and sampler identities while rebinding its pose rows."""
    result = copy.copy(task)
    for name in PACKER_TASK_POSE_TENSORS:
        setattr(result, name, getattr(task, name)[first:last])
    result.pose_stack = pose
    result._coordinate_shape = pose.coords.shape
    result._topology = (
        pose.block_type_ind,
        pose.block_coord_offset,
        pose.inter_residue_connections,
    )
    return SetPackerTask.from_packer_task(result)
