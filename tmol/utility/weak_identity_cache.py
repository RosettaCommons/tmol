"""Bounded derived-data caches for immutable, possibly unhashable owners."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass

import torch

from tmol_core.weak_identity_cache import WeakIdentityLRU as _WeakIdentityLRU


def _cuda_devices(value):
    """Find CUDA tensor devices nested in supported cache value containers."""
    devices = set()
    pending = [value]
    seen = set()
    while pending:
        item = pending.pop()
        item_id = id(item)
        if item_id in seen:
            continue
        seen.add(item_id)

        if isinstance(item, torch.Tensor):
            if item.device.type == "cuda":
                devices.add(item.device)
        elif isinstance(item, Mapping):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (tuple, list, set, frozenset)):
            pending.extend(item)
        elif not isinstance(item, type) and is_dataclass(item):
            pending.extend(getattr(item, field.name) for field in fields(item))
        else:
            attrs_fields = getattr(type(item), "__attrs_attrs__", ())
            pending.extend(getattr(item, field.name) for field in attrs_fields)
    return devices


def _record_cuda_readiness(value):
    """Record producer-stream readiness for every CUDA device in a value."""
    readiness = []
    for device in sorted(_cuda_devices(value), key=lambda item: item.index):
        with torch.cuda.device(device):
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device))
        readiness.append((device, event))
    return tuple(readiness)


def _wait_for_cuda_readiness(readiness):
    """Make each device's caller-current stream wait for cached production."""
    for device, event in readiness:
        torch.cuda.current_stream(device).wait_event(event)


class WeakIdentityLRU(_WeakIdentityLRU):
    """Shared owner-identity cache with Torch CUDA stream readiness."""

    @staticmethod
    def _record_readiness(value):
        return _record_cuda_readiness(value)

    @staticmethod
    def _wait_readiness(readiness):
        _wait_for_cuda_readiness(readiness)
