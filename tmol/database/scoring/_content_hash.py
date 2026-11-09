"""Torch storage adapter for shared scientific parameter identity."""

import torch
from tmol_core.database.content_hash import _content_hash as _host_content_hash


def _tensor_array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return None


def _content_hash(*parts):
    return _host_content_hash(*parts, array_view=_tensor_array)
