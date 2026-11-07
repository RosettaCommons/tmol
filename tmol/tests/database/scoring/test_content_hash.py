import torch

from tmol.database.scoring._content_hash import _content_hash


def test_grid_metadata_boundaries_are_unambiguous():
    assert _content_hash((1.0, 23.0)) != _content_hash((1.02, 3.0))
    assert _content_hash((1, (2, 3))) != _content_hash(((1, 2), 3))


def test_tensor_values_shape_and_dtype_contribute():
    original = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert _content_hash(original) == _content_hash(original.clone())
    assert _content_hash(original) == _content_hash(original.t().contiguous().t())
    assert _content_hash(original) != _content_hash(original + 1)
    assert _content_hash(original) != _content_hash(original.reshape(4, 3))
    assert _content_hash(original) != _content_hash(original.double())
    assert _content_hash(torch.empty((0, 3))) != _content_hash(torch.empty((0, 4)))


def test_host_and_tensor_parameter_identity_agree():
    from tmol_core.database.content_hash import _content_hash as host_hash

    original = torch.arange(24, dtype=torch.float64).reshape(4, 6)[:, ::2]
    assert _content_hash(("table", original)) == host_hash(("table", original.numpy()))
    changed = original.numpy().copy()
    changed[2, 1] += 1
    assert _content_hash(("table", original)) != host_hash(("table", changed))
