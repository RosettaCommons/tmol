"""Content-derived identity for parameter databases.

A database's id is used as a cache key, so it must change whenever the contents
change; otherwise a cache hands one database's parameters to another.
"""

import hashlib

import attr
import numpy as np


def _content_hash(*parts, array_view=None) -> str:
    """A sha256 over attrs records, arrays and scalars, in the order given."""
    digest = hashlib.sha256()
    for part in parts:
        _update(digest, part, array_view)
    return digest.hexdigest()


def _update(digest, value, array_view=None) -> None:
    if array_view is not None:
        converted = array_view(value)
        if converted is not None:
            value = converted
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("content hashing does not describe object arrays")
        digest.update(b"array:")
        _update(digest, (tuple(value.shape), value.dtype.str), array_view)
        array = np.ascontiguousarray(value).reshape(-1)
        _write(digest, memoryview(array).cast("B"))
    elif attr.has(type(value)):
        digest.update(b"attrs:")
        _write(digest, type(value).__qualname__.encode())
        for field in attr.fields(type(value)):
            _write(digest, field.name.encode())
            _update(digest, getattr(value, field.name), array_view)
    elif isinstance(value, (tuple, list)):
        digest.update(b"sequence:")
        digest.update(len(value).to_bytes(8, "big"))
        for item in value:
            _update(digest, item, array_view)
    elif value is None or isinstance(value, (str, bytes, bool, int, float)):
        digest.update(b"scalar:")
        _write(digest, repr(value).encode())
    else:
        # repr is not a safe stand-in for contents: numpy truncates past 1000
        # elements, so two large arrays differing in the middle hash alike, and
        # a set reprs in a different order each interpreter run. A collision
        # here hands one database's parameters to another, which is the single
        # thing this identity exists to prevent.
        raise TypeError(
            f"content hashing does not describe {type(value).__name__}; add a "
            "case for it rather than letting repr stand in for its contents"
        )


def _write(digest, data):
    """Frame each byte string so distinct adjacent values cannot alias."""
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)
