"""Preparation reuse cannot alias distinct sources or outlive its owner."""

from dataclasses import dataclass
import gc
import weakref

import pytest

from tmol_core.cache import AnnotationKey, PreparationCache


@dataclass
class Source:
    value: int


def test_annotations_follow_source_identity_settings_and_owner_lifetime():
    cache = PreparationCache()
    first, equal = Source(3), Source(3)

    def prepare(source, setting=1):
        return cache.get_or_create(
            "table",
            AnnotationKey.from_sources(source, settings=(setting,)),
            lambda: (source, source.value * setting),
        )

    original = prepare(first)
    assert prepare(first) is original
    assert prepare(equal) is not original
    assert prepare(equal, 4)[1] == 12
    del original
    ref = weakref.ref(equal)
    del equal
    cache.clear()
    gc.collect()
    assert ref() is None


def test_failed_preparation_is_not_published_and_retention_is_bounded():
    cache = PreparationCache()
    source = Source(1)

    def fail():
        raise ValueError("invalid new parameter source")

    with pytest.raises(ValueError, match="invalid new"):
        cache.get_or_create("bad", AnnotationKey.from_sources(source), fail)
    assert (
        cache.get_or_create("bad", AnnotationKey.from_sources(source), lambda: 7) == 7
    )
    refs = []
    for i in range(20):
        value = Source(i)
        refs.append(weakref.ref(value))
        cache.get_or_create(
            str(i), AnnotationKey.from_sources(value), lambda source=value: source
        )
    del value
    gc.collect()
    assert sum(r() is not None for r in refs) == 8
    del cache
    gc.collect()
    assert all(r() is None for r in refs)
