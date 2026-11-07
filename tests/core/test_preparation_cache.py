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


@pytest.mark.parametrize("capacity", (None, 1, 13))
def test_failed_preparation_is_not_published_and_retention_is_bounded(capacity):
    cache = PreparationCache() if capacity is None else PreparationCache(capacity)
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
    bound = 8 if capacity is None else capacity
    assert [i for i, ref in enumerate(refs) if ref() is not None] == list(
        range(20 - bound, 20)
    )
    del cache
    gc.collect()
    assert all(r() is None for r in refs)


def test_capacity_preserves_hits_and_replaces_only_one_namespace():
    cache = PreparationCache(max_entries=13)
    source = Source(3)
    key = AnnotationKey.from_sources(source)
    values = [object() for _ in range(13)]
    for i, value in enumerate(values):
        assert cache.get_or_create(str(i), key, lambda: value) is value

    def unexpected_prepare():
        pytest.fail("A retained annotation was unnecessarily prepared again")

    for i, value in enumerate(values):
        assert cache.get_or_create(str(i), key, unexpected_prepare) is value
    replacement = object()
    changed = AnnotationKey.from_sources(source, settings=("new",))
    assert cache.get_or_create("4", changed, lambda: replacement) is replacement
    assert cache.get_or_create("0", key, unexpected_prepare) is values[0]
    cache.get_or_create("extra", key, object)
    assert cache.get_or_create("1", key, unexpected_prepare) is values[1]
    assert cache.get_or_create("4", changed, unexpected_prepare) is replacement
    # Hits and replacements preserve the original insertion eviction order.
    assert cache.get_or_create("0", key, object) is not values[0]


@pytest.mark.parametrize("capacity", (True, False, None, "8", 1.5, float("inf")))
def test_capacity_rejects_nonintegers(capacity):
    with pytest.raises(TypeError, match="positive integer"):
        PreparationCache(max_entries=capacity)


@pytest.mark.parametrize("capacity", (0, -1))
def test_capacity_rejects_nonpositive_integers(capacity):
    with pytest.raises(ValueError, match="positive integer"):
        PreparationCache(max_entries=capacity)
