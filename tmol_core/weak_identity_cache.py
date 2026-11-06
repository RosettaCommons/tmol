"""Bounded weak-owner identity caches shared by scientific host preparation."""

from collections import OrderedDict
import threading
import weakref


class WeakIdentityLRU:
    """Cache by owner identity and configuration without retaining the owner.

    Values must not retain their owner. Owners must be weak-referenceable and
    remain unchanged while cached; publish a new owner after changing its data.
    Factories run outside the lock, so simultaneous misses may compute twice.
    Framework adapters may attach readiness records to newly published values.
    """

    @staticmethod
    def _record_readiness(value):
        return ()

    @staticmethod
    def _wait_readiness(readiness):
        pass

    def __init__(self, capacity=32):
        if capacity < 1:
            raise ValueError("cache capacity must be positive")
        self.capacity = capacity
        self._entries = OrderedDict()
        self._lock = threading.RLock()

    def __len__(self):
        with self._lock:
            return len(self._entries)

    def get_or_create(self, owner, configuration, factory):
        return self.get_or_create_many((owner,), configuration, factory)

    def get_or_create_many(self, owners, configuration, factory):
        """Cache a value derived from several independently owned databases."""
        owners = tuple(owners)
        if not owners:
            raise ValueError("at least one cache owner is required")
        key = (tuple(id(owner) for owner in owners), configuration)

        def matches(entry):
            return entry is not None and all(
                ref() is owner for ref, owner in zip(entry[0], owners)
            )

        with self._lock:
            entry = self._entries.get(key)
            if matches(entry):
                self._entries.move_to_end(key)
                value = entry[1]
                readiness = entry[2] if len(entry) > 2 else ()
                found = True
            else:
                found = False
        if found:
            self._wait_readiness(readiness)
            return value

        value = factory()
        readiness = self._record_readiness(value)

        def discard(ref):
            with self._lock:
                current = self._entries.get(key)
                if current is not None and any(stored is ref for stored in current[0]):
                    del self._entries[key]

        with self._lock:
            # Another caller may have completed the same miss meanwhile.
            current = self._entries.get(key)
            if matches(current):
                self._entries.move_to_end(key)
                value = current[1]
                readiness = current[2] if len(current) > 2 else ()
            else:
                self._entries[key] = (
                    tuple(weakref.ref(owner, discard) for owner in owners),
                    value,
                    readiness,
                )
                self._entries.move_to_end(key)
                while len(self._entries) > self.capacity:
                    self._entries.popitem(last=False)
        self._wait_readiness(readiness)
        return value
