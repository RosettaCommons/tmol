"""Framework-neutral weak source identity for scientific preparation caches."""

from dataclasses import dataclass
import weakref


@dataclass(frozen=True, slots=True, eq=False)
class AnnotationKey:
    """Identify an annotation by live source objects and scalar settings.

    Equal but distinct sources intentionally do not match. Sources are expected
    to be immutable snapshots; invalidate the owning cache after an unsupported
    in-place mutation of data used by an annotation.
    """

    sources: tuple
    settings: tuple = ()

    @classmethod
    def from_sources(cls, *sources, settings=()):
        return cls(
            tuple(weakref.ref(source) for source in sources),
            tuple(settings),
        )

    def matches(self, other):
        # A compiled configuration remains valid after its source expires.
        # Distinct keys still require live, identical source referents.
        if self is other:
            return True
        return (
            isinstance(other, AnnotationKey)
            and self.settings == other.settings
            and len(self.sources) == len(other.sources)
            and all(
                (source := first()) is not None and source is second()
                for first, second in zip(self.sources, other.sources)
            )
        )
