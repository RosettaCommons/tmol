"""Locations of optional framework-free native execution libraries."""

from pathlib import Path

ABI_VERSION = 1


def library_path(platform="cpu") -> Path | None:
    """Find the requested platform library; absent CUDA is an explicit capability."""
    platform = platform.lower()
    if platform not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported TMol native platform {platform!r}")
    libraries = tuple(Path(__file__).parent.glob(f"_tmol_{platform}*.so"))
    if not libraries:
        return None
    if len(libraries) != 1:
        raise RuntimeError(
            f"tmol-jax-native does not contain exactly one {platform.upper()} library"
        )
    return libraries[0]


__all__ = ["ABI_VERSION", "library_path"]
