"""Locations of optional framework-free native execution libraries."""

import json
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


def build_info():
    """Exact staged sources, compiler settings and native artifact identities."""
    root = Path(__file__).parent
    return {
        "source": json.loads((root / "source.json").read_text()),
        "builds": {
            path.stem.removeprefix("build-"): json.loads(path.read_text())
            for path in sorted(root.glob("build-*.json"))
        },
    }


__all__ = ["ABI_VERSION", "library_path", "build_info"]
