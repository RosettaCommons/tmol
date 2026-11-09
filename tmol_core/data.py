"""Canonical data locations for installed packages and source checkouts."""

from pathlib import Path


def chemical_directory() -> Path:
    packaged = Path(__file__).parent / "default" / "chemical"
    if packaged.is_dir():
        return packaged
    # Source checkouts keep one canonical copy; the core build includes it.
    return Path(__file__).parents[1] / "tmol/database/default/chemical"
