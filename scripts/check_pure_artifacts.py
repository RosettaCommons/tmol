#!/usr/bin/env python3
"""Validate core/JAX wheel and sdist ownership, dependencies and required data."""

import argparse
from email.parser import BytesParser
from pathlib import Path
import tarfile
from zipfile import ZipFile

from packaging.requirements import Requirement
from packaging.utils import parse_sdist_filename, parse_wheel_filename


def check_pure(wheel, sdist):
    name, version, _, _ = parse_wheel_filename(wheel.name)
    source_name, source_version = parse_sdist_filename(sdist.name)
    assert name == source_name and name in {"tmol-core", "tmol-jax"}
    assert version == source_version
    core = name == "tmol-core"
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata = BytesParser().parsebytes(
            archive.read(next(n for n in names if n.endswith(".dist-info/METADATA")))
        )
        requirements = [Requirement(r) for r in metadata.get_all("Requires-Dist", [])]
        forbidden = (
            {"torch", "tmol", "jax", "jaxlib", "tmol-jax"}
            if core
            else {"torch", "tmol"}
        )
        assert not forbidden & {
            r.name if core else r.name.lower() for r in requirements
        }
        if core:
            packages = {
                n.split("/")[0] for n in names if "/" in n and ".dist-info/" not in n
            }
            assert packages == {"tmol_core"}, packages
            required = (
                "tmol_core/default/chemical/chemical.yaml",
                "tmol_core/default/chemical/metals.yaml",
                "tmol_core/extern/pysmiles/LICENSE.md",
                "tmol_core/NOTICE",
            )
            assert any(n.endswith("/licenses/LICENSE") for n in names)
        else:
            core_requirements = [r for r in requirements if r.name == "tmol-core"]
            # Exactly one plain dependency; extras must retain the same version pin.
            assert len([r for r in core_requirements if r.marker is None]) == 1
            assert all(str(r.specifier) == f"=={version}" for r in core_requirements)
            assert not any(n.startswith(("tmol/", "tmol_core/")) for n in names)
            required = (
                "database/default/scoring/dunbrack.npz",
                "database/default/chemical/chemical.yaml",
            )
        for suffix in required:
            assert (
                suffix in names if core else any(n.endswith(suffix) for n in names)
            ), suffix
    with tarfile.open(sdist, "r:*" if core else "r:gz") as archive:
        names = archive.getnames()
        required = (
            (
                "/pyproject.toml",
                "/VERSION",
                "/tmol_core/chemical/patching.py",
                "/tmol_core/extern/pysmiles/LICENSE.md",
                "/tmol_core/default/chemical/chemical.yaml",
            )
            if core
            else (
                "/pyproject.toml",
                "/tmol_jax/database/default/scoring/dunbrack.npz",
            )
        )
        for suffix in required:
            assert any(n.endswith(suffix) for n in names), suffix
        forbidden = ("/tmol/", "/tmol_jax/") if core else ("/tmol/",)
        assert not any(prefix in n for n in names for prefix in forbidden)
    print(
        f"validated {'framework-free' if core else 'Torch-free'} {name} {version} wheel and sdist"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("sdist", type=Path)
    args = parser.parse_args()
    check_pure(args.wheel, args.sdist)
