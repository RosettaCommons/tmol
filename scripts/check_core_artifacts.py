#!/usr/bin/env python3
"""Check core/JAX namespace ownership, dependencies, data and license inclusion."""

import argparse
from email.parser import BytesParser
from pathlib import Path
import tarfile
from zipfile import ZipFile

from packaging.requirements import Requirement
from packaging.utils import parse_sdist_filename, parse_wheel_filename


def check_core(wheel, sdist):
    name, version, _, _ = parse_wheel_filename(wheel.name)
    source_name, source_version = parse_sdist_filename(sdist.name)
    assert name == source_name == "tmol-core"
    assert version == source_version
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata = BytesParser().parsebytes(
            archive.read(next(n for n in names if n.endswith(".dist-info/METADATA")))
        )
        requirements = {
            Requirement(r).name for r in metadata.get_all("Requires-Dist", [])
        }
        assert not requirements & {"torch", "tmol", "jax", "jaxlib", "tmol-jax"}
        packages = {
            n.split("/")[0] for n in names if "/" in n and ".dist-info/" not in n
        }
        assert packages == {"tmol_core"}, packages
        for suffix in (
            "tmol_core/default/chemical/chemical.yaml",
            "tmol_core/default/chemical/metals.yaml",
            "tmol_core/extern/pysmiles/LICENSE.md",
            "tmol_core/NOTICE",
        ):
            assert suffix in names, suffix
        assert any(n.endswith("/licenses/LICENSE") for n in names)
    with tarfile.open(sdist) as archive:
        names = archive.getnames()
        for suffix in (
            "/pyproject.toml",
            "/VERSION",
            "/tmol_core/chemical/patching.py",
            "/tmol_core/extern/pysmiles/LICENSE.md",
            "/tmol_core/default/chemical/chemical.yaml",
        ):
            assert any(n.endswith(suffix) for n in names), suffix
        assert not any("/tmol/" in n or "/tmol_jax/" in n for n in names)
    print(f"validated framework-free tmol-core {version} wheel and sdist")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("sdist", type=Path)
    args = parser.parse_args()
    check_core(args.wheel, args.sdist)
