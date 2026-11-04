#!/usr/bin/env python3
"""Inspect candidate metadata without importing a frontend or extracting code."""

import argparse
from email.parser import BytesParser
from pathlib import Path
import re
import tarfile
from zipfile import ZipFile


def requires_core(path):
    if path.suffix == ".whl":
        with ZipFile(path) as archive:
            name = next(
                n for n in archive.namelist() if n.endswith(".dist-info/METADATA")
            )
            metadata = archive.read(name)
    else:
        with tarfile.open(path) as archive:
            member = next(
                m
                for m in archive.getmembers()
                if m.name.count("/") == 1 and m.name.endswith("/PKG-INFO")
            )
            metadata = archive.extractfile(member).read()
    requirements = BytesParser().parsebytes(metadata).get_all("Requires-Dist", [])
    return any(
        re.match(r"tmol[-_]core\s*(?:[<>=!~;(]|$)", r, re.I) for r in requirements
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path, nargs="+")
    parser.add_argument("--github-output", type=Path, required=True)
    args = parser.parse_args()
    paths = [p for p in args.artifacts if p.is_file()]
    if not paths:
        parser.error("no candidate artifacts found")
    with args.github_output.open("a") as output:
        output.write(f"required={str(any(requires_core(p) for p in paths)).lower()}\n")
