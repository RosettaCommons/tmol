#!/usr/bin/env python3
"""Build standalone, framework-free tmol-core artifacts from this checkout."""

import argparse
from pathlib import Path
import shutil
import tempfile
import tomllib

from build_source import build_stage, stamp_package


def build_core(outdir, *, no_isolation=False, source_revision=None):
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="tmol-core-build-") as temporary:
        stage = Path(temporary)
        shutil.copytree(
            root / "tmol_core",
            stage / "tmol_core",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "default"),
        )
        shutil.copytree(
            root / "tmol/database/default/chemical",
            stage / "tmol_core/default/chemical",
        )
        for source, target in (
            ("packages/core/pyproject.toml", "pyproject.toml"),
            ("LICENSE", "LICENSE"),
            ("docs/shared_core.md", "README.md"),
        ):
            shutil.copyfile(root / source, stage / target)
        metadata = tomllib.loads((root / "pyproject.toml").read_text())
        (stage / "VERSION").write_text(metadata["project"]["version"] + "\n")
        stamp_package(root, stage, "tmol_core", source_revision)
        build_stage(stage, outdir, no_isolation=no_isolation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("dist"))
    parser.add_argument("--no-isolation", action="store_true")
    parser.add_argument("--source-revision")
    args = parser.parse_args()
    build_core(
        args.outdir,
        no_isolation=args.no_isolation,
        source_revision=args.source_revision,
    )
