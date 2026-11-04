#!/usr/bin/env python3
"""Build standalone, framework-free tmol-core artifacts from this checkout."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib


def build_core(outdir, *, no_isolation=False):
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
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                *(["--no-isolation"] if no_isolation else []),
                "--outdir",
                str(Path(outdir).resolve()),
                str(stage),
            ],
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("dist"))
    parser.add_argument("--no-isolation", action="store_true")
    args = parser.parse_args()
    build_core(args.outdir, no_isolation=args.no_isolation)
