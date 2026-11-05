#!/usr/bin/env python3
"""Build CPU and optional CUDA XLA bindings from the shared scientific source."""

import argparse
import hashlib
import json
import re
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib


def stage_native_headers(root, stage):
    # Include canonical source headers, never a second scientific implementation.
    for relative in (
        "score/common",
        "score/genbonded/potentials",
        "score/cartbonded/potentials",
        "score/backbone_torsion/potentials",
        "score/metal/potentials",
        "score/disulfide/potentials",
        "score/na_torsion/potentials",
        "score/dunbrack/potentials",
        "score/ljlk/potentials",
        "score/elec/potentials",
        "score/lk_ball/potentials",
        "kinematics/compiled",
        "utility/portable",
        "utility/tensor",
        "numeric",
        "extern/moderngpu",
        "extern/Eigen",
    ):
        shutil.copytree(
            root / "tmol" / relative,
            stage / "include/tmol" / relative,
            ignore=shutil.ignore_patterns(
                "*.py", "*.pyc", "__pycache__", "*.cpp", "*.cu"
            ),
        )
    for relative in (
        "hbond/identification.hh",
        "bonded_atom.hh",
        "unresolved_atom_types.hh",
    ):
        destination = stage / "include/tmol/score" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / "tmol/score" / relative, destination)
    shutil.copyfile(
        root / "tmol/utility/nvtx.hh", stage / "include/tmol/utility/nvtx.hh"
    )


def stage_native_licenses(root, stage):
    licenses = stage / "licenses"
    licenses.mkdir()
    eigen_licenses = licenses / "Eigen"
    eigen_licenses.mkdir()
    for path in (root / "tmol/extern/Eigen").glob("COPYING.*"):
        shutil.copyfile(path, eigen_licenses / path.name)
    shutil.copyfile(
        root / "tmol/extern/Eigen/COPYING.APACHE", licenses / "APACHE-2.0.txt"
    )
    shutil.copyfile(
        root / "tmol/extern/moderngpu/LICENSE.md", licenses / "moderngpu-LICENSE.md"
    )


def stage_native(root, stage, source_revision=None):
    package = root / "packages/jax-native"
    for name in ("pyproject.toml", "setup.py", "NOTICE"):
        shutil.copyfile(package / name, stage / name)
    shutil.copyfile(root / "LICENSE", stage / "LICENSE")
    shutil.copytree(package / "tmol_jax_native", stage / "tmol_jax_native")
    backend = root / "tmol/backends/jax"
    for path in (
        backend / "genbonded_cpu.cpp",
        backend / "genbonded_cuda.cu",
        *backend.glob("*_ffi.hh"),
    ):
        shutil.copyfile(path, stage / path.name)
    stage_native_headers(root, stage)
    source_files = {
        str(path.relative_to(stage)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(stage.rglob("*"))
        if path.is_file()
    }
    if source_revision is not None and not re.fullmatch(
        r"[0-9a-f]{40}", source_revision
    ):
        raise ValueError("--source-revision must be the verified full Git commit SHA")
    probe = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    top = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if (
        probe.returncode == 0
        and top.returncode == 0
        and Path(top.stdout.strip()).resolve() == root.resolve()
    ):
        revision = probe.stdout.strip()
        if source_revision is not None and source_revision != revision:
            raise ValueError("--source-revision disagrees with the checkout")
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"], text=True
            ).strip()
        )
    else:
        if source_revision is None:
            raise ValueError(
                "Building a source archive requires --source-revision with its verified commit SHA"
            )
        revision, dirty = source_revision, None
    (stage / "tmol_jax_native/source.json").write_text(
        json.dumps(
            {
                "revision": revision,
                "dirty": dirty,
                "sha256": source_files,
            },
            indent=2,
        )
        + "\n"
    )
    stage_native_licenses(root, stage)
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    (stage / "VERSION").write_text(project["version"] + "\n")
    (stage / "MANIFEST.in").write_text(
        "include VERSION *.hh *.cu *.cpp\nrecursive-include include *\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("dist/jax-native"))
    parser.add_argument(
        "--cuda", action="store_true", help="Include CUDA as well as CPU"
    )
    parser.add_argument("--cuda-architectures", help="Semicolon-separated SM values")
    parser.add_argument(
        "--source-revision",
        help="Verified full Git SHA for an immutable source archive",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="tmol-jax-native-") as temporary:
        stage = Path(temporary)
        stage_native(root, stage, args.source_revision)
        environment = os.environ.copy()
        environment["TMOL_JAX_NATIVE_CUDA"] = "1" if args.cuda else "0"
        if args.cuda_architectures:
            environment["CMAKE_CUDA_ARCHITECTURES"] = args.cuda_architectures
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--no-isolation",
                "--outdir",
                str(args.outdir.resolve()),
                str(stage),
            ],
            env=environment,
            check=True,
        )


if __name__ == "__main__":
    main()
