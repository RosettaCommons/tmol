#!/usr/bin/env python3
"""Build CPU and optional CUDA XLA bindings from the shared scientific source."""

import argparse
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib


def stage_native(root, stage):
    package = root / "packages/jax-native"
    for name in ("pyproject.toml", "setup.py", "NOTICE"):
        shutil.copyfile(package / name, stage / name)
    shutil.copyfile(root / "LICENSE", stage / "LICENSE")
    shutil.copytree(package / "tmol_jax_native", stage / "tmol_jax_native")
    backend = root / "tmol/backends/jax"
    for name in (
        "genbonded_cpu.cpp",
        "genbonded_cuda.cu",
        "genbonded_ffi.hh",
        "cartbonded_ffi.hh",
    ):
        shutil.copyfile(backend / name, stage / name)
    # Include canonical source headers, never a second scientific implementation.
    for relative in (
        "score/common",
        "score/genbonded/potentials",
        "score/cartbonded/potentials",
        "utility/portable",
        "utility/tensor",
        "numeric",
        "extern/moderngpu",
    ):
        shutil.copytree(
            root / "tmol" / relative,
            stage / "include/tmol" / relative,
            ignore=shutil.ignore_patterns(
                "*.py", "*.pyc", "__pycache__", "*.cpp", "*.cu"
            ),
        )
    licenses = stage / "licenses"
    licenses.mkdir()
    shutil.copyfile(
        root / "tmol/extern/Eigen/COPYING.APACHE", licenses / "APACHE-2.0.txt"
    )
    shutil.copyfile(
        root / "tmol/extern/moderngpu/LICENSE.md", licenses / "moderngpu-LICENSE.md"
    )
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
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="tmol-jax-native-") as temporary:
        stage = Path(temporary)
        stage_native(root, stage)
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
