"""Build thin XLA bindings around the canonical TMol native kernels."""

import os
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

import jax
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class NativeBuildExt(build_ext):
    def build_extension(self, extension):
        cuda = extension.name.endswith("_cuda")
        compiler = ["nvcc"] if cuda else shlex.split(os.environ.get("CXX", "c++"))
        if not compiler or shutil.which(compiler[0]) is None:
            raise RuntimeError(f"Native build compiler unavailable: {compiler}")
        output = Path(self.get_ext_fullpath(extension.name)).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        flags = ["-std=c++17", "-O3"]
        if cuda:
            flags += [
                "--expt-relaxed-constexpr",
                "--extended-lambda",
                "-Xcompiler",
                "-fPIC",
            ]
            for arch in os.environ.get("CMAKE_CUDA_ARCHITECTURES", "native").split(";"):
                if arch == "native":
                    flags.append("-arch=native")
                elif arch:
                    flags.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")
        else:
            flags += ["-fPIC", "-pthread"]
        includes = [
            f"-I{jax.ffi.include_dir()}",
            "-Iinclude",
            "-Iinclude/tmol/extern",
        ]
        commands = []
        sources = extension.sources
        if cuda:
            objects = []
            temporary = Path(self.build_temp).resolve()
            temporary.mkdir(parents=True, exist_ok=True)
            for source in sources:
                obj = temporary / (Path(source).stem + ".o")
                # Torch applies this policy to LJ/LK and electrostatic kernels.
                # Never apply it to torsion, bonded or kinematic execution.
                math_flags = (
                    ["--use_fast_math"] if source == "nonbonded_cuda.cu" else []
                )
                commands.append(
                    [
                        *compiler,
                        *flags,
                        *math_flags,
                        *includes,
                        "-c",
                        source,
                        "-o",
                        str(obj),
                    ]
                )
                objects.append(str(obj))
            command = [*compiler, "-shared", *objects, "-o", str(output)]
        else:
            command = [
                *compiler,
                *flags,
                "-shared",
                *includes,
                *sources,
                "-o",
                str(output),
            ]
        commands.append(command)
        for invocation in commands:
            subprocess.run(invocation, check=True)
        manifest = {
            "compiler": subprocess.check_output(
                [*compiler, "--version"], text=True
            ).strip(),
            "command": command,
            "commands": commands,
            "jax_version": jax.__version__,
            "library_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
        (output.parent / ("build-cuda.json" if cuda else "build-cpu.json")).write_text(
            json.dumps(manifest, indent=2) + "\n"
        )


extensions = [
    Extension("tmol_jax_native._tmol_cpu", ["genbonded_cpu.cpp", "refinement.cpu.cpp"])
]
if os.environ.get("TMOL_JAX_NATIVE_CUDA", "0") == "1":
    extensions.append(
        Extension(
            "tmol_jax_native._tmol_cuda", ["genbonded_cuda.cu", "nonbonded_cuda.cu"]
        )
    )

setup(
    ext_modules=extensions,
    cmdclass={"build_ext": NativeBuildExt},
    install_requires=[
        "jax>=0.11.2",
        f"tmol-jax=={Path('VERSION').read_text().strip()}",
    ],
)
