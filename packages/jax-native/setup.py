"""Build thin XLA bindings around the canonical TMol native kernels."""

import os
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
        flags = ["-std=c++17", "-O3", "-shared"]
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
        subprocess.run(
            [
                *compiler,
                *flags,
                f"-I{jax.ffi.include_dir()}",
                "-Iinclude",
                "-Iinclude/tmol/extern",
                *extension.sources,
                "-o",
                str(output),
            ],
            check=True,
        )


extensions = [Extension("tmol_jax_native._tmol_cpu", ["genbonded_cpu.cpp"])]
if os.environ.get("TMOL_JAX_NATIVE_CUDA", "0") == "1":
    extensions.append(Extension("tmol_jax_native._tmol_cuda", ["genbonded_cuda.cu"]))

setup(
    ext_modules=extensions,
    cmdclass={"build_ext": NativeBuildExt},
    install_requires=[
        "jax>=0.11.2",
        f"tmol-jax=={Path('VERSION').read_text().strip()}",
    ],
)
