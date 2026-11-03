# Backends

Device-specific entry points live here, one directory per backend, mirroring the
module path they belong to (`tmol/backends/cpu/score/ljlk/ljlk_pose_score.cpu.cpp`
is the CPU instantiation of `tmol/score/ljlk/potentials`).

- `cpu/`, `cuda/`: explicit instantiations of the shared `*.impl.hh` code and the
  `DeviceOperations` implementation for that device.
- Physics, parameters and Python bindings stay beside their module; a backend
  directory holds only instantiation and launch glue, never a second copy of an
  energy.
- `jax/`: CUDA launch code and JAX FFI entry points for the optional
  `tmol-jax-cuda` wheel. The adapters call the shared scalar potentials under
  `tmol/score`; their pair traversal, buffer layouts and gradient accumulation
  remain specific to JAX. `scripts/build_jax_cuda.py` stages these sources with
  package metadata from `packages/jax-cuda` for the Torch-free build.
- The separately installable Python frontend stays in `tmol_jax/` so importing
  it does not initialize the Torch package. New device backends add sibling
  directories here.
