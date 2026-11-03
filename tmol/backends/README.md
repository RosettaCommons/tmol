# Backends

Device-specific entry points live here, one directory per backend, mirroring the
module path they belong to (`tmol/backends/cpu/score/ljlk/ljlk_pose_score.cpu.cpp`
is the CPU instantiation of `tmol/score/ljlk/potentials`).

- `cpu/`, `cuda/`: explicit instantiations of the shared `*.impl.hh` code and the
  `DeviceOperations` implementation for that device.
- Physics, parameters and Python bindings stay beside their module; a backend
  directory holds only instantiation and launch glue, never a second copy of an
  energy.
- New backends (Metal, JAX) add their own sibling directory.
