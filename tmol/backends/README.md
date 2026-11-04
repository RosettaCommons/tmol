# Native execution backends

`cpu/`, `cuda/` and `metal/` contain device instantiations, launch glue and
Metal shaders, mirroring their module paths. Shared physics, parameters and
portable types remain under `tmol/score`, `tmol/numeric` and `tmol/utility`.

`jax/` contains CUDA traversal and XLA FFI adapters for the optional JAX native
wheel. It denotes a framework binding, while the other directories denote
devices. These adapters use the same canonical scalar potentials; their buffer
layouts and gradient accumulation remain specific to JAX. The Torch-free Python
frontend lives in `tmol_jax/`. `scripts/build_jax_cuda.py` stages the native
adapter sources with metadata from `packages/jax-cuda`.
