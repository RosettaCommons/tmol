# Native execution backends

`cpu/`, `cuda/` and `metal/` contain device instantiations, launch glue and
Metal shaders, mirroring their module paths. Shared physics, parameters and
portable types remain under `tmol/score`, `tmol/numeric` and `tmol/utility`.

`jax/` contains CPU/CUDA execution policies and XLA FFI adapters for the
optional `tmol-jax-native` wheel. Both frontends call the common native traversal,
physics and analytic derivative kernels. Framework bindings own registration,
stream selection and buffers. The Torch-free Python frontend lives in
`tmol_jax/`; `scripts/build_jax_native.py` stages the shared headers and thin
bindings using metadata from `packages/jax-native`.
