// Match the mature Torch CUDA LJ/LK/electrostatic compilation policy.
// Other scientific kernels remain in the exact-math translation unit.
#include "cuda_execution_ffi.hh"

#define TMOL_JAX_ONLY_NONBONDED
#include "handlers_ffi.hh"
