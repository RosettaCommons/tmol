// Array contracts shared by CPU and CUDA annealer adapters.
#pragma once
#include "genbonded_ffi.hh"

namespace tmol::jax {
inline void validate_annealer_graph(const std::vector<ffi::AnyBuffer>& a) {
  require_shape(a[0], ffi::DataType::S32, {-1});
  auto poses = a[0].dimensions()[0];
  for (int i = 1; i < 3; ++i) require_shape(a[i], ffi::DataType::S32, {poses});
  require_shape(a[3], ffi::DataType::S32, {poses, -1});
  auto residues = a[3].dimensions()[1];
  require_shape(a[4], ffi::DataType::S32, {poses, residues});
  require_shape(a[5], ffi::DataType::S32, {-1});
  auto rotamers = a[5].dimensions()[0];
  // Opaque int64 metadata retains CSR capacity with JAX x64 disabled.
  require_shape(a[6], ffi::DataType::U8, {poses * residues + 1, 8});
  require_shape(a[7], ffi::DataType::S32, {-1});
  require_shape(a[8], ffi::DataType::U8, {a[7].dimensions()[0], 8});
  require_shape(a[9], ffi::DataType::U8, {-1, 8});
  require_shape(a[10], ffi::DataType::F32, {rotamers});
  require_shape(a[11], ffi::DataType::F32, {-1});
  for (int i : {6, 8, 9})
    if (reinterpret_cast<uintptr_t>(a[i].untyped_data()) % alignof(int64_t))
      throw std::invalid_argument("unaligned annealer CSR metadata");
}
}  // namespace tmol::jax
