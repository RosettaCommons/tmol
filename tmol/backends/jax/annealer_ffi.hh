// Caller-owned XLA buffers for the shared native CPU annealing schedule.
#pragma once
#include "genbonded_ffi.hh"
#include <tmol/pack/compiled/annealer_cpu.hh>
#include <tmol/score/common/tview_binder.hh>
#include <cstring>

namespace tmol::jax {
inline ffi::Error anneal_cpu(
    ffi::RemainingArgs inputs,
    ffi::RemainingRets outputs,
    int64_t chunk_size,
    int64_t seed) {
  try {
    auto a = native_arguments(inputs);
    if (a.size() != 12 || outputs.size() != 4 || chunk_size < 1
        || chunk_size > INT32_MAX)
      throw std::invalid_argument("invalid CPU annealer arguments");
    require_shape(a[0], ffi::DataType::S32, {-1});
    auto poses = a[0].dimensions()[0];
    for (int i = 1; i < 3; ++i)
      require_shape(a[i], ffi::DataType::S32, {poses});
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
    std::vector<ffi::AnyBuffer> out;
    for (size_t i = 0; i < outputs.size(); ++i) {
      auto result = outputs.get<ffi::AnyBuffer>(i);
      if (!result)
        throw std::invalid_argument("annealer expected result buffers");
      out.push_back(**result);
    }
    require_shape(out[0], ffi::DataType::F32, {poses, 1});
    for (int i = 1; i <= 2; ++i)
      require_shape(out[i], ffi::DataType::S32, {poses, 1, residues});
    require_shape(out[3], ffi::DataType::S32, {rotamers});
    for (auto result : out)
      if (result.size_bytes())
        std::memset(result.untyped_data(), 0, result.size_bytes());
    constexpr Device D = Device::CPU;
    pack::compiled::CpuAnnealerContext<
        score::common::TViewBinder<D>::template View>
        c;
    c.pose_n_res = buffer_view<int32_t, 1, D>(a[0]);
    c.n_rotamers_for_pose = buffer_view<int32_t, 1, D>(a[1]);
    c.rotamer_offset_for_pose = buffer_view<int32_t, 1, D>(a[2]);
    c.n_rotamers_for_res = buffer_view<int32_t, 2, D>(a[3]);
    c.oneb_offsets = buffer_view<int32_t, 2, D>(a[4]);
    c.res_for_rot = buffer_view<int32_t, 1, D>(a[5]);
    c.neighbor_row_offsets = buffer_view<int64_t, 1, D>(a[6]);
    c.neighbor_blocks = buffer_view<int32_t, 1, D>(a[7]);
    c.neighbor_chunk_offset_offsets = buffer_view<int64_t, 1, D>(a[8]);
    c.chunk_offsets = buffer_view<int64_t, 1, D>(a[9]);
    c.energy1b = buffer_view<float, 1, D>(a[10]);
    c.energy2b = buffer_view<float, 1, D>(a[11]);
    c.scores = buffer_view<float, 2, D>(out[0]);
    c.best_rotamer_assignments = buffer_view<int32_t, 3, D>(out[1]);
    c.current_rotamer_assignments = buffer_view<int32_t, 3, D>(out[2]);
    c.quench_order = buffer_view<int32_t, 1, D>(out[3]);
    c.chunk_size = static_cast<int>(chunk_size);
    pack::compiled::run_cpu_annealer(c, static_cast<uint64_t>(seed));
    return ffi::Error::Success();
  } catch (const std::exception& error) {
    return ffi::Error::InvalidArgument(error.what());
  }
}
}  // namespace tmol::jax
