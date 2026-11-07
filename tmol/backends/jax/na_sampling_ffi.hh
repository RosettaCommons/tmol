// Allocation-free execution of the common nucleic-acid sampler stages.
#pragma once
#include "graph_topology_ffi.hh"
#include <tmol/pack/rotamer/na_sampling.hh>

namespace tmol::jax {
struct NaSamplingStage {
  static int fill(int64_t, size_t) { return 0; }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t,
      int64_t bases,
      double temperature,
      int64_t syn,
      double max_syn,
      double mean) {
    if (op < 0 || op > 2
        || a.size()
               != (op == 0   ? 3
                   : op == 1 ? 11
                             : 13)
        || out.size() != (op == 2 ? 2 : 1) || bases < 1 || bases > INT32_MAX / 2
        || !std::isfinite(temperature) || temperature <= 0
        || (syn != 0 && syn != 1) || !std::isfinite(max_syn)
        || !std::isfinite(mean))
      throw std::invalid_argument("invalid NA sampling stage operands");
    auto i = ffi::DataType::S32, f = ffi::DataType::F32;
    if (op == 0) {
      auto real = a[0].element_type();
      if (real != f && real != ffi::DataType::F64)
        throw std::invalid_argument("NA pucker coordinates require float32/64");
      require_shape(a[0], real, {-1, 3});
      require_shape(a[1], ffi::DataType::U8, {-1, 5, 8});
      auto n = a[1].dimensions()[0];
      score::common::checked_dispatch_size(n, "NA sugar count");
      require_shape(a[2], ffi::DataType::PRED, {n, 5});
      graph_wide(out[0], n);
      return;
    }
    graph_wide(a[0]);
    auto n = a[0].dimensions()[0];
    score::common::checked_dispatch_size(n, "NA candidate count");
    graph_wide(a[1]);
    auto t = a[1].dimensions()[0];
    graph_wide(a[2], n);
    graph_wide(a[3], t);
    require_shape(a[4], ffi::DataType::PRED, {n});
    require_shape(a[5], i, {t, -1});
    auto chi = a[5].dimensions()[1];
    if (!chi) throw std::invalid_argument("NA sampler requires a chi axis");
    require_shape(a[6], f, {t, -1, chi - 1});
    constexpr auto pucker = score::na_torsion::potentials::N_PUCKER;
    require_shape(a[8], f, {2 * bases, pucker});
    require_shape(a[7], f, {2, pucker, 2 * bases});
    require_shape(a[9], f, {2});
    require_shape(a[10], f, {-1});
    if (!a[10].dimensions()[0])
      throw std::invalid_argument("NA sampler requires chi steps");
    if (op == 1)
      graph_wide(out[0], n);
    else {
      graph_wide(a[11], n);
      require_shape(a[12], i, {-1});
      auto r = a[12].dimensions()[0];
      score::common::checked_dispatch_size(r, "NA sample count");
      require_shape(out[0], i, {r, chi});
      require_shape(out[1], f, {r, chi});
    }
  }

  template <template <Device> class Ops, Device D>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t,
      int64_t bases,
      double temperature,
      int64_t syn,
      double max_syn,
      double mean) {
    using namespace pack::rotamer;
    ContextManager manager;
#define IN(T, R, K) buffer_view<T, R, D>(a[K])
#define OUT(T, R, K) buffer_view<T, R, D>(out[K])
    if (op == 0) {
      if (a[0].element_type() == ffi::DataType::F32)
        na_sampling_pucker_native<Ops, D>(
            manager,
            IN(float, 2, 0),
            IN(int64_t, 2, 1),
            IN(bool, 2, 2),
            float(temperature),
            OUT(int64_t, 1, 0));
      else
        na_sampling_pucker_native<Ops, D>(
            manager,
            IN(double, 2, 0),
            IN(int64_t, 2, 1),
            IN(bool, 2, 2),
            temperature,
            OUT(int64_t, 1, 0));
      return;
    }
    NaChiSamplingContext<D> ctx{
        IN(int64_t, 1, 0),
        IN(int64_t, 1, 1),
        IN(int64_t, 1, 2),
        IN(int64_t, 1, 3),
        IN(bool, 1, 4),
        IN(int32_t, 2, 5),
        IN(float, 3, 6),
        IN(float, 3, 7),
        IN(float, 2, 8),
        IN(float, 1, 9),
        IN(float, 1, 10),
        {bool(syn), int(bases), float(max_syn), float(mean)}};
    if (op == 1)
      na_chi_count_native<Ops, D>(manager, ctx, OUT(int64_t, 1, 0));
    else
      na_chi_fill_native<Ops, D>(
          manager,
          ctx,
          IN(int64_t, 1, 11),
          IN(int32_t, 1, 12),
          OUT(int32_t, 2, 0),
          OUT(float, 2, 1));
#undef IN
#undef OUT
  }
};
}  // namespace tmol::jax
