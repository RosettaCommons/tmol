// Thin typed execution for shared disulfide and metal candidate physics.
#pragma once
#include "graph_topology_ffi.hh"
#include "disulfide_ffi.hh"
#include "metal_ffi.hh"
#include <tmol/score/disulfide/potentials/disulfide_rotamer_kernels.hh>
#include <tmol/score/metal/potentials/metal_rotamer_kernels.hh>
namespace tmol::jax {
template <bool Metal>
struct ConnectionOperands {
  static constexpr int block = Metal ? 15 : 8;
  static constexpr int pose = block + 1, type = block + 2, counts = block + 3,
                       extra = block + 4;
  static void validate(const std::vector<ffi::AnyBuffer>& a) {
    auto real = a[0].element_type(), i = ffi::DataType::S32;
    if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
      throw std::invalid_argument("connection coordinates require float32/64");
    require_shape(a[0], real, {-1, 3});
    require_shape(a[1], i, {-1});
    auto r = a[1].dimensions()[0];
    for (int k : {block, pose, type}) require_shape(a[k], i, {r});
    require_shape(a[2], i, {-1, -1});
    auto p = a[2].dimensions()[0], b = a[2].dimensions()[1];
    require_shape(a[3], i, {p, b});
    require_shape(a[counts], i, {p, b});
    require_shape(a[4], i, {p, b, -1, 2});
    if constexpr (Metal)
      validate_metal_tables(a, real, a[4].dimensions()[2]);
    else
      validate_disulfide_tables(a, real, a[4].dimensions()[2]);
  }
};
template <bool Metal>
struct ConnectionTopologyStage : ConnectionOperands<Metal> {
  using C = ConnectionOperands<Metal>;
  static int fill(int64_t, size_t) { return 0; }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t extent,
      int64_t width) {
    if ((op != 0 && op != 1) || a.size() != C::extra + 2 * op + 1
        || out.size() != (op ? 2 : 1))
      throw std::invalid_argument("invalid connection candidate operands");
    C::validate(a);
    require_shape(
        a.back(),
        ffi::DataType::S32,
        {a[2].dimensions()[0], a[2].dimensions()[1]});
    auto r = a[1].dimensions()[0];
    if (op) {
      graph_wide(a[C::extra], r);
      graph_wide(a[C::extra + 1], r);
      require_shape(out[0], ffi::DataType::S32, {3, -1});
      require_shape(
          out[1], ffi::DataType::S32, {Metal ? 3 : 2, out[0].dimensions()[1]});
      if (extent < 0 || extent > INT32_MAX
          || width
                 != score::common::checked_dispatch_product(
                     r, extent, "connection dispatch"))
        throw std::invalid_argument("invalid connection dispatch extent");
    } else
      graph_wide(out[0], r);
  }
  template <typename Real, template <Device> class Ops, Device D>
  static void launch(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t extent,
      int64_t width) {
    using I = int32_t;
    using I2 = Vec<I, 2>;
    using V3 = Vec<Real, 3>;
    ContextManager manager;
#define IN(T, R, K) buffer_view<T, R, D>(a[K])
#define OUT(T, R, K) buffer_view<T, R, D>(out[K])
    if constexpr (Metal) {
      using Site = score::metal::potentials::MetalSiteParams<Real>;
      auto args = std::make_tuple(
          IN(I, 2, 3),
          IN(I, 1, C::block),
          IN(I, 1, C::pose),
          IN(I, 1, C::type),
          IN(I, 2, C::counts),
          IN(I2, 3, 4),
          IN(I, 2, 5),
          IN(I, 2, 6),
          IN(I, 2, 7),
          IN(I, 2, 8),
          IN(Site, 3, 9),
          IN(I2, 2, 10),
          IN(I, 3, 12),
          IN(Real, 3, 13));
      std::apply(
          [&](auto... common) {
            if (op)
              score::metal::potentials::
                  metal_rotamer_fill_native<Ops, D, Real, I>(
                      manager,
                      IN(I, 2, 2),
                      common...,
                      IN(int64_t, 1, C::extra),
                      IN(int64_t, 1, C::extra + 1),
                      OUT(I, 2, 0),
                      OUT(I, 2, 1),
                      extent,
                      width,
                      buffer_view<I, 2, D>(a.back()));
            else
              score::metal::potentials::
                  metal_rotamer_count_native<Ops, D, Real, I>(
                      manager,
                      common...,
                      OUT(int64_t, 1, 0),
                      buffer_view<I, 2, D>(a.back()));
          },
          args);
    } else {
      auto args = std::make_tuple(
          IN(I, 1, C::block),
          IN(I, 1, C::pose),
          IN(I, 1, C::type),
          IN(I, 2, C::counts),
          IN(I2, 3, 4),
          IN(bool, 2, 5));
      std::apply(
          [&](auto... common) {
            if (op)
              score::disulfide::potentials::
                  disulfide_rotamer_fill_native<Ops, D, Real, I>(
                      manager,
                      IN(I, 2, 2),
                      common...,
                      IN(int64_t, 1, C::extra),
                      IN(int64_t, 1, C::extra + 1),
                      OUT(I, 2, 0),
                      OUT(I, 2, 1),
                      extent,
                      width,
                      buffer_view<I, 2, D>(a.back()));
            else
              score::disulfide::potentials::
                  disulfide_rotamer_count_native<Ops, D, Real, I>(
                      manager,
                      common...,
                      OUT(int64_t, 1, 0),
                      buffer_view<I, 2, D>(a.back()));
          },
          args);
    }
#undef IN
#undef OUT
  }
  template <template <Device> class Ops, Device D>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t extent,
      int64_t width) {
    if (a[0].element_type() == ffi::DataType::F32)
      launch<float, Ops, D>(a, out, op, extent, width);
    else
      launch<double, Ops, D>(a, out, op, extent, width);
  }
};
template <bool Metal>
struct RotamerConnectionStage : ConnectionOperands<Metal> {
  using C = ConnectionOperands<Metal>;
  static int fill(int64_t, size_t) { return 0; }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t,
      int64_t,
      double) {
    if ((op != 0 && op != 1) || a.size() != C::extra + 2 + op
        || out.size() != (Metal ? 1 : 2))
      throw std::invalid_argument("invalid sparse connection score operands");
    C::validate(a);
    auto real = a[0].element_type();
    require_shape(a[C::extra], ffi::DataType::S32, {Metal ? 3 : 2, -1});
    auto n = a[C::extra].dimensions()[1], atoms = a[0].dimensions()[0];
    require_shape(a[C::extra + 1], ffi::DataType::S32, {3, n});
    if (op) {
      require_shape(a[C::extra + 2], real, {1, n});
      require_shape(out[0], real, {atoms, 3});
    } else
      require_shape(out[0], real, {1, n});
    if constexpr (!Metal)
      require_shape(
          out[1],
          ffi::DataType::U8,
          {op ? 0 : atoms * 3 * (real == ffi::DataType::F32 ? 4 : 8)});
  }
  template <bool Backward, template <Device> class Ops, Device D, typename Real>
  static void launch(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out) {
    using I = int32_t;
    using I2 = Vec<I, 2>;
    using V3 = Vec<Real, 3>;
    ContextManager manager;
    auto n = a[C::extra].dimensions()[1];
    auto e = contiguous_view<Real, 2, D>(
        Backward ? nullptr : out[0].untyped_data(), {1, Backward ? 0 : n});
    void* gradient = Backward ? out[0].untyped_data()
                              : (Metal ? nullptr : out[1].untyped_data());
    auto g = contiguous_view<Vec<Real, 3>, 2, D>(
        gradient, {1, gradient ? a[0].dimensions()[0] : 0});
    auto weights = contiguous_view<Real, 2, D>(
        Backward ? a[C::extra + 2].untyped_data() : nullptr,
        {1, Backward ? n : 0});
#define IN(T, R, K) buffer_view<T, R, D>(a[K])
    if constexpr (Metal) {
      std::apply(
          [&](auto... tables) {
            score::metal::potentials::
                metal_rotamer_score_native<Backward, Ops, D, Real, I>(
                    manager,
                    IN(V3, 1, 0),
                    IN(I, 1, 1),
                    IN(I, 2, 3),
                    IN(I, 1, C::block),
                    IN(I, 1, C::pose),
                    IN(I, 1, C::type),
                    IN(I2, 3, 4),
                    tables...,
                    IN(I, 2, C::extra + 1),
                    IN(I, 2, C::extra),
                    Backward,
                    true,
                    e,
                    g,
                    weights);
          },
          metal_tables<D, Real>(a));
    } else {
      std::apply(
          [&](auto... tables) {
            score::disulfide::potentials::
                disulfide_rotamer_score_native<Backward, Ops, D, Real, I>(
                    manager,
                    IN(V3, 1, 0),
                    IN(I, 1, 1),
                    IN(I, 1, C::block),
                    IN(I, 1, C::type),
                    tables...,
                    IN(I, 2, C::extra + 1),
                    IN(I, 2, C::extra),
                    e,
                    g,
                    weights);
          },
          disulfide_tables<D, Real>(a));
    }
#undef IN
  }
  template <template <Device> class Ops, Device D>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t,
      int64_t,
      double) {
    if (a[0].element_type() == ffi::DataType::F32) {
      if (op)
        launch<true, Ops, D, float>(a, out);
      else
        launch<false, Ops, D, float>(a, out);
    } else {
      if (op)
        launch<true, Ops, D, double>(a, out);
      else
        launch<false, Ops, D, double>(a, out);
    }
  }
};
}  // namespace tmol::jax
