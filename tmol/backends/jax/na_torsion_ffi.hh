// Typed JAX binding for shared nucleic-acid coordinate kernels.
#pragma once
#include "genbonded_ffi.hh"
#include "graph_topology_ffi.hh"
#include <tmol/score/na_torsion/potentials/na_torsion_native.hh>
#include <tmol/score/na_torsion/potentials/na_rotamer_topology.hh>

namespace tmol::jax {
inline void validate_na_tables(
    const std::vector<ffi::AnyBuffer>& a, ffi::DataType real) {
  auto boolean = ffi::DataType::PRED;
  require_shape(a[8], real, {2, 6, 3});
  require_shape(a[9], real, {2, 6});
  require_shape(a[10], real, {2, 10, 4});
  require_shape(a[11], real, {10, 10});
  for (int index : {12, 13, 20, 21, 22}) require_shape(a[index], real, {2});
  require_shape(a[14], real, {2, 10});
  require_shape(a[15], real, {2, 3, 3});
  require_shape(a[16], real, {2, 2, 2});
  require_shape(a[17], real, {2, 3, 2});
  require_shape(a[18], real, {2, 10, 10});
  require_shape(a[19], boolean, {10});
}

template <typename Real, typename Int>
auto na_context(
    const std::vector<ffi::AnyBuffer>& a,
    double pucker_temperature,
    double bin_blend_sdev,
    int poses,
    int blocks,
    int atoms,
    Real* output,
    Real* derivatives,
    const Real* weights = nullptr) {
  if (!std::isfinite(pucker_temperature) || pucker_temperature <= 0
      || !std::isfinite(bin_blend_sdev) || bin_blend_sdev <= 0)
    throw std::invalid_argument(
        "NA smoothing widths must be finite and positive");
  const auto* coords = static_cast<const Real*>(a[0].untyped_data());
  const auto* base = static_cast<const Int*>(a[1].untyped_data());
  const auto* is_na = static_cast<const bool*>(a[2].untyped_data());
  const auto* torsion_indices = static_cast<const Int*>(a[3].untyped_data());
  const auto* torsion_ok = static_cast<const bool*>(a[4].untyped_data());
  const auto* ring_indices = static_cast<const Int*>(a[5].untyped_data());
  const auto* ring_ok = static_cast<const bool*>(a[6].untyped_data());
  const auto* prev = static_cast<const Int*>(a[7].untyped_data());
  const auto* backbone_means = static_cast<const Real*>(a[8].untyped_data());
  const auto* backbone_sdev = static_cast<const Real*>(a[9].untyped_data());
  const auto* sugar_means = static_cast<const Real*>(a[10].untyped_data());
  const auto* chi_means = static_cast<const Real*>(a[11].untyped_data());
  const auto* sdev_sugar = static_cast<const Real*>(a[12].untyped_data());
  const auto* sdev_chi = static_cast<const Real*>(a[13].untyped_data());
  const auto* well_pucker = static_cast<const Real*>(a[14].untyped_data());
  const auto* well_alpha_gamma = static_cast<const Real*>(a[15].untyped_data());
  const auto* well_bibii_pucker =
      static_cast<const Real*>(a[16].untyped_data());
  const auto* well_alphanext_bibii =
      static_cast<const Real*>(a[17].untyped_data());
  const auto* well_chi_syn = static_cast<const Real*>(a[18].untyped_data());
  const auto* weight_bb = static_cast<const Real*>(a[20].untyped_data());
  const auto* weight_chi = static_cast<const Real*>(a[21].untyped_data());
  const auto* weight_sugar = static_cast<const Real*>(a[22].untyped_data());
  const auto* is_north = static_cast<const bool*>(a[19].untyped_data());

  return score::na_torsion::potentials::NATorsionContext<Real, Int>{
      coords,
      base,
      is_na,
      torsion_indices,
      torsion_ok,
      ring_indices,
      ring_ok,
      prev,
      backbone_means,
      backbone_sdev,
      sugar_means,
      chi_means,
      sdev_sugar,
      sdev_chi,
      well_pucker,
      well_alpha_gamma,
      well_bibii_pucker,
      well_alphanext_bibii,
      well_chi_syn,
      is_north,
      weight_bb,
      weight_chi,
      weight_sugar,
      Real(pucker_temperature),
      Real(bin_blend_sdev),
      poses,
      blocks,
      atoms,
      int(a[11].dimensions()[0]),
      derivatives,
      output,
      weights,
      int64_t(poses) * blocks,
      1};
}

struct NaTorsionKernel {
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      ffi::AnyBuffer energy,
      const ffi::AnyBuffer* gradient) {
    if (a.size() != 23)
      throw std::invalid_argument("NA FFI expects 23 operands");
    auto real = a[0].element_type();
    if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
      throw std::invalid_argument("NA coordinates must be float32/64");
    auto integer = ffi::DataType::S32, boolean = ffi::DataType::PRED;
    require_shape(a[0], real, {-1, -1, 3});
    auto poses = a[0].dimensions()[0], atoms = a[0].dimensions()[1];
    require_shape(a[1], integer, {poses, -1});
    auto blocks = a[1].dimensions()[1];
    require_shape(a[2], boolean, {poses, blocks});
    require_shape(a[3], integer, {poses, blocks, 10, 4});
    require_shape(a[4], boolean, {poses, blocks, 10});
    require_shape(a[5], integer, {poses, blocks, 5});
    require_shape(a[6], boolean, {poses, blocks, 5});
    require_shape(a[7], integer, {poses, blocks});
    validate_na_tables(a, real);
    require_shape(energy, real, {2, poses});
    if (gradient) require_shape(*gradient, real, {2, poses, atoms, 3});
    if (poses * atoms > std::numeric_limits<int32_t>::max()
        || poses * blocks > std::numeric_limits<int32_t>::max())
      throw std::invalid_argument("NA flattened layout exceeds int32");
  }

  template <template <Device> class Ops, Device D, typename Real>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      ffi::AnyBuffer energy,
      const ffi::AnyBuffer* gradient,
      double pucker_temperature,
      double bin_blend_sdev) {
    ContextManager manager;
    auto ctx = na_context<Real, int32_t>(
        a,
        pucker_temperature,
        bin_blend_sdev,
        a[0].dimensions()[0],
        a[1].dimensions()[1],
        a[0].dimensions()[0] * a[0].dimensions()[1],
        static_cast<Real*>(energy.untyped_data()),
        gradient ? static_cast<Real*>(gradient->untyped_data()) : nullptr);
    using namespace score::na_torsion::potentials;
    if (gradient)
      launch_na_torsion_native<true, false, Ops, D>(manager, ctx);
    else
      launch_na_torsion_native<false, false, Ops, D>(manager, ctx);
  }
};
struct NaRotamerTopologyStage {
  static int fill(int64_t, size_t) { return 0; }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t operation,
      int64_t,
      int64_t) {
    if (operation != 0 || a.size() != 12 || out.size() != 8)
      throw std::invalid_argument("invalid NA candidate topology operands");
    auto i = ffi::DataType::S32, b = ffi::DataType::PRED,
         wide = ffi::DataType::U8;
    require_shape(a[0], i, {-1});
    auto r = a[0].dimensions()[0];
    for (int k : {1, 2, 3}) require_shape(a[k], i, {r});
    require_shape(a[4], i, {-1, -1});
    auto p = a[4].dimensions()[0], n = a[4].dimensions()[1];
    require_shape(a[5], i, {p, n});
    require_shape(a[6], i, {p, n, -1, 2});
    require_shape(a[8], i, {-1});
    auto t = a[8].dimensions()[0];
    require_shape(a[7], i, {t, a[6].dimensions()[2], -1});
    require_shape(a[9], i, {t, 10, 4, 3});
    require_shape(a[10], i, {t, 5});
    require_shape(a[11], i, {t});
    graph_wide(out[0], r);
    graph_wide(out[1], r);
    require_shape(out[2], b, {r});
    require_shape(out[3], wide, {r, 10, 4, 8});
    require_shape(out[4], b, {r, 10, 4});
    require_shape(out[5], b, {r, 10});
    require_shape(out[6], wide, {r, 5, 8});
    require_shape(out[7], b, {r, 5});
  }
  template <template <Device> class Ops, Device D>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t,
      int64_t,
      int64_t) {
    using namespace score::na_torsion::potentials;
    using I2 = Vec<int32_t, 2>;
    using I3 = Vec<int32_t, 3>;
#define IN(T, R, K) buffer_view<T, R, D>(a[K])
#define OUT(T, R, K) buffer_view<T, R, D>(out[K])
    NARotamerTopology<int32_t, D> ctx{
        IN(int32_t, 1, 0),  IN(int32_t, 1, 1),  IN(int32_t, 1, 2),
        IN(int32_t, 1, 3),  IN(int32_t, 1, 8),  IN(int32_t, 1, 11),
        IN(int32_t, 2, 4),  IN(int32_t, 2, 5),  IN(int32_t, 2, 10),
        IN(I2, 3, 6),       IN(int32_t, 3, 7),  IN(I3, 3, 9),
        OUT(int64_t, 1, 0), OUT(int64_t, 1, 1), OUT(bool, 1, 2),
        OUT(int64_t, 3, 3), OUT(bool, 3, 4),    OUT(bool, 2, 5),
        OUT(bool, 2, 7),    OUT(int64_t, 2, 6)};
#undef IN
#undef OUT
    ContextManager manager;
    na_torsion_rotamer_topology_native<Ops, D>(manager, ctx);
  }
};

struct RotamerNaTorsionStage {
  static int fill(int64_t, size_t) { return 0; }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t,
      int64_t,
      double,
      double) {
    if ((op != 0 && op != 1) || a.size() != 23 + op || out.size() != 1)
      throw std::invalid_argument("invalid sparse NA scoring operands");
    auto real = a[0].element_type(), b = ffi::DataType::PRED,
         wide = ffi::DataType::U8;
    if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
      throw std::invalid_argument("NA coordinates require float32/64");
    require_shape(a[0], real, {-1, 3});
    graph_wide(a[1]);
    auto r = a[1].dimensions()[0];
    require_shape(a[2], b, {r});
    require_shape(a[3], wide, {r, 10, 4, 8});
    require_shape(a[4], b, {r, 10});
    require_shape(a[5], wide, {r, 5, 8});
    require_shape(a[6], b, {r, 5});
    graph_wide(a[7], r);
    validate_na_tables(a, real);
    if (op) {
      require_shape(a[23], real, {2, r});
      require_shape(out[0], real, {a[0].dimensions()[0], 3});
    } else
      require_shape(out[0], real, {2, r});
  }
  template <typename Real, template <Device> class Ops, Device D>
  static void launch(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      bool backward,
      double temperature,
      double width) {
    auto ctx = na_context<Real, int64_t>(
        a,
        temperature,
        width,
        a[1].dimensions()[0],
        1,
        a[0].dimensions()[0],
        backward ? nullptr : static_cast<Real*>(out[0].untyped_data()),
        backward ? static_cast<Real*>(out[0].untyped_data()) : nullptr,
        backward ? static_cast<const Real*>(a[23].untyped_data()) : nullptr);
    ContextManager manager;
    using namespace score::na_torsion::potentials;
    if (backward)
      launch_na_torsion_native<true, true, Ops, D>(manager, ctx);
    else
      launch_na_torsion_native<false, false, Ops, D>(manager, ctx);
  }
  template <template <Device> class Ops, Device D>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t,
      int64_t,
      double temperature,
      double width) {
    if (a[0].element_type() == ffi::DataType::F32)
      launch<float, Ops, D>(a, out, op, temperature, width);
    else
      launch<double, Ops, D>(a, out, op, temperature, width);
  }
};
}  // namespace tmol::jax
