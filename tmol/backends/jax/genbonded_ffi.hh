// Typed XLA buffers for the common whole-pose generic-bonded kernels.
#pragma once

#include <array>
#include <limits>
#include <stdexcept>
#include <vector>
#include <tuple>
#include <type_traits>
#include "xla/ffi/api/ffi.h"
#include <tmol/score/genbonded/potentials/genbonded_pose_kernels.hh>

namespace tmol::jax {
namespace ffi = xla::ffi;
using tmol::portable::Vec;

inline void require_shape(
    ffi::AnyBuffer buffer,
    ffi::DataType dtype,
    std::initializer_list<int64_t> shape) {
  auto dims = buffer.dimensions();
  if (buffer.element_type() != dtype || dims.size() != shape.size()) {
    throw std::invalid_argument("native FFI buffer dtype/rank mismatch");
  }
  size_t i = 0;
  for (int64_t size : shape) {
    if ((size >= 0 && dims[i] != size)
        || dims[i] > std::numeric_limits<int32_t>::max()) {
      throw std::invalid_argument("native FFI buffer shape mismatch");
    }
    ++i;
  }
}

template <typename T, size_t Rank, Device D>
TView<T, Rank, D> contiguous_view(
    void* pointer, std::array<int64_t, Rank> sizes) {
  std::array<int64_t, Rank> strides;
  int64_t stride = 1;
  for (size_t i = Rank; i-- > 0;) {
    strides[i] = stride;
    stride *= sizes[i];
  }
  return {reinterpret_cast<T*>(pointer), sizes.data(), strides.data()};
}

template <typename T, size_t Rank, Device D>
TView<T, Rank, D> buffer_view(ffi::AnyBuffer buffer) {
  std::array<int64_t, Rank> sizes;
  std::copy_n(buffer.dimensions().begin(), Rank, sizes.begin());
  return contiguous_view<T, Rank, D>(buffer.untyped_data(), sizes);
}

inline void validate_genbonded_tables(
    const std::vector<ffi::AnyBuffer>& a, ffi::DataType real, int64_t ports) {
  auto integer = ffi::DataType::S32;
  require_shape(a[7], integer, {-1});
  auto types = a[7].dimensions()[0];
  require_shape(a[5], integer, {types, ports, 13, 3});
  require_shape(a[6], integer, {-1, 5});
  require_shape(a[8], real, {a[6].dimensions()[0], 5});
  require_shape(a[9], integer, {types, -1, 5});
  auto atoms = a[9].dimensions()[1];
  require_shape(a[10], integer, {types, atoms});
  require_shape(a[11], integer, {types, ports, a[5].dimensions()[2] + 1});
  require_shape(a[12], integer, {types, ports});
  require_shape(a[13], integer, {types, atoms});
  require_shape(a[14], integer, {types});
  require_shape(a[15], integer, {-1, 6});
  require_shape(a[16], real, {-1, 5});
  require_shape(a[17], integer, {-1, 5});
  require_shape(a[18], real, {-1, 5});
}
template <Device D, typename Real>
auto genbonded_tables(const std::vector<ffi::AnyBuffer>& a) {
  using I = int32_t;
  return std::make_tuple(
      buffer_view<Vec<I, 3>, 3, D>(a[5]),
      buffer_view<Vec<I, 5>, 1, D>(a[6]),
      buffer_view<I, 1, D>(a[7]),
      buffer_view<Vec<Real, 5>, 1, D>(a[8]),
      buffer_view<Vec<I, 5>, 2, D>(a[9]),
      buffer_view<I, 2, D>(a[10]),
      buffer_view<I, 3, D>(a[11]),
      buffer_view<I, 2, D>(a[12]),
      buffer_view<I, 2, D>(a[13]),
      buffer_view<I, 1, D>(a[14]),
      buffer_view<Vec<I, 6>, 1, D>(a[15]),
      buffer_view<Vec<Real, 5>, 1, D>(a[16]),
      buffer_view<Vec<I, 5>, 1, D>(a[17]),
      buffer_view<Vec<Real, 5>, 1, D>(a[18]));
}

// All topology buffers are prepared and validated on the host before tracing.
// Validate their dimensions again at the ABI boundary before creating views.
inline void validate_genbonded(
    const std::vector<ffi::AnyBuffer>& a,
    ffi::AnyBuffer energy,
    const ffi::AnyBuffer* gradient) {
  if (a.size() != 19) {
    throw std::invalid_argument("generic bonded FFI expects 19 operands");
  }
  auto real = a[0].element_type();
  if (real != ffi::DataType::F32 && real != ffi::DataType::F64) {
    throw std::invalid_argument(
        "generic bonded coordinates must be float32/64");
  }
  auto integer = ffi::DataType::S32;
  require_shape(a[0], real, {-1, -1, 3});
  require_shape(a[3], integer, {a[0].dimensions()[0], -1});
  auto poses = a[3].dimensions()[0], blocks = a[3].dimensions()[1];
  require_shape(a[1], integer, {poses * blocks});
  require_shape(a[2], integer, {poses, blocks});
  require_shape(a[4], integer, {poses, blocks, -1, 2});
  auto ports = a[4].dimensions()[2];
  validate_genbonded_tables(a, real, ports);
  require_shape(energy, real, {poses});
  if (gradient) {
    require_shape(*gradient, real, {poses, a[0].dimensions()[1], 3});
  }
  if (poses * a[0].dimensions()[1] > std::numeric_limits<int32_t>::max()) {
    throw std::invalid_argument("generic bonded atom count exceeds int32");
  }
}

template <
    template <Device> class Ops,
    Device D,
    typename Real,
    typename AccumReal = Real>
void execute_genbonded(
    const std::vector<ffi::AnyBuffer>& a, void* energy, void* gradient) {
  using I = int32_t;
  auto poses = a[0].dimensions()[0], atoms = a[0].dimensions()[1];
  auto xyz =
      contiguous_view<Vec<Real, 3>, 1, D>(a[0].untyped_data(), {poses * atoms});
  auto e = contiguous_view<AccumReal, 4, D>(energy, {1, poses, 1, 1});
  auto g = contiguous_view<Vec<AccumReal, 3>, 2, D>(
      gradient, {1, gradient ? poses * atoms : 0});
  ContextManager context;
  std::apply(
      [&](auto... tables) {
        score::genbonded::potentials::
            genbonded_pose_forward<Ops, D, Real, I, AccumReal, AccumReal>(
                context,
                xyz,
                buffer_view<I, 1, D>(a[1]),
                buffer_view<I, 2, D>(a[2]),
                buffer_view<I, 2, D>(a[3]),
                buffer_view<Vec<I, 2>, 3, D>(a[4]),
                tables...,
                false,
                gradient != nullptr,
                e,
                g);
      },
      genbonded_tables<D, Real>(a));
}

inline std::vector<ffi::AnyBuffer> native_arguments(ffi::RemainingArgs args) {
  std::vector<ffi::AnyBuffer> result;
  for (size_t i = 0; i < args.size(); ++i) {
    auto value = args.get<ffi::AnyBuffer>(i);
    if (!value)
      throw std::invalid_argument("native FFI expected buffer operand");
    result.push_back(*value);
  }
  return result;
}
struct GenBondedKernel {
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      ffi::AnyBuffer e,
      const ffi::AnyBuffer* g) {
    validate_genbonded(a, e, g);
  }
  template <template <Device> class Ops, Device D, typename Real>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      ffi::AnyBuffer e,
      const ffi::AnyBuffer* g) {
    execute_genbonded<Ops, D, Real>(
        a, e.untyped_data(), g ? g->untyped_data() : nullptr);
  }
};

// Separate target preserves the legacy result ABI and CPU/float64 execution.
struct GenBondedPreciseKernel : GenBondedKernel {
  static size_t values(size_t poses, size_t atoms, bool derivative) {
    auto gradients = derivative ? score::common::checked_dispatch_product(
                                      poses,
                                      score::common::checked_dispatch_product(
                                          atoms, 3, "generic bonded atoms"),
                                      "generic bonded gradients")
                                : 0;
    if (poses > INT32_MAX - gradients)
      throw std::length_error("generic bonded accumulator exceeds int32");
    return poses + gradients;
  }
  static size_t workspace_bytes(
      const std::vector<ffi::AnyBuffer>& a, bool derivative) {
    return a[0].element_type() == ffi::DataType::F32
               ? values(a[0].dimensions()[0], a[0].dimensions()[1], derivative)
                     * sizeof(double)
               : 0;
  }
  template <template <Device> class Ops, Device D, typename Real>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      ffi::AnyBuffer energy,
      const ffi::AnyBuffer* gradient,
      ffi::AnyBuffer workspace) {
    if constexpr (!std::is_same_v<Real, float>) {
      GenBondedKernel::execute<Ops, D, Real>(a, energy, gradient);
    } else {
      auto poses = a[0].dimensions()[0], atoms = a[0].dimensions()[1];
      auto count = values(poses, atoms, gradient != nullptr);
      auto* wide = static_cast<double*>(workspace.untyped_data());
      execute_genbonded<Ops, D, Real, double>(
          a, wide, gradient ? wide + poses : nullptr);
      auto* e = static_cast<Real*>(energy.untyped_data());
      auto* g =
          gradient ? static_cast<Real*>(gradient->untyped_data()) : nullptr;
      ContextManager context;
      LAUNCH_BOX_32;
      Ops<D>::template forall<launch_t>(
          context, count, [=] TMOL_DEVICE_FUNC(int i) {
            if (i < poses)
              e[i] = Real(wide[i]);
            else
              g[i - poses] = Real(wide[i]);
          });
    }
  }
};
}  // namespace tmol::jax
