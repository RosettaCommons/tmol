// XLA-owned storage for CUDA totals; pair arithmetic remains caller precision.
#pragma once
#include "workspace_ffi.hh"

namespace tmol::jax {
template <template <Device> class Ops, Device D, typename Real>
class EnergyAccumulator {
 public:
  static constexpr bool precise =
      D == Device::CUDA && std::is_same_v<Real, float>;
  using Value = std::conditional_t<precise, double, Real>;

  EnergyAccumulator(
      ContextManager& context,
      ffi::AnyBuffer output,
      double* scratch,
      bool preserve_output = false)
      : output_(output),
        values_(precise ? static_cast<void*>(scratch) : output.untyped_data()) {
    if (output.size_bytes() / sizeof(Real) > INT32_MAX)
      throw std::invalid_argument("native energy dispatch exceeds int32");
    initialize(context, output, scratch, preserve_output);
  }

  // CUDA extended device lambdas require an addressable enclosing function.
  static void initialize(
      ContextManager& context,
      ffi::AnyBuffer output,
      double* scratch,
      bool preserve_output = false) {
    if constexpr (precise) {
      LAUNCH_BOX_32;
      auto count = output.size_bytes() / sizeof(Real);
      auto source = static_cast<Real*>(output.untyped_data());
      Ops<D>::template forall<launch_t>(
          context, count, [=] TMOL_DEVICE_FUNC(int i) {
            scratch[i] = preserve_output ? double(source[i]) : 0;
          });
    }
  }

  template <size_t Rank>
  auto view(std::array<int64_t, Rank> dimensions) const {
    return contiguous_view<Value, Rank, D>(values_, dimensions);
  }

  void finish(ContextManager& context) const {
    if constexpr (precise) {
      LAUNCH_BOX_32;
      auto count = output_.size_bytes() / sizeof(Real);
      auto source = static_cast<double*>(values_);
      auto destination = static_cast<Real*>(output_.untyped_data());
      Ops<D>::template forall<launch_t>(
          context, count, [=] TMOL_DEVICE_FUNC(int i) {
            destination[i] = Real(source[i]);
          });
    }
  }

 private:
  ffi::AnyBuffer output_;
  void* values_;
};
}  // namespace tmol::jax
