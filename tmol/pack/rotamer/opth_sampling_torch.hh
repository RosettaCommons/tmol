#pragma once
#include <tmol/utility/tensor/native.hh>
#include <tmol/utility/tensor/context_manager.hh>
namespace tmol::pack::rotamer {
template <template <Device> class Ops, Device D>
struct OptHSamplingTorch {
  using Tensor = tmol::native::Tensor;
  static std::tuple<Tensor, Tensor> counts(
      ContextManager&, std::vector<Tensor>, bool);
  static std::tuple<Tensor, Tensor, Tensor, Tensor> fill(
      ContextManager&, std::vector<Tensor>, Tensor, int64_t);
  static std::tuple<Tensor, Tensor, Tensor> proton(
      ContextManager&, std::vector<Tensor>, int64_t);
};
}  // namespace tmol::pack::rotamer
