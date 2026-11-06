#pragma once
#include <tmol/utility/tensor/native.hh>
#include <tmol/utility/tensor/context_manager.hh>
#include <vector>
namespace tmol::pack::rotamer {
template <template <Device> class Ops, Device D>
struct PoseImpositionTorch {
  using Tensor = tmol::native::Tensor;
  static Tensor select(ContextManager&, std::vector<Tensor>);
  static std::vector<Tensor> plan(
      ContextManager&, std::vector<Tensor>, int64_t, int64_t);
};
}  // namespace tmol::pack::rotamer
