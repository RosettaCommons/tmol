#pragma once

#include <type_traits>
#include <tmol/utility/portable/device.hh>

namespace tmol::score::common {

// Unordered CUDA additions need wider storage for float pose totals. Each
// interaction is still evaluated in Real; bindings cast the completed total
// once to the public dtype. Block-pair, CPU and Metal execution retain their
// existing storage and reduction order, without a dense pair scratch buffer.
template <Device D, typename Real, typename Evaluate>
auto with_pose_energy_accumulation(bool block_pairs, Evaluate evaluate) {
  if constexpr (D == Device::CUDA && std::is_same_v<Real, float>) {
    if (!block_pairs) return evaluate(double(0));
  }
  return evaluate(Real(0));
}

}  // namespace tmol::score::common
