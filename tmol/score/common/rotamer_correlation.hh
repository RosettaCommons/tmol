#pragma once

#include <tmol/utility/portable/qualifiers.hh>
#include <tmol/utility/tensor/TensorAccessor.h>

namespace tmol::score::common {
// Empty metadata means independent sampling. Nonnegative equal groups require
// validated equal active counts; their local conformer indices correspond.
template <typename Int, tmol::Device D>
TMOL_HOST_DEVICE_FUNC bool same_rotamer_group(
    TView<Int, 2, D> groups, int pose, int first, int second) {
  if (groups.size(0) == 0) return false;
  Int const group = groups[pose][first];
  return group >= 0 && group == groups[pose][second];
}
}  // namespace tmol::score::common
