#pragma once

#include <tmol/utility/portable/matrix.hh>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/tuple.hh>

namespace tmol {
namespace score {
namespace constraint {
namespace potentials {

using tmol::portable::Vec;

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real>
struct GetTorsionAngleDispatch {
  static auto forward(ContextManager& mgr, TView<Vec<Real, 3>, 2, D> coords)
      -> std::tuple<TPack<Real, 1, D>, TPack<Vec<Real, 3>, 2, D>>;
};

template <template <tmol::Device> class Ops, tmol::Device D, typename Int>
struct ConstraintCandidateDispatch {
  static std::
      tuple<tmol::native::Tensor, tmol::native::Tensor, tmol::native::Tensor>
      forward(
          ContextManager& mgr,
          TView<Vec<Int, 3>, 2, D> atoms,
          TView<Int, 1, D> n_unique,
          TView<Vec<Int, 3>, 1, D> unique_blocks,
          TView<Int, 1, D> coordinate_offsets,
          TView<Int, 2, D> n_conformers,
          TView<Int, 2, D> first_conformer,
          bool expanded);
};

}  // namespace potentials
}  // namespace constraint
}  // namespace score
}  // namespace tmol
