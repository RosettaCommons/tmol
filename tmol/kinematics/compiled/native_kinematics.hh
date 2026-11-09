#pragma once
#include "kinematics_math.hh"
#include "params_types.hh"
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/score/common/launch_box_macros.hh>

namespace tmol {
namespace kinematics {
template <class Real>
using KinDofs = Eigen::Matrix<Real, 9, 1>;
template <class Real>
using KinTransform = Eigen::Matrix<Real, 4, 4>;
template <class Real>
using KinCoord = Eigen::Matrix<Real, 3, 1>;

// All atom indices address the supplied coordinate/forest arrays, including
// the virtual root at index zero. The caller allocates outputs and transform
// scratch; both are fully assigned. Transform elements retain Eigen's native
// column-major storage. Ops owns only execution and stream selection.
template <
    template <tmol::Device> class Ops,
    tmol::Device D,
    class Real,
    class Int>
void inverse_kinematics_native(
    ContextManager& mgr,
    TView<KinCoord<Real>, 1, D> coords,
    TView<Int, 1, D> parent,
    TView<Int, 1, D> frame_x,
    TView<Int, 1, D> frame_y,
    TView<Int, 1, D> frame_z,
    TView<Int, 1, D> doftype,
    TView<KinTransform<Real>, 1, D> HTs,
    TView<KinDofs<Real>, 1, D> dofs) {
  int const num_atoms = coords.size(0);
  LAUNCH_BOX_128;
  auto k_coords2hts = ([=] EIGEN_DEVICE_FUNC(int i) {
    if (i == 0) {
      HTs[i] = KinTransform<Real>::Identity();
    } else {
      HTs[i] = common<D, Real, Int>::hts_from_frames(
          coords[i],
          coords[frame_x[i]],
          coords[frame_y[i]],
          coords[frame_z[i]]);
    }
  });

  Ops<D>::template forall<launch_t>(mgr, num_atoms, k_coords2hts);

  auto k_hts2dofs = ([=] EIGEN_DEVICE_FUNC(int i) {
    KinTransform<Real> lclHT;
    if (doftype[i] == ROOT) {
      dofs[i] = KinDofs<Real>::Constant(0);  // for num deriv check
    } else {
      lclHT = HTs[i] * common<D, Real, Int>::ht_inv(HTs[parent[i]]);

      if (doftype[i] == JUMP) {
        dofs[i] = common<D, Real, Int>::invJumpTransform(lclHT);
      } else if (doftype[i] == BOND) {
        dofs[i] = common<D, Real, Int>::invBondTransform(lclHT);
      }
    }
  });

  Ops<D>::template forall<launch_t>(mgr, num_atoms, k_hts2dofs);
}
}  // namespace kinematics
}  // namespace tmol
