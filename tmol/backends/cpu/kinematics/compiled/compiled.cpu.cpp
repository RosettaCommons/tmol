#include <Eigen/Core>

#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/backends/cpu/score/common/device_operations.cpu.impl.hh>

#include <tmol/kinematics/compiled/common.hh>
#include <tmol/kinematics/compiled/native_kinematics_cpu.hh>
#include <tmol/kinematics/compiled/params.hh>
#include <tmol/kinematics/compiled/compiled.impl.hh>

namespace tmol {
namespace kinematics {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define HomogeneousTransform Eigen::Matrix<Real, 4, 4>
#define KintreeDof Eigen::Matrix<Real, 9, 1>
#define Coord Eigen::Matrix<Real, 3, 1>

template <tmol::Device D, typename Real, typename Int>
struct ForwardKinDispatch {
  static auto f(
      ContextManager&,
      TView<KintreeDof, 1, D> dofs,
      TView<Int, 1, D> nodes,
      TView<Int, 1, D> scans,
      TView<KinForestGenData<Int>, 1, tmol::Device::CPU> gens,
      TView<KinForestParams<Int>, 1, D> kintree)
      -> std::tuple<TPack<Coord, 1, D>, TPack<HomogeneousTransform, 1, D> > {
    auto num_atoms = dofs.size(0);

    auto HTs_t = TPack<HomogeneousTransform, 1, D>::empty({num_atoms});
    auto HTs = HTs_t.view;
    auto xs_t = TPack<Coord, 1, D>::empty({num_atoms});
    auto xs = xs_t.view;

    forward_kinematics_cpu_native<Real, Int>(
        dofs, nodes, scans, gens, kintree, HTs, xs);

    return {xs_t, HTs_t};
  }
};

template <tmol::Device D, typename Real, typename Int>
struct InverseKinDispatch {
  static auto f(
      ContextManager& mgr,
      TView<Coord, 1, D> coords,
      TView<Int, 1, D> parent,
      TView<Int, 1, D> frame_x,
      TView<Int, 1, D> frame_y,
      TView<Int, 1, D> frame_z,
      TView<Int, 1, D> doftype) -> TPack<KintreeDof, 1, D> {
    auto num_atoms = coords.size(0);

    // fd: we could eliminate HT allocation and calculate on the fly
    auto HTs_t = TPack<HomogeneousTransform, 1, D>::empty({num_atoms});
    auto HTs = HTs_t.view;
    auto dofs_t = TPack<KintreeDof, 1, D>::empty({num_atoms});
    auto dofs = dofs_t.view;

    inverse_kinematics_native<
        tmol::score::common::DeviceOperations,
        D,
        Real,
        Int>(
        mgr, coords, parent, frame_x, frame_y, frame_z, doftype, HTs, dofs);

    return dofs_t;
  }
};

template <tmol::Device D, typename Real, typename Int>
struct KinDerivDispatch {
  static auto f(
      ContextManager&,
      TView<Coord, 1, D> dVdx,
      TView<HomogeneousTransform, 1, D> hts,
      TView<KintreeDof, 1, D> dofs,
      TView<Int, 1, D> nodes,
      TView<Int, 1, D> scans,
      TView<KinForestGenData<Int>, 1, tmol::Device::CPU> gens,
      TView<KinForestParams<Int>, 1, D> kintree) -> TPack<KintreeDof, 1, D> {
    auto num_atoms = dVdx.size(0);

    auto f1f2s_t = TPack<Vec<Real, 6>, 1, D>::empty({num_atoms});
    auto f1f2s = f1f2s_t.view;
    auto dsc_ddofs_t = TPack<KintreeDof, 1, D>::empty({num_atoms});
    auto dsc_ddofs = dsc_ddofs_t.view;

    kin_derivatives_cpu_native<Real, Int>(
        dVdx, hts, dofs, nodes, scans, gens, kintree, f1f2s, dsc_ddofs);

    return dsc_ddofs_t;
  }
};

template struct ForwardKinDispatch<tmol::Device::CPU, float, int32_t>;
template struct ForwardKinDispatch<tmol::Device::CPU, double, int32_t>;
template struct InverseKinDispatch<tmol::Device::CPU, float, int32_t>;
template struct InverseKinDispatch<tmol::Device::CPU, double, int32_t>;
template struct KinDerivDispatch<tmol::Device::CPU, float, int32_t>;
template struct KinDerivDispatch<tmol::Device::CPU, double, int32_t>;

template struct KinForestFromStencil<
    tmol::score::common::DeviceOperations,
    tmol::Device::CPU,
    int32_t>;

// NOTE: Intetionally not enabling int64_t as there are atomic_incremement
// operations that are needed for several steps in the kin-forest construction
// algorithm and atomic_increment is not implemented in CUDA for int64_t.
// template struct KinForestFromStencil<
//     tmol::score::common::DeviceOperations,
//     tmol::Device::CPU,
//     int64_t>;

#undef HomogeneousTransform
#undef KintreeDof
#undef Coord

}  // namespace kinematics
}  // namespace tmol
