#include <Eigen/Core>

#include <tmol/utility/tensor/TensorPack.h>

#include <tmol/kinematics/compiled/kernel_segscan.cuh>
#include <tmol/score/common/tuple.hh>
#include <tmol/backends/cuda/score/common/device_operations.cuda.impl.cuh>
#include <tmol/utility/nvtx.hh>

#include <moderngpu/transform.hxx>

#include <tmol/kinematics/compiled/common.hh>
#include <tmol/kinematics/compiled/native_kinematics_cuda.cuh>
#include <tmol/kinematics/compiled/params.hh>
#include <tmol/kinematics/compiled/compiled.impl.hh>

namespace tmol {
namespace kinematics {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define HomogeneousTransform Eigen::Matrix<Real, 4, 4>
#define KintreeDof Eigen::Matrix<Real, 9, 1>
#define f1f2Vectors Eigen::Matrix<Real, 6, 1>
#define Coord Eigen::Matrix<Real, 3, 1>

// Dispatch class for cuda-based generational segmented scan operations.
//
// # Scan Overview
//
// The scan operation processes linear scan paths with an associative binary
// operator, where the scan paths many have any number of additional off-path
// node inputs added *before* the scan path. For example, consider the
// operation composed of path values (P), off path values (OP) joined by an
// operator (+)::
//
//     OP_0            OP_1
//       +               +
//       |               |
//       v               v
//     P_0+--->P_1+--->P_2+--->P_3+--->P_4
//                       ^
//                       |
//                       +
//                     OP_2
//
// Represents the complete operation::
//   (OP_0 + P_0) + P_1 + (OP_1 + OP_2 + P_2) + P_3 + P_4
//
// As this is a scan, rather than reduction, this results in::
//
//     R_0-----R_1-----R_2-----R_3-----R_4
//     R_0 = OP_0 + P_0
//     R_1 = (OP_0 + P_0) + P_1
//     R_2 = (OP_0 + P_0) + P_1 + (OP_1 + OP_2 + P_2)
//     R_3 = (OP_0 + P_0) + P_1 + (OP_1 + OP_2 + P_2) + P_3
//     R_4 = (OP_0 + P_0) + P_1 + (OP_1 + OP_2 + P_2) + P_3 + P_4
//
// The off-path inputs of a scan are taken from the result values of previous
// scan operations. Scans are processed by "generation", arranged such that
// the off-path inputs of any segment are draw *exclusively* from earlier
// generations. Scans within a generation are processed via a parallel
// segmented scan.
//
// The input arguments 'nodes', 'scans' give the individual scans, with
// 'nodes' providing the scan ordering and 'scans' giving scan boundaries.
//
// The input argument 'gens' breaks these down into individual generations,
// providing both 'node' and 'scan' boundaries for each generation.
//
template <tmol::Device D, typename Real, typename Int>
struct ForwardKinDispatch {
  static auto f(
      ContextManager& mgr,
      TView<KintreeDof, 1, D> dofs,
      TView<Int, 1, D> nodes,
      TView<Int, 1, D> scans,
      TView<KinForestGenData<Int>, 1, tmol::Device::CPU> gens,
      TView<KinForestParams<Int>, 1, D> kintree)
      -> std::tuple<TPack<Coord, 1, D>, TPack<HomogeneousTransform, 1, D>> {
    NVTXRange _function(__FUNCTION__);
    using tmol::score::common::tie;
    typedef typename mgpu::launch_params_t<128, 2> launch_t;
    constexpr int nt = launch_t::nt, vt = launch_t::vt;
    auto num_atoms = dofs.size(0);

    nvtx_range_push("dispatch::alloc");
    auto HTs_t = TPack<HomogeneousTransform, 1, D>::empty({num_atoms});
    auto HTs = HTs_t.view;
    auto xs_t = TPack<Coord, 1, D>::empty({num_atoms});
    auto xs = xs_t.view;
    nvtx_range_pop();

    // temp memory for scan
    nvtx_range_push("dispatch::alloc_temp");
    int carryoutBuffer, scanBuffer, lbsBuffer;
    mgpu::tie(scanBuffer, carryoutBuffer, lbsBuffer) =
        getScanBufferSize(gens, nt, vt);
    auto scanCarryout_t =
        TPack<HomogeneousTransform, 1, D>::empty({carryoutBuffer});
    auto scanCarryout = scanCarryout_t.view;
    auto scanCodes_t = TPack<int, 1, D>::empty({carryoutBuffer});
    auto scanCodes = scanCodes_t.view;
    auto LBS_t = TPack<Int, 1, D>::empty({lbsBuffer});
    auto LBS = LBS_t.view;
    auto HTscan_t = TPack<HomogeneousTransform, 1, D>::empty({scanBuffer});
    auto HTscan = HTscan_t.view;
    nvtx_range_pop();
    forward_kinematics_cuda_native<Real, Int>(
        *current_context(mgr),
        dofs,
        nodes,
        scans,
        gens,
        kintree,
        HTs,
        xs,
        scanCarryout,
        scanCodes,
        LBS,
        HTscan);

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
      ContextManager& mgr,
      TView<Coord, 1, D> dVdx,
      TView<HomogeneousTransform, 1, D> hts,
      TView<KintreeDof, 1, D> dofs,
      TView<Int, 1, D> nodes,
      TView<Int, 1, D> scans,
      TView<KinForestGenData<Int>, 1, tmol::Device::CPU> gens,
      TView<KinForestParams<Int>, 1, D> kintree) -> TPack<KintreeDof, 1, D> {
    NVTXRange _function(__FUNCTION__);
    using tmol::score::common::tie;
    typedef typename mgpu::launch_params_t<256, 3> launch_t;
    constexpr int nt = launch_t::nt, vt = launch_t::vt;

    auto num_atoms = dVdx.size(0);

    nvtx_range_push("dispatch::dalloc");
    auto f1f2s_t = TPack<f1f2Vectors, 1, D>::empty({num_atoms});
    auto f1f2s = f1f2s_t.view;
    auto dsc_ddofs_t = TPack<KintreeDof, 1, D>::empty({num_atoms});
    auto dsc_ddofs = dsc_ddofs_t.view;
    nvtx_range_pop();

    // temp memory for scan
    nvtx_range_push("dispatch::dalloc_temp");
    int carryoutBuffer, scanBuffer, lbsBuffer;
    mgpu::tie(scanBuffer, carryoutBuffer, lbsBuffer) =
        getScanBufferSize(gens, nt, vt);
    auto scanCarryout_t = TPack<f1f2Vectors, 1, D>::empty({carryoutBuffer});
    auto scanCarryout = scanCarryout_t.view;
    auto scanCodes_t = TPack<int, 1, D>::empty({carryoutBuffer});
    auto scanCodes = scanCodes_t.view;
    auto LBS_t = TPack<Int, 1, D>::empty({lbsBuffer});
    auto LBS = LBS_t.view;
    auto f1f2scan_t = TPack<f1f2Vectors, 1, D>::empty({scanBuffer});
    auto f1f2scan = f1f2scan_t.view;
    nvtx_range_pop();
    kin_derivatives_cuda_native<Real, Int>(
        *current_context(mgr),
        dVdx,
        hts,
        dofs,
        nodes,
        scans,
        gens,
        kintree,
        f1f2s,
        dsc_ddofs,
        scanCarryout,
        scanCodes,
        LBS,
        f1f2scan);

    return dsc_ddofs_t;
  }
};

template struct ForwardKinDispatch<tmol::Device::CUDA, float, int32_t>;
template struct ForwardKinDispatch<tmol::Device::CUDA, double, int32_t>;
template struct InverseKinDispatch<tmol::Device::CUDA, float, int32_t>;
template struct InverseKinDispatch<tmol::Device::CUDA, double, int32_t>;
template struct KinDerivDispatch<tmol::Device::CUDA, float, int32_t>;
template struct KinDerivDispatch<tmol::Device::CUDA, double, int32_t>;

template struct KinForestFromStencil<
    tmol::score::common::DeviceOperations,
    tmol::Device::CUDA,
    int32_t>;
// NOTE: Intetionally not enabling int64_t as there are atomic_incremement
// operations that are needed for several steps in the kin-forest construction
// algorithm and atomic_increment is not implemented in CUDA for int64_t.
// template struct KinForestFromStencil<
//     tmol::score::common::DeviceOperations,
//     tmol::Device::CUDA,
//     int64_t>;

#undef HomogeneousTransform
#undef KintreeDof
#undef f1f2Vectors
#undef Coord

}  // namespace kinematics
}  // namespace tmol
