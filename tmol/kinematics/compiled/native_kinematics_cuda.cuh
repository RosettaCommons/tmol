#pragma once
#include "native_kinematics.hh"
#include "kernel_segscan.cuh"
#include <tmol/utility/portable/scan_workspace.hh>
#include <moderngpu/transform.hxx>
#include <tmol/utility/nvtx.hh>
namespace tmol {
namespace kinematics {
#define HomogeneousTransform Eigen::Matrix<Real, 4, 4>
#define f1f2Vectors Eigen::Matrix<Real, 6, 1>
#define Coord Eigen::Matrix<Real, 3, 1>
template <typename Real>
struct HTRawBuffer {
  Real data[16];
};

template <typename Real>
struct f1f2VecsRawBuffer {
  Real data[6];
};

// This function gets the memory needed for the output and temporary buffers
//   in the segmented scan.
//
// It first loops over all generations and finds the generation with longest
//   scan_length = #elements + #segments
//
// Then using the kernel launch parameters 'nt' and 'vt':
//      nt: # of threads per CTA
//      vt: # of elements processed per thread
// it computes three quantities:
//   scanSize - "scan_length" above, used for the output buffer
//   lbsSize - a temporary buffer used for the load-balancing scan parameters
//     = ceil (scanSize / (vt*nt)) + 1
//   carryoutSize - a temporary buffer used for managing "carryout" from
//                  each of the subscans.
//      = ceil( (count+num_segments) / (vt*nt) )
//        + ceil( (count+num_segments) / (vt*nt*nt) )
//        + ceil( (count+num_segments) / (vt*nt*nt*nt) )
//        + ...
//      [stopping the summation when the arg to ceil becomes < 1].
//
// These are used to preallocate the memory used in each generation of the scan.
template <typename Int>
auto getScanBufferSize(
    TView<KinForestGenData<Int>, 1, tmol::Device::CPU> gens, Int nt, Int vt)
    -> mgpu::tuple<Int, Int, Int> {
  auto ngens = gens.size(0) - 1;
  int64_t scanSize = 0;
  for (int gen = 0; gen < ngens; ++gen) {
    int64_t node_begin = gens[gen].node_start;
    int64_t scan_begin = gens[gen].scan_start;
    int64_t node_end = gens[gen + 1].node_start;
    int64_t scan_end = gens[gen + 1].scan_start;
    if (node_begin < 0 || scan_begin < 0 || node_end < node_begin
        || scan_end < scan_begin)
      throw std::invalid_argument("Nonmonotonic kinematic scan generations");
    int64_t nnodes = node_end - node_begin;
    int64_t nsegs = scan_end - scan_begin;
    if (nnodes > std::numeric_limits<int64_t>::max() - nsegs)
      throw std::overflow_error("Kinematic scan generation exceeds capacity");
    scanSize = std::max(nnodes + nsegs, scanSize);
  }
  auto sizes = portable::scan_workspace_sizes<Int>(scanSize, nt, vt);
  return {sizes[0], sizes[1], sizes[2]};
}

// the composite operation for the forward pass: apply a transform
//   qt1/qt2 -> HT1/HT2 -> HT1*HT2 -> qt12' -> norm(qt12')
template <tmol::Device D, typename Real, typename Int>
struct htcompose_t : public std::binary_function<
                         HTRawBuffer<Real>,
                         HTRawBuffer<Real>,
                         HTRawBuffer<Real>> {
  MGPU_HOST_DEVICE HTRawBuffer<Real> operator()(
      HTRawBuffer<Real> p, HTRawBuffer<Real> i) const {
    HomogeneousTransform ab = Eigen::Map<HomogeneousTransform>(i.data)
                              * Eigen::Map<HomogeneousTransform>(p.data);

    return *((HTRawBuffer<Real>*)ab.data());
  }
};

// the composite operation for the backward pass: sum f1s & f2s
template <tmol::Device D, typename Real, typename Int>
struct f1f2compose_t : public std::binary_function<
                           f1f2VecsRawBuffer<Real>,
                           f1f2VecsRawBuffer<Real>,
                           f1f2VecsRawBuffer<Real>> {
  MGPU_HOST_DEVICE f1f2VecsRawBuffer<Real> operator()(
      f1f2VecsRawBuffer<Real> p, f1f2VecsRawBuffer<Real> i) const {
    f1f2Vectors ab =
        Eigen::Map<f1f2Vectors>(i.data) + Eigen::Map<f1f2Vectors>(p.data);

    return *((f1f2VecsRawBuffer<Real>*)ab.data());
  }
};

template <class Real, class Int>
void forward_kinematics_cuda_native(
    mgpu::context_t& context,
    TView<KinDofs<Real>, 1, Device::CUDA> dofs,
    TView<Int, 1, Device::CUDA> nodes,
    TView<Int, 1, Device::CUDA> scans,
    TView<KinForestGenData<Int>, 1, Device::CPU> gens,
    TView<KinForestParams<Int>, 1, Device::CUDA> kintree,
    TView<KinTransform<Real>, 1, Device::CUDA> HTs,
    TView<KinCoord<Real>, 1, Device::CUDA> xs,
    TView<KinTransform<Real>, 1, Device::CUDA> scanCarryout,
    TView<int, 1, Device::CUDA> scanCodes,
    TView<Int, 1, Device::CUDA> LBS,
    TView<KinTransform<Real>, 1, Device::CUDA> HTscan) {
  constexpr Device D = Device::CUDA;
  using launch_t = mgpu::launch_params_t<128, 2>;
  int const num_atoms = dofs.size(0);
  HTRawBuffer<Real> init = {
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};  // identity xform

  // dofs -> HTs
  nvtx_range_push("dispatch::dof2ht");
  auto k_dof2ht = ([=] EIGEN_DEVICE_FUNC(int i) {
    DOFtype doftype = (DOFtype)kintree[i].doftype;
    if (doftype == ROOT) {
      HTs[i] = HomogeneousTransform::Identity();
    } else if (doftype == JUMP) {
      HTs[i] = common<D, Real, Int>::jumpTransform(dofs[i]);
    } else if (doftype == BOND) {
      HTs[i] = common<D, Real, Int>::bondTransform(dofs[i]);
    }
  });

  mgpu::transform(k_dof2ht, num_atoms, context);
  nvtx_range_pop();

  auto ngens = gens.size(0) - 1;
  for (int gen = 0; gen < ngens; ++gen) {
    int nodestart = gens[gen].node_start, scanstart = gens[gen].scan_start;

    int nnodes = gens[gen + 1].node_start - nodestart;
    int nscans = gens[gen + 1].scan_start - scanstart;
    if (nnodes == 0 && nscans == 0) {
      continue;
    }

    // reindexing function
    nvtx_range_push("dispatch::segscan");
    auto k_reindex = [=] MGPU_DEVICE(int index, int seg, int rank) {
      if (nodestart + index >= nodes.size(0) || nodestart + index < 0) {
        return *((HTRawBuffer<Real>*)HTs[0].data());
      }
      if (nodes[nodestart + index] >= HTs.size(0)
          || nodes[nodestart + index] < 0) {
        return *((HTRawBuffer<Real>*)HTs[0].data());
      }

      assert(nodestart + index < nodes.size(0) && nodestart + index >= 0);
      assert(
          nodes[nodestart + index] < HTs.size(0)
          && nodes[nodestart + index] >= 0);
      return *((HTRawBuffer<Real>*)HTs[nodes[nodestart + index]].data());
    };

    // mgpu does not play nicely with eigen types
    // instead, we wrap the raw data buffer as QuatTransRawBuffer
    //      and use eigen:map to reconstruct on device
    tmol::kinematics::kernel_segscan<launch_t>(
        k_reindex,
        nnodes,
        &scans.data()[scanstart],
        nscans,
        (HTRawBuffer<Real>*)(HTscan.data()->data()),
        (HTRawBuffer<Real>*)(scanCarryout.data()->data()),
        &scanCodes.data()[0],
        &LBS.data()[0],
        htcompose_t<D, Real, Int>(),
        init,
        context);
    nvtx_range_pop();
    // gpuErrPeek;
    // gpuErrSync;

    // unindex for gen i
    // this would be nice to incorporate into kernel_segscan (as the indexing
    // is)
    nvtx_range_push("dispatch::unindex");
    auto k_unindex = [=] MGPU_DEVICE(int index) {
      if (nodestart + index >= nodes.size(0) || nodestart + index < 0) {
        return;  // *((HTRawBuffer<Real>*)HTs[0].data());
      }
      assert(nodestart + index < nodes.size(0) && nodestart + index >= 0);
      if (nodes[nodestart + index] >= HTs.size(0)
          || nodes[nodestart + index] < 0) {
        return;  // *((HTRawBuffer<Real>*)HTs[0].data());
      }

      assert(
          nodes[nodestart + index] < HTs.size(0)
          && nodes[nodestart + index] >= 0);
      HTs[nodes[nodestart + index]] = HTscan[index];
    };

    mgpu::transform(k_unindex, nnodes, context);
    nvtx_range_pop();
    nvtx_range_pop();
  }

  // copy atom positions
  auto k_getcoords = ([=] EIGEN_DEVICE_FUNC(int i) {
    assert(i < HTs.size(0) && i >= 0);
    xs[i] = HTs[i].block(3, 0, 1, 3).transpose();
  });

  mgpu::transform(k_getcoords, num_atoms, context);
}

template <class Real, class Int, class SumReal = Real>
void kin_derivatives_cuda_native(
    mgpu::context_t& context,
    TView<KinCoord<Real>, 1, Device::CUDA> dVdx,
    TView<KinTransform<Real>, 1, Device::CUDA> hts,
    TView<KinDofs<Real>, 1, Device::CUDA> dofs,
    TView<Int, 1, Device::CUDA> nodes,
    TView<Int, 1, Device::CUDA> scans,
    TView<KinForestGenData<Int>, 1, Device::CPU> gens,
    TView<KinForestParams<Int>, 1, Device::CUDA> kintree,
    TView<Eigen::Matrix<SumReal, 6, 1>, 1, Device::CUDA> f1f2s,
    TView<KinDofs<Real>, 1, Device::CUDA> dsc_ddofs,
    TView<Eigen::Matrix<SumReal, 6, 1>, 1, Device::CUDA> scanCarryout,
    TView<int, 1, Device::CUDA> scanCodes,
    TView<Int, 1, Device::CUDA> LBS,
    TView<Eigen::Matrix<SumReal, 6, 1>, 1, Device::CUDA> f1f2scan) {
  constexpr Device D = Device::CUDA;
  using launch_t = mgpu::launch_params_t<256, 3>;
  int const num_atoms = dVdx.size(0);
  f1f2VecsRawBuffer<SumReal> init = {0, 0, 0, 0, 0, 0};  // identity

  // Preserve coordinate arithmetic in Real, but allow wider scan and atomic
  // sums when branches contribute to the same destination. Cast once before
  // the existing degree-of-freedom derivative calculation.
  // calculate f1s and f2s from dVdx and HT
  nvtx_range_push("dispatch::ddof2ht");
  auto k_f1f2s = ([=] EIGEN_DEVICE_FUNC(int i) {
    Coord trans = hts[i].block(3, 0, 1, 3).transpose();
    Coord f1 =
        dVdx[i].isZero(0) ? dVdx[i] : trans.cross(trans - dVdx[i]).transpose();
    f1f2s[i].topRows(3) = f1.template cast<SumReal>();
    f1f2s[i].bottomRows(3) = dVdx[i].template cast<SumReal>();
  });

  // mgpu::standard_context_t context;
  mgpu::transform(k_f1f2s, num_atoms, context);
  nvtx_range_pop();

  auto ngens = gens.size(0) - 1;
  for (int gen = 0; gen < ngens; ++gen) {
    int nodestart = gens[gen].node_start, scanstart = gens[gen].scan_start;
    int nnodes = gens[gen + 1].node_start - nodestart;
    int nscans = gens[gen + 1].scan_start - scanstart;

    if (nnodes == 0 && nscans == 0) {
      continue;
    }

    // reindexing function
    nvtx_range_push("dispatch::dsegscan");
    auto k_reindex = [=] MGPU_DEVICE(int index, int seg, int rank) {
      assert(nodestart + index < nodes.size(0) && nodestart + index >= 0);
      assert(
          nodes[nodestart + index] < f1f2s.size(0)
          && nodes[nodestart + index] >= 0);
      return *(
          (f1f2VecsRawBuffer<SumReal>*)f1f2s[nodes[nodestart + index]].data());
    };

    // mgpu does not play nicely with eigen types
    // instead, we wrap the raw data buffer
    //      and use eigen:map to reconstruct on device
    tmol::kinematics::kernel_segscan<launch_t, mgpu::scan_type_exc>(
        k_reindex,
        nnodes,
        &scans.data()[scanstart],
        nscans,
        (f1f2VecsRawBuffer<SumReal>*)(f1f2scan.data()->data()),
        (f1f2VecsRawBuffer<SumReal>*)(scanCarryout.data()->data()),
        &scanCodes.data()[0],
        &LBS.data()[0],
        f1f2compose_t<D, SumReal, Int>(),
        init,
        context);
    nvtx_range_pop();

    // unindex for gen i.  ENSURE ATOMIC
    nvtx_range_push("dispatch::dunindex");
    auto k_unindex = [=] MGPU_DEVICE(int index) {
      assert(nodestart + index < nodes.size(0) && nodestart + index >= 0);
      assert(
          nodes[nodestart + index] < f1f2s.size(0)
          && nodes[nodestart + index] >= 0);
      for (int kk = 0; kk < 6; ++kk) {
        atomicAdd(&(f1f2s[nodes[nodestart + index]][kk]), f1f2scan[index][kk]);
      }
    };

    mgpu::transform(k_unindex, nnodes, context);
    nvtx_range_pop();
  }

  nvtx_range_push("dispatch::f1f2_to_deriv");
  auto k_f1f2s2derivs = ([=] EIGEN_DEVICE_FUNC(int i) {
    Vec<Real, 3> f1 = f1f2s[i].topRows(3).template cast<Real>();
    Vec<Real, 3> f2 = f1f2s[i].bottomRows(3).template cast<Real>();
    if (kintree[i].doftype == ROOT) {
      dsc_ddofs[i] = Vec<Real, 9>::Constant(0);
    } else if (kintree[i].doftype == JUMP) {
      dsc_ddofs[i] = common<D, Real, Int>::jumpDerivatives(
          dofs[i], hts[i], hts[kintree[i].parent], f1, f2);
    } else if (kintree[i].doftype == BOND) {
      dsc_ddofs[i] = common<D, Real, Int>::bondDerivatives(
          dofs[i], hts[i], hts[kintree[i].parent], f1, f2);
    }
  });

  mgpu::transform(k_f1f2s2derivs, num_atoms, context);
  nvtx_range_pop();
}
#undef HomogeneousTransform
#undef f1f2Vectors
#undef Coord
}  // namespace kinematics
}  // namespace tmol
