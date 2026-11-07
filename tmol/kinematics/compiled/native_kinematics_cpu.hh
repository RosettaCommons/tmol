#pragma once
#include "native_kinematics.hh"
namespace tmol {
namespace kinematics {
template <class Real, class Int>
void forward_kinematics_cpu_native(
    TView<KinDofs<Real>, 1, Device::CPU> dofs,
    TView<Int, 1, Device::CPU> nodes,
    TView<Int, 1, Device::CPU> scans,
    TView<KinForestGenData<Int>, 1, Device::CPU> gens,
    TView<KinForestParams<Int>, 1, Device::CPU> kintree,
    TView<KinTransform<Real>, 1, Device::CPU> HTs,
    TView<KinCoord<Real>, 1, Device::CPU> xs) {
  constexpr Device D = Device::CPU;
  int const num_atoms = dofs.size(0);
  // dofs -> HTs
  auto k_dof2ht = ([=] EIGEN_DEVICE_FUNC(int i) {
    DOFtype doftype = (DOFtype)kintree[i].doftype;
    if (doftype == ROOT) {
      HTs[i] = KinTransform<Real>::Identity();
    } else if (doftype == JUMP) {
      HTs[i] = common<D, Real, Int>::jumpTransform(dofs[i]);
    } else if (doftype == BOND) {
      HTs[i] = common<D, Real, Int>::bondTransform(dofs[i]);
    }
  });

  for (int i = 0; i < num_atoms; i++) {
    k_dof2ht(i);
  }

  // scan and accumulate HTs down atom tree
  auto k_compose =
      ([=] EIGEN_DEVICE_FUNC(int p, int i) { HTs[i] = HTs[i] * HTs[p]; });

  int ngens = gens.size(0) - 1;
  for (int gen = 0; gen < ngens; gen++) {  // loop over generations
    int scanstart = gens[gen].scan_start;
    int scanstop = gens[gen + 1].scan_start;
    for (int j = scanstart; j < scanstop; j++) {  // loop over scans
      int nodestart = gens[gen].node_start + scans[j];
      int nodestop = (j == scanstop - 1)
                         ? gens[gen + 1].node_start
                         : (gens[gen].node_start + scans[j + 1]);
      for (int k = nodestart; k < nodestop - 1; k++) {  // loop over path
        k_compose(nodes[k], nodes[k + 1]);
      }
    }
  }

  // copy atom positions
  auto k_getcoords = ([=] EIGEN_DEVICE_FUNC(int i) {
    xs[i] = HTs[i].block(3, 0, 1, 3).transpose();
  });

  for (int i = 0; i < num_atoms; i++) {
    k_getcoords(i);
  }
}

template <class Real, class Int>
void kin_derivatives_cpu_native(
    TView<KinCoord<Real>, 1, Device::CPU> dVdx,
    TView<KinTransform<Real>, 1, Device::CPU> hts,
    TView<KinDofs<Real>, 1, Device::CPU> dofs,
    TView<Int, 1, Device::CPU> nodes,
    TView<Int, 1, Device::CPU> scans,
    TView<KinForestGenData<Int>, 1, Device::CPU> gens,
    TView<KinForestParams<Int>, 1, Device::CPU> kintree,
    TView<Eigen::Matrix<Real, 6, 1>, 1, Device::CPU> f1f2s,
    TView<KinDofs<Real>, 1, Device::CPU> dsc_ddofs) {
  constexpr Device D = Device::CPU;
  int const num_atoms = dVdx.size(0);
  // calculate f1s and f2s from dVdx and HT
  auto k_f1f2s = ([=] EIGEN_DEVICE_FUNC(int i) {
    KinCoord<Real> trans = hts[i].block(3, 0, 1, 3).transpose();
    KinCoord<Real> f1 =
        dVdx[i].isZero(0) ? dVdx[i] : trans.cross(trans - dVdx[i]).transpose();
    f1f2s[i].topRows(3) = f1;
    f1f2s[i].bottomRows(3) = dVdx[i];
  });

  for (int i = 0; i < num_atoms; i++) {
    k_f1f2s(i);
  }

  // scan and accumulate f1s/f2s up atom tree
  auto k_compose =
      ([=] EIGEN_DEVICE_FUNC(int p, int i) { f1f2s[i] = f1f2s[i] + f1f2s[p]; });

  // note: if this is parallelized (over j/k)
  //   then k_compose needs to be atomic
  int ngens = gens.size(0) - 1;
  for (int gen = 0; gen < ngens; gen++) {  // loop over generations
    int scanstart = gens[gen].scan_start;
    int scanstop = gens[gen + 1].scan_start;
    for (int j = scanstart; j < scanstop; j++) {  // loop over scans
      int nodestart = gens[gen].node_start + scans[j];
      int nodestop = (j == scanstop - 1)
                         ? gens[gen + 1].node_start
                         : (gens[gen].node_start + scans[j + 1]);
      for (int k = nodestart; k < nodestop - 1; k++) {  // loop over path
        k_compose(nodes[k], nodes[k + 1]);
      }
    }
  }

  auto k_f1f2s2derivs = ([=] EIGEN_DEVICE_FUNC(int i) {
    Vec<Real, 3> f1 = f1f2s[i].topRows(3);
    Vec<Real, 3> f2 = f1f2s[i].bottomRows(3);
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

  for (int i = 0; i < num_atoms; i++) {
    k_f1f2s2derivs(i);
  }
}
}  // namespace kinematics
}  // namespace tmol
