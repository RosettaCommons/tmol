#include <tmol/score/common/workgroup_lane.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/ljlk/potentials/ljlk.hh>
#include <tmol/score/ljlk/potentials/lj.hh>
#include <tmol/score/elec/potentials/elec.hh>
#include <tmol/score/lk_ball/potentials/lk_ball.hh>
#include <tmol/score/ljlk/potentials/ljlk_elec_pose_kernels.hh>
#include <tmol/score/ljlk/potentials/ljlk_elec_kernels.hh>
#include "native_test_ops.hh"
#include <iostream>
#include <vector>

using tmol_test::SerialOps;
using tmol_test::view;
using namespace tmol::score::ljlk::potentials;

template <class Real, class Storage>
bool check(Storage& storage, char const* name) {
  // Two chains are joined at atom 299 of the first residue and atom 0 of
  // the second. Its atom 298 is two bonds from the second residue's atom 0.
  int const connection_atom = 299;
  storage.conn_ats1[0] = connection_atom;
  std::vector<int> paths(300 * 300);
  for (int i = 0; i < 300; ++i)
    for (int j = 0; j < 300; ++j)
      paths[i * 300 + j] = std::min(6, std::abs(i - j));
  tmol::portable::Vec<Real, 3> coords[300] = {};
  int atom_types[300] = {};
  LJLKTypeParams<Real> types[1] = {};
  types[0].lj_radius = Real(1.7);
  types[0].lj_sqrt_wdepth = Real(.2);
  LJLKBlockPairSharedData<Real, 32, 12> tile;
  LJLKSingleResData<Real> residue{};
  residue.n_atoms = 300;
  residue.n_conn = 1;
  residue.coords = tile.coords1;
  residue.atom_types = tile.atom_types1;
  residue.path_dist = tile.path_dist1;
  ljlk_load_block_into_shared<SerialOps, tmol::Device::CPU, 1, 32, Real, int>(
      view(coords, {300}),
      view(atom_types, {1, 300}),
      view(types, {1}),
      view(paths.data(), {1, 300, 300}),
      0,
      residue,
      12,
      288,
      true,
      storage.conn_ats1);
  unsigned char other_path[32] = {}, connection_separation[1] = {1};
  int const separation =
      tmol::score::common::count_pair::shared_mem_inter_block_separation<32>(
          4, 10, 0, 1, 1, residue.path_dist, other_path, connection_separation);
  LJGlobalParams<Real> global{6, Real(.6), Real(2.6), Real(2.6), Real(1.75)};
  auto const p = types[0].lj_params();
  auto const bonded =
      lj_score<Real>::V_dV(Real(3), Real(separation), p, p, global);
  auto const nonbonded = lj_score<Real>::V_dV(Real(3), Real(5), p, p, global);
  bool const ok = separation == 2 && bonded.Vatr == 0 && bonded.Vrep == 0
                  && bonded.dVatr_ddist == 0 && bonded.dVrep_ddist == 0
                  && nonbonded.Vatr != 0 && nonbonded.dVrep_ddist != 0;
  if (!ok)
    std::cerr << name << ": expected excluded two-bond pair; got separation "
              << separation << ", energy " << bonded.Vatr + bonded.Vrep << "\n";
  return ok;
}

template <class Real>
bool run() {
  LJLKBlockPairSharedData<Real, 32, 12> lj;
  tmol::score::elec::potentials::ElecBlockPairSharedData<Real, 32, 12> elec;
  tmol::score::lk_ball::potentials::LKBallBlockPairSharedData<Real, 32, 4, 12>
      lk;
  ljlk_elec_detail::SharedData<Real> fused;
  LJLKElecSharedData<Real, 32, 12> portable_fused;
  bool ok = check<Real>(lj, "LJ");
  ok = check<Real>(elec, "Elec") && ok;
  ok = check<Real>(lk, "LK-ball") && ok;
  ok = check<Real>(fused, "fused CPU/CUDA") && ok;
  ok = check<Real>(portable_fused, "portable fused") && ok;
  return ok;
}
int main() { return run<float>() && run<double>() ? 0 : 1; }
