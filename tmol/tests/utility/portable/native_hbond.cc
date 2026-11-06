#include <tmol/score/hbond/potentials/native_hbond_pose.hh>
#include <tmol/score/hbond/potentials/native_hbond_bases.hh>
#include "native_test_ops.hh"
#include <cassert>
#include <vector>

using namespace tmol::score::hbond::potentials;
using tmol_test::SerialOps;
using tmol_test::view;

// Compile both complete native entries without framework headers or linkage.
template <typename Real>
void instantiate() {
  tmol::ContextManager mgr;
  HBondBasesCtx<Real, int, tmol::Device::CPU> bases;
  launch_hbond_bases<SerialOps, tmol::Device::CPU, Real, int>(mgr, bases);
  launch_hbond_bases<SerialOps, tmol::Device::CPU, Real, int, false>(
      mgr, bases);
  gather_hbond_bases<SerialOps, tmol::Device::CPU, Real>(mgr, bases);
  HBondCtx<Real, int, tmol::Device::CPU> pose;
  launch_hbond_pose<SerialOps, tmol::Device::CPU, Real, int>(
      pose, 1, 2, 1, false, [](auto kernel) { kernel(0); });
}
template void instantiate<float>();
template void instantiate<double>();

template <typename Real>
void check_indices() {
  HBondBlockPairSharedData<Real, 32, 12> storage;
  storage.conn_ats1[0] = 299;
  std::vector<int> paths(300 * 300);
  for (int i = 0; i < 300; ++i)
    for (int j = 0; j < 300; ++j)
      paths[i * 300 + j] = std::min(6, std::abs(i - j));
  tmol::portable::Vec<Real, 3> coords[300] = {};
  int donors[320] = {}, acceptors[320] = {}, donor_types[320] = {},
      acceptor_types[320] = {}, hybridization[320] = {};
  donors[288] = 10;
  acceptors[288] = 11;
  donor_types[288] = 299;
  acceptor_types[288] = 300;
  HBondSingleResData<Real> residue{};
  residue.n_atoms = 300;
  residue.n_conn = 1;
  residue.n_donH = residue.n_acc = 1;
  residue.coords = storage.coords1;
  residue.donH_tile_inds = storage.don_inds1;
  residue.acc_tile_inds = storage.acc_inds1;
  residue.donH_type = storage.don_type1;
  residue.acc_type = storage.acc_type1;
  residue.acc_hybridization = storage.acc_hybridization1;
  residue.path_dist = storage.path_dist1;
  hbond_load_block_into_shared<SerialOps, tmol::Device::CPU, 1, 32, Real>(
      view(coords, {300}),
      view(donors, {1, 10, 32}),
      view(acceptors, {1, 10, 32}),
      view(donor_types, {1, 10, 32}),
      view(acceptor_types, {1, 10, 32}),
      view(hybridization, {1, 10, 32}),
      view(paths.data(), {1, 300, 300}),
      0,
      9,
      residue,
      12,
      288,
      true,
      storage.conn_ats1);
  unsigned char other_paths[32] = {}, connection_distance[1] = {1};
  int separation =
      tmol::score::common::count_pair::shared_mem_inter_block_separation<32>(
          4, 10, 0, 1, 1, residue.path_dist, other_paths, connection_distance);
  assert(separation == 2);  // atom298--atom299--other residue atom0
  assert(residue.donH_type[0] == 299);
  assert(residue.acc_type[0] == 300);
}
template <typename Real>
void check_gather() {
  tmol::portable::Vec<Real, 3> coords[2] = {{1, 2, 3}, {4, 5, 6}};
  int indices[6] = {1, -1, -2, 0, 1, -2};
  tmol::portable::Vec<Real, 3> output[6];
  HBondBasesCtx<Real, int, tmol::Device::CPU> ctx{};
  ctx.rot_coords = view(coords, {2});
  ctx.derived_atom_inds = view(indices, {2, 3});
  ctx.derived_coords = view(output, {2, 3});
  tmol::ContextManager mgr;
  gather_hbond_bases<SerialOps, tmol::Device::CPU, Real>(mgr, ctx);
  for (int component = 0; component < 3; ++component) {
    assert(output[0][component] == coords[1][component]);
    assert(std::isnan(output[1][component]));
    assert(output[2][component] == 0);
    assert(output[3][component] == coords[0][component]);
    assert(output[5][component] == 0);
  }
  coords[1][0] = 19;
  gather_hbond_bases<SerialOps, tmol::Device::CPU, Real>(mgr, ctx);
  assert(output[0][0] == 19);
}

int main() {
  check_indices<float>();
  check_indices<double>();
  check_gather<float>();
  check_gather<double>();
}
