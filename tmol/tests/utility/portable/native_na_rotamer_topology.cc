#include <tmol/score/na_torsion/potentials/na_rotamer_topology.hh>
#include "native_test_ops.hh"
#include <cassert>
using namespace tmol::score::na_torsion::potentials;
using tmol_test::SerialOps;
using tmol_test::view;
template <typename Int>
void check() {
  tmol::ContextManager mgr;
  Int start = sizeof(Int) == 8 ? Int(2500000000LL) : Int(100);
  Int types[] = {0, 0, 1, -1}, poses[] = {0, 0, 0, -1},
      blocks[] = {0, 0, 1, -1},
      offsets[] = {start, Int(start + 10), Int(start + 20), 0};
  Int first[] = {0, 2}, first_type[] = {0, 1}, bases[] = {0, 1},
      down[] = {-1, 0}, rings[] = {0, 1, 0, 1}, downstream[] = {2, 3, 2, 3};
  tmol::portable::Vec<Int, 2> connections[2];
  connections[0][0] = 1;
  connections[0][1] = 0;
  connections[1][0] = 0;
  connections[1][1] = 0;
  tmol::portable::Vec<Int, 3> uaids[8];
  for (int t = 0; t < 2; ++t)
    for (int a = 0; a < 4; ++a) {
      auto& u = uaids[t * 4 + a];
      u[0] = a % 2 == 0 ? a : -1;
      u[1] = a % 2 == 0 ? -1 : 0;
      u[2] = a / 2;
    }
  int64_t out_base[4], prev[4], torsions[16], out_rings[8];
  bool na[4], atom_ok[16], tor_ok[4], ring_ok[8];
  NARotamerTopology<Int, tmol::Device::CPU> ctx;
  ctx.types = view(types, {4});
  ctx.poses = view(poses, {4});
  ctx.blocks = view(blocks, {4});
  ctx.offsets = view(offsets, {4});
  ctx.bt_base = view(bases, {2});
  ctx.bt_down = view(down, {2});
  ctx.first = view(first, {1, 2});
  ctx.first_type = view(first_type, {1, 2});
  ctx.bt_ring = view(rings, {2, 2});
  ctx.connections = view(connections, {1, 2, 1});
  ctx.downstream = view(downstream, {2, 1, 2});
  ctx.bt_uaids = view(uaids, {2, 1, 4});
  ctx.base = view(out_base, {4});
  ctx.previous = view(prev, {4});
  ctx.is_na = view(na, {4});
  ctx.torsions = view(torsions, {4, 1, 4});
  ctx.atoms_ok = view(atom_ok, {4, 1, 4});
  ctx.torsions_ok = view(tor_ok, {4, 1});
  ctx.ring_ok = view(ring_ok, {4, 2});
  ctx.rings = view(out_rings, {4, 2});
  na_torsion_rotamer_topology_native<SerialOps, tmol::Device::CPU, Int>(
      mgr, ctx);
  assert(out_base[3] == -1 && !na[3] && prev[3] == -1 && !tor_ok[3]);
  for (int r = 0; r < 3; ++r) {
    assert(na[r] && tor_ok[r]);
    int other = r < 2 ? 2 : 0;
    assert(torsions[4 * r] == offsets[r]);
    assert(torsions[4 * r + 2] == int64_t(offsets[r]) + 2);
    assert(torsions[4 * r + 1] == int64_t(offsets[other]) + 2);
    assert(torsions[4 * r + 3] == int64_t(offsets[other]) + 3);
    assert(
        out_rings[2 * r] == offsets[r]
        && out_rings[2 * r + 1] == int64_t(offsets[r]) + 1);
  }
  assert(prev[0] == -1 && prev[1] == -1 && prev[2] == 0);
  for (int a = 0; a < 4; ++a)
    assert(torsions[12 + a] == -1 && !atom_ok[12 + a]);
  // An explicit connection with no background type must not dereference type
  // -1.
  first_type[1] = -1;
  na_torsion_rotamer_topology_native<SerialOps, tmol::Device::CPU, Int>(
      mgr, ctx);
  assert(!tor_ok[0] && !tor_ok[1] && torsions[1] == -1 && !atom_ok[1]);
  ctx.types = view(types, {0});
  out_base[0] = 777;
  na_torsion_rotamer_topology_native<SerialOps, tmol::Device::CPU, Int>(
      mgr, ctx);
  assert(out_base[0] == 777);
}
int main() {
  check<int32_t>();
  check<int64_t>();
}
