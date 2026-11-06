#include <tmol/pack/compiled/annealer_cpu.hh>
#include <tmol/score/common/tview_binder.hh>
#include "native_test_ops.hh"
#include <cassert>

using tmol_test::view;
using Binder = tmol::score::common::TViewBinder<tmol::Device::CPU>;

int main() {
  int32_t nres[] = {0, 2}, nrots[] = {0, 4}, offsets[] = {0, 0};
  int32_t counts[] = {0, 0, 2, 2}, oneb[] = {0, 0, 0, 2};
  int32_t res[] = {0, 0, 1, 1}, cols[] = {1, 0}, quench[4] = {};
  int64_t rows[] = {0, 0, 0, 1, 2}, edge_chunks[] = {0, 1};
  int64_t chunks[] = {0, 4};
  float e1[] = {5, 6, 5, 6}, e2[] = {0, 1, 1, 2, 0, 1, 1, 2};
  float scores[2] = {};
  int32_t current[4] = {}, best[4] = {};
  tmol::pack::compiled::CpuAnnealerContext<Binder::template View> c;
  c.pose_n_res = view(nres, {2});
  c.n_rotamers_for_pose = view(nrots, {2});
  c.rotamer_offset_for_pose = view(offsets, {2});
  c.res_for_rot = view(res, {4});
  c.quench_order = view(quench, {4});
  c.neighbor_blocks = view(cols, {2});
  c.n_rotamers_for_res = view(counts, {2, 2});
  c.oneb_offsets = view(oneb, {2, 2});
  c.neighbor_row_offsets = view(rows, {5});
  c.neighbor_chunk_offset_offsets = view(edge_chunks, {2});
  c.chunk_offsets = view(chunks, {2});
  c.energy1b = view(e1, {4});
  c.energy2b = view(e2, {8});
  c.scores = view(scores, {2, 1});
  c.current_rotamer_assignments = view(current, {2, 1, 2});
  c.best_rotamer_assignments = view(best, {2, 1, 2});
  c.chunk_size = 2;
  for (uint64_t seed : {0, 42, 520}) {
    tmol::pack::compiled::run_cpu_annealer(c, seed);
    // Each selected nonzero rotamer adds one one-body and one pair unit.
    // The unique minimum is independently known, including the empty pose.
    assert(scores[0] == 0 && scores[1] == 10);
    for (auto rot : best) assert(rot == 0);
    assert(e1[1] == 6 && e2[3] == 2 && counts[2] == 2);
  }
}
