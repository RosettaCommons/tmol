#include <tmol/pack/compiled/interaction_graph_bounds.hh>
#include <tmol/score/common/serial_cpu_operations.hh>
#include <tmol/utility/tensor/owned_host_tensor.hh>
#include "native_test_ops.hh"
#include <cassert>

using tmol::Device;
template <Device D>
struct HostOps : tmol::score::common::SerialCPUOperations {
  template <class L, class F>
  static void forall_independent(tmol::ContextManager& mgr, int n, F f) {
    forall<L>(mgr, n, f);
  }
  template <class L, class F>
  static void foreach_workgroup(tmol::ContextManager& mgr, int n, F f) {
    forall<L>(mgr, n, f);
  }
  template <class L, class F>
  static void foreach_grouped_workgroup(
      tmol::ContextManager& mgr, int p, int n, F f) {
    forall<L>(mgr, p * n, f);
  }
  template <int N, class F>
  static void for_each_in_workgroup(F f) {
    f(0);
  }
  template <int N, class T, class Op>
  static T shuffle_reduce_in_workgroup(T v, Op) {
    return v;
  }
  template <class T>
  static void store_idempotent(T& dst, T value) {
    dst = value;
  }
};

template <class Real, class Int>
void check() {
  using tmol_test::view;
  Int counts[] = {3, 2, 0, 0, 0, 0}, offsets[] = {0, 3, 5, 5, 5, 5};
  Int poses[] = {0, 0, 0, 0, 0};
  int32_t blocks[] = {0, 0, 0, 1, 1};
  // Five one-body entries, six pair entries, and a duplicate tail.
  int32_t indices[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0,
                       0, 1, 1, 2, 2, 0, 0, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4, 3};
  Real energies[] = {1, 8, 2, 3, 4, 1, 5, 4, 1, 0, 3, 1};
  tmol::ContextManager mgr;
  auto [rot, block] = tmol::pack::compiled::prepare_ig_bump_bounds<
      HostOps,
      Device::CPU,
      Real,
      Int,
      tmol::OwnedHostTensor>(
      mgr,
      2,
      view(counts, {2, 3}),
      view(offsets, {2, 3}),
      view(poses, {5}),
      view(blocks, {5}),
      view(indices, {3, 12}),
      view(energies, {12}),
      11);
  Real expected[] = {3, 9, 2, 3, 5};
  for (int i = 0; i < 5; ++i) assert(rot.view[i] == expected[i]);
  assert(block.view[0][0] == 2 && block.view[0][1] == 3);
  assert(block.view[0][2] == 1234 && block.view[1][0] == 1234);

  using namespace tmol::pack::compiled;
  BumpGraphContext<Real, Int, Device::CPU> data{
      2,
      11,
      view(counts, {2, 3}),
      view(offsets, {2, 3}),
      view(poses, {5}),
      view(blocks, {5}),
      view(indices, {3, 12}),
      view(energies, {12})};
  int32_t chunks[6]{}, adjacent[18]{};
  int64_t pair_counts[18]{}, pair_offsets[18]{};
  int64_t chunk_sizes[4]{}, chunk_offsets[4]{};
  Real one_body[5]{}, pair_energies[12]{}, rot_bounds[5]{};
  Real block_bounds[] = {1234, 1234, 1234, 1234, 1234, 1234};
  data.n_chunks_for_block = view(chunks, {2, 3});
  data.respair_is_adjacent = view(adjacent, {2, 3, 3});
  data.n_chunks_for_block_pair = view(pair_counts, {2, 3, 3});
  data.chunk_pair_offset_for_block_pair = view(pair_offsets, {2, 3, 3});
  data.energy1b = view(one_body, {5});
  launch_ig_bump_block_counts<HostOps, Device::CPU>(mgr, data);
  assert(chunks[0] == 2 && chunks[1] == 1 && chunks[2] == 0);
  assert(pair_counts[1] == 2 && pair_counts[3] == 2);
  auto total_chunks = HostOps<Device::CPU>::template scan_and_return_total<
      tmol::portable::scan_type::exclusive>(
      mgr, pair_counts, pair_offsets, 18, tmol::portable::plus_t<int64_t>());
  assert(total_chunks == 4);
  data.chunk_pair_adjacency = view(chunk_sizes, {4});
  data.chunk_pair_offsets = view(chunk_offsets, {4});
  launch_ig_bump_chunk_sizes<HostOps, Device::CPU>(mgr, data);
  assert(chunk_sizes[0] == 4 && chunk_sizes[1] == 2);
  auto total_energies = HostOps<Device::CPU>::template scan_and_return_total<
      tmol::portable::scan_type::exclusive>(
      mgr, chunk_sizes, chunk_offsets, 4, tmol::portable::plus_t<int64_t>());
  assert(total_energies == 12);
  data.energy2b = view(pair_energies, {12});
  data.best_energy_for_rot = view(rot_bounds, {5});
  data.best_energy_per_block = view(block_bounds, {2, 3});
  launch_ig_bump_bounds<HostOps, Device::CPU>(mgr, data);
  for (int i = 0; i < 5; ++i) assert(rot_bounds[i] == expected[i]);
  assert(block_bounds[0] == 2 && block_bounds[1] == 3);
  for (int i = 2; i < 6; ++i) assert(block_bounds[i] == 1234);
}
int main() {
  check<float, int32_t>();
  check<double, int32_t>();
  check<float, int64_t>();
  check<double, int64_t>();
  using tmol::pack::compiled::ig_bump_workspace_sizes;
  auto empty = ig_bump_workspace_sizes(0, 0, 0, 0, 2, -1);
  assert(empty.pair_cells == 0 && empty.unique_entries == 0);
  bool invalid = false, overflow = false;
  try {
    ig_bump_workspace_sizes(1, 1, 1, 1, 0, -1);
  } catch (std::invalid_argument const&) {
    invalid = true;
  }
  try {
    ig_bump_workspace_sizes(1, 65536, 1, 1, 2, -1);
  } catch (std::overflow_error const&) {
    overflow = true;
  }
  assert(invalid && overflow);
}
