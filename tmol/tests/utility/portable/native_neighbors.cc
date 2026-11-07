#include <tmol/score/ljlk/potentials/ljlk_elec_pose_kernels.hh>
#include <tmol/score/na_torsion/potentials/na_torsion_pose_kernels.hh>
#include <tmol/score/cartbonded/potentials/cartbonded_pose_kernels.hh>
#include <algorithm>
#include <cassert>
#include <tmol/score/common/sphere_overlap_preparation.hh>

template <tmol::Device D>
struct SerialOps {
  template <int nt, class F>
  static void for_each_in_workgroup(F f) {
    f(0);
  }
  template <int nt, class T, class S, class O>
  static T reduce_in_workgroup(T v, S&, O) {
    return v;
  }
  template <int nt, int v, class T>
  static void copy_contiguous_data(T* dst, T const* src, int n) {
    std::copy(src, src + n, dst);
  }
  template <class Lane>
  static void synchronize_workgroup(Lane const&) {}
  static void synchronize_workgroup() {}
  template <int nt, class T, class O>
  static T shuffle_reduce_and_broadcast_in_workgroup(T v, O) {
    return v;
  }
  template <int nt, class T, class O>
  static T shuffle_reduce_in_workgroup(T v, O) {
    return v;
  }
  template <class L, class F>
  static void forall_independent(tmol::ContextManager&, int n, F f) {
    for (int i = 0; i < n; ++i) f(i);
  }

  template <class L, class F>
  static void forall(tmol::ContextManager&, int n, F f) {
    for (int i = 0; i < n; ++i) f(i);
  }
  template <class L, class F>
  static void foreach_workgroup(tmol::ContextManager&, int n, F f) {
    for (int i = 0; i < n; ++i) f(i);
  }
  template <class L, class F>
  static void foreach_independent_workgroup(tmol::ContextManager&, int n, F f) {
    for (int i = 0; i < n; ++i) f(i);
  }
  template <class L, class F>
  static void foreach_pose_workgroup(tmol::ContextManager&, int p, int n, F f) {
    for (int i = 0; i < p * n; ++i) f(i);
  }
};

using namespace tmol;
using namespace tmol::score::common::sphere_overlap;
template <class T, size_t N>
TView<T, N, Device::CPU> view(T* ptr, int64_t const (&shape)[N]) {
  int64_t stride[N];
  int64_t st = 1;
  for (int i = N - 1; i >= 0; --i) {
    stride[i] = st;
    st *= shape[i];
  }
  return TView<T, N, Device::CPU>(ptr, shape, stride);
}
int main() {
  // Candidate dispatch and atom-pair partitioning are independent: every
  // requested candidate/partition must run once, including strided groups.
  for (int count : {0, 1, 3, 129, 1025}) {
    for (int partitions : {1, 4, 16, 32}) {
      ContextManager mgr;
      std::vector<int> visits(count * partitions, 0);
      auto visit = [&](int candidate, int shard, int total) {
        assert(total == partitions && candidate >= 0 && candidate < count);
        assert(shard >= 0 && shard < partitions);
        ++visits[candidate * partitions + shard];
      };
      assert((launch_partitioned_block_neighbors<SerialOps, Device::CPU, void>(
          mgr,
          view(static_cast<int*>(nullptr), {0}),
          count,
          1,
          partitions,
          visit)));
      for (int n : visits) assert(n == 1);
    }
  }
  {
    ContextManager mgr;
    int neighbors[] = {3, 6, 2, 10, -999};
    int visits[11][4] = {};
    auto visit = [&](int candidate, int shard, int total) {
      assert(total == 4 && candidate >= 0 && candidate < 11);
      ++visits[candidate][shard];
    };
    assert((launch_partitioned_block_neighbors<SerialOps, Device::CPU, void>(
        mgr, view(neighbors, {5}), 2, 3, 4, visit)));
    for (int candidate = 0; candidate < 11; ++candidate)
      for (int shard = 0; shard < 4; ++shard)
        assert(
            visits[candidate][shard]
            == int(candidate == 6 || candidate == 2 || candidate == 10));
    neighbors[0] = 0;
    auto forbidden = [](int, int, int) { assert(false); };
    assert((launch_partitioned_block_neighbors<SerialOps, Device::CPU, void>(
        mgr, view(neighbors, {5}), 1, 1, 4, forbidden)));
    assert((!launch_partitioned_block_neighbors<SerialOps, Device::CPU, void>(
        mgr, view(static_cast<int*>(nullptr), {0}), 2, 3, 4, forbidden)));
  }
  // Empty batches still initialize their count; large batches cover more
  // than one launch group. Every pose has exactly one self-pair candidate,
  // including padded poses which the score body rejects.
  for (int n_poses : {0, 1, 3, 129}) {
    std::vector<int> neighbors(n_poses + 2, -73);
    ContextManager mgr;
    single_block_neighbor_candidates_native<SerialOps, Device::CPU, int>(
        mgr, n_poses, view(neighbors.data(), {n_poses + 1}));
    assert(neighbors[0] == n_poses);
    assert(neighbors.back() == -73);
    for (int p = 0; p < n_poses; ++p) {
      int candidate = neighbors[p + 1];
      // One upper-triangular pair per pose means candidate is the pose ID.
      assert(candidate == p);
    }
  }
  portable::Vec<double, 3> xyz[6] = {
      {0, 0, 0}, {2, 0, 0}, {10, 0, 0}, {12, 0, 0}, {-1, 0, 0}, {1, 0, 0}};
  int offset[] = {0, 2, 4}, block[] = {0, 1, 0}, pose[] = {0, 0, 1},
      types[] = {0, 1, 0}, natoms[] = {2, 2};
  int pose_types[] = {0, 1, 0, -1}, neighbors[7] = {};
  double spheres[16] = {};
  ContextManager mgr;
  auto prepare = [&] {
    compute_block_spheres_native<SerialOps, Device::CPU, double, int>(
        mgr,
        view(xyz, {6}),
        view(offset, {3}),
        view(block, {3}),
        view(pose, {3}),
        view(types, {3}),
        view(natoms, {2}),
        view(spheres, {2, 2, 4}));
    detect_compact_block_neighbors<SerialOps, Device::CPU, double, int>::f(
        mgr,
        view(pose_types, {2, 2}),
        view(spheres, {2, 2, 4}),
        view(neighbors, {7}),
        5.0);
  };
  prepare();
  assert(
      spheres[0] == 1 && spheres[3] == 1 && spheres[4] == 11
      && spheres[7] == 1);
  assert(
      neighbors[0] == 3 && neighbors[1] == 0 && neighbors[2] == 2
      && neighbors[3] == 3);
  xyz[2][0] = 4;
  xyz[3][0] = 6;
  prepare();
  assert(spheres[4] == 5);
  assert(
      neighbors[0] == 4 && neighbors[1] == 0 && neighbors[2] == 1
      && neighbors[3] == 2 && neighbors[4] == 3);
}
