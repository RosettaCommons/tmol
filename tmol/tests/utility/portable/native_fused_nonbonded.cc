#include <tmol/score/ljlk/potentials/ljlk_elec_pose_kernels.hh>
#include <tmol/score/na_torsion/potentials/na_torsion_pose_kernels.hh>
#include <tmol/score/cartbonded/potentials/cartbonded_pose_kernels.hh>
#include <algorithm>
#include <cassert>
#include <vector>
template <typename Real>
struct PairData {
  struct Residue {
    int n_atoms;
  } r1, r2;
};
template <typename T, size_t N>
auto cpu_view(T* data, int64_t const (&shape)[N]) {
  int64_t strides[N];
  int64_t stride = 1;
  for (int i = N - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return tmol::TView<T, N, tmol::Device::CPU>(data, shape, strides);
}

void check_wide_pair_derivatives() {
  using namespace tmol::score::ljlk::potentials;
  using tmol::portable::Vec;
  ljlk_elec_detail::ScoringData<float> data{};
  LJLKTypeParams<float> params{};
  float charge1 = 0.4f, charge2 = -0.7f;
  data.r1.ljlk_params = data.r2.ljlk_params = &params;
  data.r1.charges = &charge1;
  data.r2.charges = &charge2;
  data.r2.rot_coord_offset = 1;
  data.elec_global_params = {80, 6, 0.4f, 1.5f, 5.5f, 0, 0, 0, 0, 0, 0};
  float weights[4] = {0, 0, 0, 0.7f};
  Vec<double, 3> accumulated[8]{};
  long double reference[8][3]{};
  for (int i = 0; i < 129; ++i) {
    Vec<float, 3> delta{
        (i % 2 ? -1.f : 1.f) * (2.0f + i * 0.003f), 0.3f, -0.2f};
    float dist2 =
        delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
    Vec<float, 3> pair[8]{};
    auto energy = ljlk_elec_detail::
        score_atom_pair<true, false, true, float, tmol::Device::CPU>(
            0,
            0,
            0,
            0,
            data,
            delta,
            dist2,
            5,
            5,
            cpu_view(pair, {4, 2}),
            cpu_view(weights, {4}),
            1);
    auto wide_energy = ljlk_elec_detail::
        score_atom_pair<true, false, true, float, tmol::Device::CPU>(
            0,
            0,
            0,
            0,
            data,
            delta,
            dist2,
            5,
            5,
            cpu_view(accumulated, {4, 2}),
            cpu_view(weights, {4}),
            1);
    for (int term = 0; term < 4; ++term) {
      assert(energy[term] == wide_energy[term]);
      for (int axis = 0; axis < 3; ++axis) {
        assert(pair[term * 2][axis] == -pair[term * 2 + 1][axis]);
        reference[term * 2][axis] += pair[term * 2][axis];
        reference[term * 2 + 1][axis] += pair[term * 2 + 1][axis];
      }
    }
  }
  assert(reference[6][1] != 0);
  for (int index = 0; index < 8; ++index)
    for (int axis = 0; axis < 3; ++axis)
      assert(accumulated[index][axis] == double(reference[index][axis]));
}

int main() {
  check_wide_pair_derivatives();
  using namespace tmol::score::common;
  // Partitioning lanes must visit every actual atom pair exactly once, also
  // across partial tiles and the intra-residue diagonal.
  for (int shards : {1, 4, 16, 32})
    for (bool intra : {false, true}) {
      PairData<double> data{{52}, {52}};
      std::vector<int> visits(52 * 52, 0);
      auto pair = [&](int s1, int s2, int a1, int a2, PairData<double> const&) {
        ++visits[(s1 + a1) * 52 + s2 + a2];
        return tmol::portable::array<double, 1>{1.0};
      };
      for (int s1 : {0, 32})
        for (int s2 : {0, 32})
          for (int shard = 0; shard < shards; ++shard)
            for (int lane = 0; lane < 32; ++lane) {
              int const tid = shard * 32 + lane;
              if (intra)
                IntraResBlockEvaluation<
                    PairData,
                    AllAtomPairSelector,
                    tmol::Device::CPU,
                    32,
                    32,
                    1,
                    double,
                    int>::
                    eval_intrares_atom_pairs(
                        tid, s1, s2, pair, data, 32 * shards);
              else
                InterResBlockEvaluation<
                    PairData,
                    AllAtomPairSelector,
                    tmol::Device::CPU,
                    32,
                    32,
                    1,
                    double,
                    int>::
                    eval_interres_atom_pair(
                        tid, s1, s2, pair, data, 32 * shards);
            }
      for (int i = 0; i < 52; ++i)
        for (int j = 0; j < 52; ++j)
          assert(visits[i * 52 + j] == int(!intra || i < j));
    }
  return 0;
}
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
using namespace tmol::score::ljlk::potentials;
auto f32 = &ljlk_elec_native_forward<
    true,
    false,
    false,
    SerialOps,
    tmol::Device::CPU,
    float,
    int>;
auto f64 = &ljlk_elec_native_forward<
    true,
    false,
    false,
    SerialOps,
    tmol::Device::CPU,
    double,
    int>;
auto w64 = &ljlk_elec_native_forward<
    true,
    true,
    false,
    SerialOps,
    tmol::Device::CPU,
    double,
    int>;
auto r64 = &ljlk_elec_native_forward<
    false,
    true,
    true,
    SerialOps,
    tmol::Device::CPU,
    double,
    int>;
auto b64 = &ljlk_elec_native_forward<
    true,
    true,
    true,
    SerialOps,
    tmol::Device::CPU,
    double,
    int>;

// Bindings may widen derivative accumulation without changing float physics.
auto wide32 = &ljlk_elec_native_forward<
    true,
    false,
    false,
    SerialOps,
    tmol::Device::CPU,
    float,
    int,
    double>;
auto wide_weighted32 = &ljlk_elec_native_forward<
    true,
    true,
    false,
    SerialOps,
    tmol::Device::CPU,
    float,
    int,
    double>;
auto wide_rotamer32 = &ljlk_elec_native_forward<
    true,
    true,
    true,
    SerialOps,
    tmol::Device::CPU,
    float,
    int,
    double>;
