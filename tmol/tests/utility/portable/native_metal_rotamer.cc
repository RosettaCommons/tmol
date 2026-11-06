#include <tmol/score/metal/potentials/metal_rotamer_kernels.hh>
#include "native_test_ops.hh"
#include <cassert>
#include <set>
#include <utility>
using namespace tmol::score::metal::potentials;
using tmol_test::SerialOps;
using tmol_test::view;
template <typename Int, typename Real>
void check(bool coupled = false, bool bridge = false) {
  tmol::ContextManager mgr;
  Int first_types[] = {0, 1}, first[] = {0, 2}, blocks[] = {0, 0, 1, 1, -1},
      poses[] = {0, 0, 0, 0, -1}, types[] = {0, 0, 1, 1, -1}, counts[] = {2, 2};
  Int groups[] = {7, 7};
  tmol::TView<Int, 2, tmol::Device::CPU> group_view{};
  if (coupled) group_view = view(groups, {1, 2});
  Int atom[] = {0, 0}, metal[] = {0, -1}, virt[] = {-1, NOT_A_SITE},
      key[] = {-1, 0}, internal[] = {-1, -1};
  Real d0[] = {0, Real(2.1)};
  if (bridge) internal[1] = 1;
  tmol::portable::Vec<Int, 2> conns[2], fans[2];
  conns[0][0] = 1;
  conns[0][1] = 0;
  conns[1][0] = 0;
  conns[1][1] = 0;
  fans[0][0] = 0;
  fans[0][1] = 1;
  fans[1][0] = -1;
  fans[1][1] = -1;
  MetalSiteParams<Real> params[2] = {};
  int64_t sizes[5] = {}, offsets[5] = {};
  metal_rotamer_count_native<SerialOps, tmol::Device::CPU, Real, Int>(
      mgr,
      view(first_types, {1, 2}),
      view(blocks, {5}),
      view(poses, {5}),
      view(types, {5}),
      view(counts, {1, 2}),
      view(conns, {1, 2, 1}),
      view(atom, {2, 1}),
      view(metal, {2, 1}),
      view(virt, {2, 1}),
      view(key, {2, 1}),
      view(params, {2, 1, 1}),
      view(fans, {2, 1}),
      view(internal, {2, 1, 1}),
      view(d0, {2, 1, 1}),
      view(sizes, {5}),
      group_view);
  assert(sizes[0] == (coupled ? 2 : 3) && sizes[1] == sizes[0]);
  assert(sizes[2] == (bridge ? (coupled ? 1 : 2) : 0));
  assert(sizes[3] == sizes[2] && sizes[4] == 0);
  int total = 0;
  for (int i = 0; i < 5; ++i) {
    offsets[i] = total;
    total += sizes[i];
  }
  assert(total == (coupled ? (bridge ? 6 : 4) : (bridge ? 10 : 6)));
  Int pairs[30] = {}, terms[30] = {};
  metal_rotamer_fill_native<SerialOps, tmol::Device::CPU, Real, Int>(
      mgr,
      view(first, {1, 2}),
      view(first_types, {1, 2}),
      view(blocks, {5}),
      view(poses, {5}),
      view(types, {5}),
      view(counts, {1, 2}),
      view(conns, {1, 2, 1}),
      view(atom, {2, 1}),
      view(metal, {2, 1}),
      view(virt, {2, 1}),
      view(key, {2, 1}),
      view(params, {2, 1, 1}),
      view(fans, {2, 1}),
      view(internal, {2, 1, 1}),
      view(d0, {2, 1, 1}),
      view(sizes, {5}),
      view(offsets, {5}),
      view(pairs, {3, total}),
      view(terms, {3, total}),
      coupled ? 2 : 3,
      5 * (coupled ? 2 : 3),
      group_view);
  std::set<std::pair<Int, Int>> actual;
  int fan_count = 0, bridge_count = 0;
  for (int i = 0; i < total; ++i) {
    assert(pairs[i] == 0);
    Int a = pairs[total + i], b = pairs[2 * total + i];
    assert(a < 2);
    if (a == b) {
      assert(terms[2 * total + i] == -1);
      assert(terms[i] == a && terms[total + i] == a);
      ++fan_count;
    } else {
      assert(b >= 2 && b < 4);
      if (terms[2 * total + i] == -3) {
        assert(bridge && terms[i] == b && terms[total + i] == a);
        ++bridge_count;
      } else {
        assert(terms[2 * total + i] == 0);
        assert(terms[i] == a && terms[total + i] == b);
      }
    }
    if (coupled && b >= 2) assert(b == 2 + a);
    actual.emplace(a, b);
  }
  assert(actual.size() == size_t(coupled ? 4 : 6) && fan_count == 2);
  assert(bridge_count == (bridge ? (coupled ? 2 : 4) : 0));
  auto volatile f = &metal_rotamer_score_native<
      false,
      SerialOps,
      tmol::Device::CPU,
      Real,
      Int>;
  auto volatile b = &metal_rotamer_score_native<
      true,
      SerialOps,
      tmol::Device::CPU,
      Real,
      Int>;
  (void)f;
  (void)b;
}
// Bridge metadata must preserve the declared index width before atom gathers.
template <typename Int, typename Real>
void check_bridge_index_width() {
  Int high = 101;
  if constexpr (sizeof(Int) > 4) high += Int(1) << 40;
  Int types[] = {0, 1, 2};
  Int metals[] = {high, -1, -1, -1, high + 7, -1};
  Int keys[] = {-1, -1, 0, 0, -1, -1};
  Int internal[] = {-1, -1, high + 7, -1, -1, -1};
  Real distances[] = {0, 0, Real(2.2), 0, 0, 0};
  tmol::portable::Vec<Int, 2> connections[6];
  for (auto& c : connections) c[0] = c[1] = -1;
  connections[0][0] = 1;
  connections[0][1] = 0;
  connections[2][0] = 0;
  connections[2][1] = 0;
  connections[3][0] = 2;
  connections[3][1] = 0;
  MetalSiteParams<Real> params[6] = {};
  params[0].d0 = Real(2.1);
  params[4].d0 = Real(2.3);
  auto external = bridge_from_code<Int, Real, tmol::Device::CPU>(
      1,
      0,
      0,
      0,
      2,
      2,
      1,
      view(types, {1, 3}),
      view(connections, {1, 3, 2}),
      view(metals, {3, 2}),
      view(keys, {3, 2}),
      view(params, {3, 2, 1}),
      view(internal, {3, 2, 1}),
      view(distances, {3, 2, 1}));
  assert(external.valid && external.metal1 == high && external.metal2 == high + 7);
  auto local = bridge_from_code<Int, Real, tmol::Device::CPU>(
      4,
      0,
      1,
      1,
      0,
      2,
      1,
      view(types, {1, 3}),
      view(connections, {1, 3, 2}),
      view(metals, {3, 2}),
      view(keys, {3, 2}),
      view(params, {3, 2, 1}),
      view(internal, {3, 2, 1}),
      view(distances, {3, 2, 1}));
  assert(local.valid && local.metal1 == high + 7 && local.metal2 == high);
}
int main() {
  check_bridge_index_width<int32_t, float>();
  check_bridge_index_width<int32_t, double>();
  check_bridge_index_width<int64_t, float>();
  check_bridge_index_width<int64_t, double>();
  check<int32_t, float>();
  check<int32_t, float>(true);
  check<int32_t, float>(false, true);
  check<int32_t, float>(true, true);
  check<int32_t, double>();
  check<int32_t, double>(true);
  check<int32_t, double>(false, true);
  check<int32_t, double>(true, true);
  check<int64_t, float>();
  check<int64_t, float>(true);
  check<int64_t, float>(false, true);
  check<int64_t, float>(true, true);
  check<int64_t, double>();
  check<int64_t, double>(true);
  check<int64_t, double>(false, true);
  check<int64_t, double>(true, true);
}
