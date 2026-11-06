#include <tmol/pack/compiled/cuda_annealer_workspace.hh>
#include <cassert>
#include <cstdlib>
#include <vector>
#include <utility>
using namespace tmol::pack::compiled;
template <class T, size_t N>
void record(
    std::vector<std::pair<uintptr_t, uintptr_t>>& spans,
    tmol::TView<T, N, tmol::Device::CUDA> view) {
  int64_t count = 1;
  for (int i = 0; i < N; ++i) count *= view.size(i);
  if (count)
    spans.push_back(
        {reinterpret_cast<uintptr_t>(view.data()),
         reinterpret_cast<uintptr_t>(view.data()) + count * sizeof(T)});
}
int main() {
  CudaAnnealerWorkspace empty(0, 0, 0, 0);
  assert(empty.bytes == 0);
  CudaAnnealerWorkspace shape(3, 7, 53, 12);
  void* storage = std::aligned_alloc(256, (shape.bytes + 255) / 256 * 256);
  assert(storage);
  CudaAnnealerWorkspace w(3, 7, 53, 12, storage, shape.bytes);
  assert(w.bytes == shape.bytes && w.zero_bytes < w.bytes);
  assert(w.current_rotamer_assignments_lotemp.size(1) == 1250);
  assert(w.current_rotamer_assignments_fullquench.size(1) == 312);
  assert(w.quench_lite_order.data() == w.fullquench_order.data());
  assert(w.quench_lite_order.size(2) == 8);
  assert(w.fullquench_order.size(2) == 53);
  std::vector<std::pair<uintptr_t, uintptr_t>> spans;
  record(spans, w.scores_hitemp);
  record(spans, w.scores_lotemp);
  record(spans, w.current_rotamer_assignments_hitemp);
  record(spans, w.best_rotamer_assignments_hitemp);
  record(spans, w.current_rotamer_assignments_lotemp);
  record(spans, w.best_rotamer_assignments_lotemp);
  record(spans, w.current_rotamer_assignments_fullquench);
  record(spans, w.sorted_hitemp_traj);
  record(spans, w.sorted_lotemp_traj);
  record(spans, w.sorted_fullquench_traj);
  record(spans, w.segment_heads_hitemp);
  record(spans, w.segment_heads_lotemp);
  record(spans, w.segment_heads_fullquench);
  // Shared quench storage is intentionally aliased across disjoint phases.
  auto q =
      std::max(int64_t(3) * 1250 * 8, int64_t(3) * 312 * 53) * sizeof(int32_t);
  spans.push_back(
      {reinterpret_cast<uintptr_t>(w.quench_lite_order.data()),
       reinterpret_cast<uintptr_t>(w.quench_lite_order.data()) + q});
  record(spans, w.reverse_offsets);
  spans.push_back(
      {reinterpret_cast<uintptr_t>(w.sort_data),
       reinterpret_cast<uintptr_t>(w.sort_data) + w.sort_bytes});
  std::sort(spans.begin(), spans.end());
  uintptr_t end = reinterpret_cast<uintptr_t>(storage);
  for (auto span : spans) {
    assert(span.first >= end);
    end = span.second;
  }
  assert(end <= reinterpret_cast<uintptr_t>(storage) + w.bytes);
  bool rejected = false;
  try {
    CudaAnnealerWorkspace bad(3, 7, 53, 12, storage, w.bytes - 1);
  } catch (std::invalid_argument const&) {
    rejected = true;
  }
  assert(rejected);
  rejected = false;
  try {
    CudaAnnealerWorkspace bad(INT32_MAX, 1, 1, 0);
  } catch (std::invalid_argument const&) {
    rejected = true;
  }
  assert(rejected);
  // Check the bound against the vendored sort's allocation contract at both
  // default tile sizes, including changes in merge-pass count.
  for (int64_t count :
       {1LL, 1408LL, 1409LL, 1920LL, 1921LL, 100000LL, 1234567LL}) {
    for (int64_t nv : {1408LL, 1920LL}) {
      int64_t ctas = (count + nv - 1) / nv, passes = 0;
      while ((1LL << passes) < ctas) ++passes;
      int64_t capacity = ctas;
      for (int i = 0; i < passes; ++i)
        capacity += (ctas + (1LL << i) - 1) / (1LL << i);
      int64_t exact = 0;
      auto take = [&](int64_t n) {
        if (n) {
          exact = (exact + 15) / 16 * 16;
          exact += n;
        }
      };
      if (passes) {
        take(count * 4);
        take(count * 4);
      }
      take(capacity * 8);
      take(ctas * 16);
      take(ctas * 4);
      take(ctas * 4);
      take(ctas * 4);
      take(passes * 8);
      take((ctas + 1) * 4);
      assert(cuda_annealer_sort_bytes(count) >= exact);
    }
  }
  std::free(storage);
}
