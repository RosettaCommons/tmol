#pragma once
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/score/common/counting.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>

namespace tmol::io::details::compiled {
template <typename Int, Device D>
struct CanonicalCandidateContext {
  TView<bool, 3, D> present;
  TView<bool, 2, D> real;
  TView<Int, 2, D> io_class, terminus, variant;
  TView<Int, 4, D> candidates;
  TView<bool, 4, D> valid;
  TView<bool, 2, D> absent, nonterminal;
  TView<bool, 1, D> nondefault;
  TView<Int, 2, D> selected;
  TView<int64_t, 2, D> score;
  TView<Int, 2, D> candidate_index;
};
// Preserve palette order for ties and ignore missing atoms added by terminal
// patches. Each provided extraneous atom outweighs all missing canonical atoms.
// Invalid/no-candidate rows return -1 and the legacy failure sentinel; padding
// returns -1 and score zero. No temporary candidate-by-atom arrays are needed.
template <template <Device> class Ops, Device D, typename Int, class Context>
void launch_canonical_candidates(
    Context& mgr, CanonicalCandidateContext<Int, D> c) {
  assert(c.present.size(0) == c.real.size(0));
  assert(c.present.size(1) == c.real.size(1));
  for (int d = 0; d < 2; ++d) {
    assert(c.io_class.size(d) == c.real.size(d));
    assert(c.terminus.size(d) == c.real.size(d));
    assert(c.variant.size(d) == c.real.size(d));
    assert(c.selected.size(d) == c.real.size(d));
    assert(c.score.size(d) == c.real.size(d));
    assert(c.candidate_index.size(d) == c.real.size(d));
    assert(c.nonterminal.size(d) == c.absent.size(d));
  }
  for (int d = 0; d < 4; ++d) assert(c.candidates.size(d) == c.valid.size(d));
  assert(c.absent.size(1) == c.present.size(2));
  assert(c.absent.size(0) == c.nondefault.size(0));
  auto width = c.real.size(1);
  auto n = score::common::checked_dispatch_product(
      c.real.size(0), width, "canonical candidates");
  LAUNCH_BOX_128;
  Ops<D>::template forall<launch_t>(mgr, n, [=] TMOL_DEVICE_FUNC(int index) {
    auto p = index / width, r = index % width;
    Int selected = -1, best_index = 0;
    int64_t best_score = c.real[p][r] ? int64_t(1) << 30 : 0;
    auto io = c.io_class[p][r], term = c.terminus[p][r], var = c.variant[p][r];
    if (c.real[p][r] && io >= 0 && io < c.candidates.size(0) && term >= 0
        && term < c.candidates.size(1) && var >= 0
        && var < c.candidates.size(2)) {
      for (Int k = 0; k < c.candidates.size(3); ++k) {
        auto bt = c.candidates[io][term][var][k];
        if (!c.valid[io][term][var][k] || bt < 0 || bt >= c.absent.size(0))
          continue;
        int64_t score = c.nondefault[bt];
        for (int64_t a = 0; a < c.present.size(2); ++a) {
          score += c.present[p][r][a]
                       ? (c.absent[bt][a] ? 2 * (c.present.size(2) + 1) : 0)
                       : (c.nonterminal[bt][a] ? 2 : 0);
        }
        if (score < best_score) {
          best_score = score;
          best_index = k;
          selected = bt;
        }
      }
    }
    c.selected[p][r] = selected;
    c.score[p][r] = best_score;
    c.candidate_index[p][r] = best_index;
  });
}
template <template <Device> class Ops, Device D, typename Int>
struct CanonicalCandidates {
  static void f(ContextManager& mgr, CanonicalCandidateContext<Int, D> c);
};
}  // namespace tmol::io::details::compiled
