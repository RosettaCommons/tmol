#pragma once
#include <tmol/io/details/compiled/canonical_candidates.hh>
namespace tmol::io::details::compiled {
template <template <Device> class Ops, Device D, typename Int>
void CanonicalCandidates<Ops, D, Int>::f(
    ContextManager& mgr, CanonicalCandidateContext<Int, D> c) {
  launch_canonical_candidates<Ops, D, Int>(mgr, c);
}
}  // namespace tmol::io::details::compiled
