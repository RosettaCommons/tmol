#pragma once
#include <tmol/io/details/compiled/canonical_policies.hh>
namespace tmol::io::details::compiled {
template <template <Device> class Ops, Device D, typename Int>
void CanonicalPolicies<Ops, D, Int>::variants(
    ContextManager& mgr, CanonicalVariantContext<Int, D> c) {
  launch_canonical_variants<Ops, D, Int>(mgr, c);
}
template <template <Device> class Ops, Device D, typename Int>
void CanonicalPolicies<Ops, D, Int>::wiring(
    ContextManager& mgr, CanonicalWiringContext<Int, D> c) {
  launch_canonical_wiring<Ops, D, Int>(mgr, c);
}
template <template <Device> class Ops, Device D, typename Int>
void CanonicalPolicies<Ops, D, Int>::validate(
    ContextManager& mgr,
    TView<Int, 2, D> types,
    TView<Vec<Int, 2>, 3, D> connections,
    TView<bool, 3, D> bad) {
  launch_canonical_connection_validation<Ops, D, Int>(
      mgr, types, connections, bad);
}
template <template <Device> class Ops, Device D, typename Real, typename Int>
void CanonicalCycles<Ops, D, Real, Int>::f(
    ContextManager& mgr, CanonicalCycleContext<Real, Int, D> c) {
  launch_canonical_cycles<Ops, D, Real, Int>(mgr, c);
}
}  // namespace tmol::io::details::compiled
