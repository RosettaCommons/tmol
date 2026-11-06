#pragma once
#include <tmol/io/details/compiled/canonical_topology.hh>
namespace tmol::io::details::compiled {
template <template <Device> class Ops, Device D, typename Int>
void CanonicalTopology<Ops, D, Int>::chain_endings(
    ContextManager& mgr, CanonicalChainContext<Int, D> c) {
  launch_canonical_chain_endings<Ops, D, Int>(mgr, c);
}
template <template <Device> class Ops, Device D, typename Int>
void CanonicalTopology<Ops, D, Int>::atom_termini(
    ContextManager& mgr, CanonicalTerminusContext<Int, D> c) {
  launch_canonical_atom_termini<Ops, D, Int>(mgr, c);
}
}  // namespace tmol::io::details::compiled
