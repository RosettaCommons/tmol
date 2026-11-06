#pragma once
#include <tmol/io/details/compiled/protonation.hh>
namespace tmol::io::details::compiled {
template <template <Device> class Ops, Device D, class Int>
void Protonation<Ops, D, Int>::f(
    ContextManager& mgr, ProtonationContext<Int, D> c) {
  launch_protonation<Ops, D>(mgr, c);
}
}  // namespace tmol::io::details::compiled
