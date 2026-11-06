#pragma once
#include <tmol/io/details/compiled/hydrogen_completion.hh>
namespace tmol::io::details::compiled {
template <template <Device> class Ops, Device D, class Real, class Int>
void HydrogenCompletion<Ops, D, Real, Int>::f(
    ContextManager& mgr,
    int mode,
    HydrogenCompletionContext<Real, Int, D> c,
    TView<Vec<Real, 3>, 2, D> w,
    TView<Vec<Real, 3>, 2, D> v,
    TView<Real, 2, D> d,
    TView<Vec<Real, 3>, 2, D> out,
    TView<Real, 2, D> gd,
    TView<Vec<Real, 3>, 2, D> gw) {
  if (mode == 0)
    launch_hydrogen_completion<0, Ops, D>(mgr, c, w, v, d, out, gd, gw);
  else if (mode == 1)
    launch_hydrogen_completion<1, Ops, D>(mgr, c, w, v, d, out, gd, gw);
  else
    launch_hydrogen_completion<2, Ops, D>(mgr, c, w, v, d, out, gd, gw);
}
template <template <Device> class Ops, Device D, class Real, class Int>
void HydrogenInput<Ops, D, Real, Int>::f(
    ContextManager& mgr, HydrogenInputContext<Real, Int, D> c) {
  launch_hydrogen_input<Ops, D>(mgr, c);
}
}  // namespace tmol::io::details::compiled
