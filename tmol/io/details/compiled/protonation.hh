#pragma once
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/score/common/counting.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
namespace tmol::io::details::compiled {
template <class Int, Device D>
struct ProtonationContext {
  TView<Int, 2, D> types, variants;
  TView<bool, 3, D> present;
  TView<bool, 2, D> hydrogen, titratable, parents, shared, expecting;
  TView<bool, 3, D> donates;
  TView<int64_t, 2, D> covalent_bonds, required_donors;
  Int deprotonated;
  TView<Int, 2, D> output;
  TView<bool, 2, D> unresolved;
};
template <class Int, Device D>
TMOL_DEVICE_FUNC bool protonation_donates(
    ProtonationContext<Int, D> c, int p, int r, Int t, Int v) {
  if (v < 0 || v >= c.donates.size(1)) return false;
  for (int k = 0; k < c.required_donors.size(0); ++k) {
    if (c.required_donors[k][0] != p || c.required_donors[k][1] != r) continue;
    auto atom = c.required_donors[k][2];
    if (atom < 0 || atom >= c.donates.size(2) || !c.donates[t][v][atom])
      return false;
  }
  return true;
}
template <template <Device> class Ops, Device D, class Int, class Context>
void launch_protonation(Context& mgr, ProtonationContext<Int, D> c) {
  auto width = c.types.size(1), atoms = c.present.size(2);
  for (int d = 0; d < 2; ++d) {
    assert(
        c.types.size(d) == c.variants.size(d)
        && c.types.size(d) == c.present.size(d));
    assert(
        c.types.size(d) == c.output.size(d)
        && c.types.size(d) == c.unresolved.size(d));
    assert(
        c.hydrogen.size(d) == c.titratable.size(d)
        && c.hydrogen.size(d) == c.parents.size(d)
        && c.hydrogen.size(d) == c.shared.size(d));
  }
  assert(c.hydrogen.size(1) == atoms && c.donates.size(2) == atoms);
  assert(
      c.expecting.size(0) == c.hydrogen.size(0)
      && c.donates.size(0) == c.hydrogen.size(0));
  assert(c.covalent_bonds.size(1) == 5 && c.required_donors.size(1) == 3);
  auto n = score::common::checked_dispatch_product(
      c.types.size(0), width, "protonation selection");
  LAUNCH_BOX_128;
  Ops<D>::template forall<launch_t>(mgr, n, [=] TMOL_DEVICE_FUNC(int i) {
    int p = i / width, r = i % width;
    auto t = c.types[p][r], v = c.variants[p][r];
    c.unresolved[p][r] = false;
    if (t < 0 || t >= c.hydrogen.size(0)) {
      c.output[p][r] = v;
      return;
    }
    bool has_h = false, has_titratable = false, complete = true;
    for (int a = 0; a < atoms; ++a) {
      auto present = c.present[p][r][a];
      has_h |= present && c.hydrogen[t][a];
      has_titratable |= present && c.titratable[t][a];
      complete &= present || !c.shared[t][a];
    }
    if (v >= 0 && v < c.expecting.size(1) && c.expecting[t][v] && has_h
        && complete && !has_titratable) {
      bool bonded = false;
      for (int k = 0; k < c.covalent_bonds.size(0); ++k) {
        if (c.covalent_bonds[k][0] != p) continue;
        for (int side = 1; side <= 3; side += 2) {
          auto a = c.covalent_bonds[k][side + 1];
          if (c.covalent_bonds[k][side] == r && a >= 0 && a < atoms
              && c.parents[t][a])
            bonded = true;
        }
      }
      if (!bonded) v = c.deprotonated;
    }
    if (!has_h) {
      bool required = false;
      for (int k = 0; k < c.required_donors.size(0); ++k)
        required |=
            c.required_donors[k][0] == p && c.required_donors[k][1] == r;
      if (required && !protonation_donates(c, p, r, t, v)) {
        bool found = false;
        for (Int candidate = 0; candidate < c.donates.size(1); ++candidate)
          if (protonation_donates(c, p, r, t, candidate)) {
            v = candidate;
            found = true;
            break;
          }
        c.unresolved[p][r] = !found;
      }
    }
    c.output[p][r] = v;
  });
}
template <template <Device> class Ops, Device D, class Int>
struct Protonation {
  static void f(ContextManager&, ProtonationContext<Int, D>);
};
}  // namespace tmol::io::details::compiled
