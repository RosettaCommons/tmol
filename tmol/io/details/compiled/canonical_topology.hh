#pragma once
#include <tmol/io/details/compiled/canonical_candidates.hh>

namespace tmol::io::details::compiled {
template <typename Int, Device D>
struct CanonicalChainContext {
  TView<int32_t, 2, D> chains;
  TView<Int, 2, D> types;
  TView<bool, 3, D> disconnected;
  TView<bool, 2, D> real;
  TView<bool, 1, D> class_polymeric;
  TView<Int, 2, D> class_ports;
  TView<Int, 2, D> cycles;
  TView<bool, 3, D> explicit_ports;
  TView<Int, 2, D> termini;
  TView<bool, 2, D> first, last, polymeric;
};
template <typename Int, Device D>
TMOL_DEVICE_FUNC bool canonical_polymeric(
    CanonicalChainContext<Int, D> c, int p, int r) {
  auto type = c.types[p][r];
  return c.real[p][r] && type >= 0 && type < c.class_polymeric.size(0)
         && c.class_polymeric[type];
}
template <typename Int, Device D>
TMOL_DEVICE_FUNC bool canonical_port(
    CanonicalChainContext<Int, D> c, int p, int r, int port) {
  auto type = c.types[p][r];
  // Preserve legacy padding metadata (-1 indexes the final canonical class).
  if (type == -1) type = c.class_ports.size(0) - 1;
  return type >= 0 && type < c.class_ports.size(0)
         && c.class_ports[type][port] >= 0;
}
template <template <Device> class Ops, Device D, typename Int, class Context>
void launch_canonical_chain_endings(
    Context& mgr, CanonicalChainContext<Int, D> c) {
  auto width = c.chains.size(1);
  for (int dim = 0; dim < 2; ++dim) {
    auto size = c.chains.size(dim);
    assert(
        c.types.size(dim) == size && c.disconnected.size(dim) == size
        && c.real.size(dim) == size);
    assert(c.explicit_ports.size(dim) == size && c.termini.size(dim) == size);
    assert(
        c.first.size(dim) == size && c.last.size(dim) == size
        && c.polymeric.size(dim) == size);
  }
  assert(c.disconnected.size(2) == 2 && c.explicit_ports.size(2) == 2);
  assert(c.cycles.size(1) == 3 && c.class_ports.size(1) == 2);
  assert(c.class_polymeric.size(0) == c.class_ports.size(0));
  int n = score::common::checked_dispatch_product(
      c.chains.size(0), width, "canonical chain endings");
  LAUNCH_BOX_128;
  Ops<D>::template forall<launch_t>(mgr, n, [=] TMOL_DEVICE_FUNC(int index) {
    int p = index / width, r = index % width;
    bool poly = canonical_polymeric(c, p, r);
    bool previous = r > 0 && c.chains[p][r - 1] == c.chains[p][r];
    bool next = r + 1 < width && c.chains[p][r + 1] == c.chains[p][r];
    bool first = (c.real[p][r] && !previous)
                 || (previous && !canonical_polymeric(c, p, r - 1));
    bool last =
        (c.real[p][r] && !next) || (next && !canonical_polymeric(c, p, r + 1));
    bool down_port = canonical_port(c, p, r, 0),
         up_port = canonical_port(c, p, r, 1);
    if (r > 0)
      first |= !canonical_port(c, p, r - 1, 1) || !down_port
               || c.explicit_ports[p][r - 1][1] || c.explicit_ports[p][r][0];
    if (r + 1 < width)
      last |= !up_port || !canonical_port(c, p, r + 1, 0)
              || c.explicit_ports[p][r][1] || c.explicit_ports[p][r + 1][0];
    bool down = ((first && !c.disconnected[p][r][0]) || (poly && !down_port))
                && !c.explicit_ports[p][r][0];
    bool up = ((last && !c.disconnected[p][r][1]) || (poly && !up_port))
              && !c.explicit_ports[p][r][1];
    // Read-only closure traversal avoids duplicate-endpoint write races.
    for (int64_t i = 0; i < c.cycles.size(0); ++i)
      if (c.cycles[i][0] == p) {
        if (c.cycles[i][1] == r) up = false;
        if (c.cycles[i][2] == r) down = false;
      }
    c.termini[p][r] = !poly ? 1 : down ? (up ? 3 : 0) : (up ? 2 : 1);
    c.first[p][r] = first;
    c.last[p][r] = last;
    c.polymeric[p][r] = poly;
  });
}

template <typename Int, Device D>
struct CanonicalTerminusContext {
  TView<bool, 3, D> present;
  TView<Int, 2, D> types, variants, termini;
  TView<bool, 3, D> broken;
  TView<bool, 2, D> real;
  TView<Int, 4, D> candidates;
  TView<bool, 4, D> valid;
  TView<bool, 2, D> absent;
  TView<Int, 4, D>
      counts;  // caller-owned [pose,residue,original/alternative,candidate]
  TView<Int, 2, D> output;
};
// A gap may select a terminus only if fewer provided atoms are extraneous and
// every best alternative retains every atom any best original could retain.
// This prevents exchanging a supplied heavy atom for more hydrogen names.
template <template <Device> class Ops, Device D, typename Int, class Context>
void launch_canonical_atom_termini(
    Context& mgr, CanonicalTerminusContext<Int, D> c) {
  for (int d = 0; d < 2; ++d) {
    auto size = c.types.size(d);
    assert(
        c.present.size(d) == size && c.variants.size(d) == size
        && c.termini.size(d) == size);
    assert(
        c.broken.size(d) == size && c.real.size(d) == size
        && c.counts.size(d) == size && c.output.size(d) == size);
  }
  for (int d = 0; d < 4; ++d) assert(c.candidates.size(d) == c.valid.size(d));
  assert(c.counts.size(2) == 2 && c.counts.size(3) == c.candidates.size(3));
  assert(c.broken.size(2) == 2 && c.present.size(2) == c.absent.size(1));
  auto width = c.types.size(1), n_candidates = c.candidates.size(3),
       n_atoms = c.present.size(2);
  int n = score::common::checked_dispatch_product(
      c.types.size(0), width, "canonical atom termini");
  LAUNCH_BOX_128;
  Ops<D>::template forall<launch_t>(mgr, n, [=] TMOL_DEVICE_FUNC(int index) {
    int p = index / width, r = index % width;
    Int original = c.termini[p][r];
    bool down = original == 0 || original == 3 || c.broken[p][r][0];
    bool up = original == 2 || original == 3 || c.broken[p][r][1];
    Int alternative = down ? (up ? 3 : 0) : (up ? 2 : 1);
    auto type = c.types[p][r], variant = c.variants[p][r];
    if (type < 0) type = 0;
    if (variant < 0) variant = 0;
    c.output[p][r] = original;
    Int minimum[2] = {Int(n_atoms + 1), Int(n_atoms + 1)};
    Int terms[2] = {original, alternative};
    bool valid_row =
        type < c.candidates.size(0) && variant < c.candidates.size(2);
    for (int side = 0; side < 2; ++side)
      for (int64_t k = 0; k < n_candidates; ++k) {
        Int count = n_atoms + 1;
        auto term = terms[side];
        if (valid_row && term >= 0 && term < c.candidates.size(1)
            && c.valid[type][term][variant][k]) {
          auto bt = c.candidates[type][term][variant][k];
          if (bt >= 0 && bt < c.absent.size(0)) {
            count = 0;
            for (int64_t a = 0; a < n_atoms; ++a)
              count += c.present[p][r][a] && c.absent[bt][a];
          }
        }
        c.counts[p][r][side][k] = count;
        if (count < minimum[side]) minimum[side] = count;
      }
    if (!c.real[p][r] || alternative == original || minimum[1] >= minimum[0])
      return;
    for (int64_t a = 0; a < n_atoms; ++a)
      if (c.present[p][r][a]) {
        bool could_keep = false, could_lose = false;
        for (int side = 0; side < 2; ++side)
          for (int64_t k = 0; k < n_candidates; ++k) {
            auto term = terms[side];
            if (c.counts[p][r][side][k] != minimum[side] || term < 0
                || term >= c.candidates.size(1)
                || !c.valid[type][term][variant][k])
              continue;
            auto bt = c.candidates[type][term][variant][k];
            if (bt < 0 || bt >= c.absent.size(0)) continue;
            if (side == 0)
              could_keep |= !c.absent[bt][a];
            else
              could_lose |= c.absent[bt][a];
          }
        if (could_keep && could_lose) return;
      }
    c.output[p][r] = alternative;
  });
}
template <template <Device> class Ops, Device D, typename Int>
struct CanonicalTopology {
  static void chain_endings(ContextManager&, CanonicalChainContext<Int, D>);
  static void atom_termini(ContextManager&, CanonicalTerminusContext<Int, D>);
};
}  // namespace tmol::io::details::compiled
