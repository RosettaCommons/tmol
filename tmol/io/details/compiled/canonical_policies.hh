#pragma once
#include <tmol/utility/portable/matrix.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/score/common/counting.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>

namespace tmol::io::details::compiled {
using tmol::portable::Vec;
template <typename Int, Device D>
struct CanonicalVariantContext {
  TView<Int, 2, D> types, variants, termini;
  TView<bool, 4, D> valid;
  int64_t neutral_base;
  TView<Int, 2, D> output;
};
template <template <Device> class Ops, Device D, typename Int, class Context>
void launch_canonical_variants(
    Context& mgr, CanonicalVariantContext<Int, D> c) {
  for (int d = 0; d < 2; ++d) {
    assert(c.variants.size(d) == c.types.size(d));
    assert(c.termini.size(d) == c.types.size(d));
    assert(c.output.size(d) == c.types.size(d));
  }
  assert(c.neutral_base > 0);
  auto width = c.types.size(1);
  auto n = score::common::checked_dispatch_product(
      c.types.size(0), width, "canonical variants");
  LAUNCH_BOX_128;
  Ops<D>::template forall<launch_t>(mgr, n, [=] TMOL_DEVICE_FUNC(int index) {
    auto p = index / width, r = index % width;
    auto t = c.types[p][r], v = c.variants[p][r], term = c.termini[p][r];
    bool fits = false;
    if (v >= c.neutral_base && v < c.valid.size(2) && t >= 0
        && t < c.valid.size(0) && term >= 0 && term < c.valid.size(1)) {
      for (Int k = 0; k < c.valid.size(3); ++k) fits |= c.valid[t][term][v][k];
      if (!fits) v -= c.neutral_base;
    }
    c.output[p][r] = v;
  });
}

template <typename Real, typename Int, Device D>
struct CanonicalCycleContext {
  TView<Int, 2, D> types;
  TView<int32_t, 2, D> chains;
  TView<Int, 1, D> up, down;
  TView<Vec<Real, 3>, 3, D> coordinates;
  TView<Int, 2, D> explicit_pairs;
  Real cutoff;
  // At each inferred last residue, the first residue of the closed run, else
  // -1.
  TView<Int, 2, D> partner;
};
template <typename Real, typename Int, Device D>
TMOL_DEVICE_FUNC bool canonical_cycle_polymer(
    CanonicalCycleContext<Real, Int, D> c, int p, int r) {
  auto t = c.types[p][r];
  return t >= 0 && t < c.up.size(0) && c.chains[p][r] >= 0
         && (c.up[t] >= 0 || c.down[t] >= 0);
}
template <typename Real, typename Int, Device D>
TMOL_DEVICE_FUNC void canonical_close_run(
    CanonicalCycleContext<Real, Int, D> c, int p, int first, int last) {
  if (first < 0 || last <= first) return;
  auto ua = c.up[c.types[p][last]], da = c.down[c.types[p][first]];
  if (ua < 0 || da < 0 || ua >= c.coordinates.size(2)
      || da >= c.coordinates.size(2))
    return;
  auto delta = c.coordinates[p][last][ua] - c.coordinates[p][first][da];
  auto d2 = delta.squaredNorm();
  // The comparison also rejects nonfinite distances.
  if (!(d2 < c.cutoff * c.cutoff)) return;
  for (int64_t i = 0; i < c.explicit_pairs.size(0); ++i)
    if (c.explicit_pairs[i][0] == p
        && (c.explicit_pairs[i][1] == last || c.explicit_pairs[i][2] == first))
      return;
  c.partner[p][last] = first;
}
template <
    template <Device> class Ops,
    Device D,
    typename Real,
    typename Int,
    class Context>
void launch_canonical_cycles(
    Context& mgr, CanonicalCycleContext<Real, Int, D> c) {
  for (int d = 0; d < 2; ++d) {
    assert(c.chains.size(d) == c.types.size(d));
    assert(c.coordinates.size(d) == c.types.size(d));
    assert(c.partner.size(d) == c.types.size(d));
  }
  assert(c.up.size(0) == c.down.size(0));
  assert(c.explicit_pairs.size(1) == 3);
  auto width = c.types.size(1);
  LAUNCH_BOX_128;
  Ops<D>::template forall<launch_t>(
      mgr, c.types.size(0), [=] TMOL_DEVICE_FUNC(int p) {
        int first = -1;
        for (int r = 0; r < width; ++r) {
          c.partner[p][r] = -1;
          bool polymer = canonical_cycle_polymer(c, p, r);
          bool continues =
              first >= 0 && polymer && c.chains[p][r] == c.chains[p][r - 1];
          if (!continues) {
            canonical_close_run(c, p, first, r - 1);
            first = polymer ? r : -1;
          }
        }
        canonical_close_run(c, p, first, width - 1);
      });
}

template <typename Int, Device D>
struct CanonicalWiringContext {
  TView<Int, 2, D> types;
  TView<bool, 2, D> polymeric, first, last;
  TView<bool, 3, D> disconnected;
  TView<Int, 1, D> up, down, disulfide_port;
  TView<Int, 2, D> disulfides, cycles, explicit_connections;
  TView<Vec<Int, 2>, 3, D> connections;
  // 0 valid, 1 unmatched sequential endpoints, 2 invalid explicit endpoint,
  // 3 malformed final reciprocity/self-connection. One status per pose.
  TView<Int, 1, D> status;
};
template <typename Int, Device D>
TMOL_DEVICE_FUNC bool canonical_sequential_endpoint(
    CanonicalWiringContext<Int, D> c, int p, int r, bool next) {
  auto neighbor = r + (next ? 1 : -1);
  return neighbor >= 0 && neighbor < c.types.size(1) && c.types[p][r] >= 0
         && c.types[p][neighbor] >= 0 && c.polymeric[p][r]
         && !(next ? c.last[p][r] : c.first[p][r])
         && !c.disconnected[p][r][next ? 1 : 0];
}
template <typename Int, Device D>
TMOL_DEVICE_FUNC void canonical_join(
    CanonicalWiringContext<Int, D> c, int p, Int a, Int ac, Int b, Int bc) {
  if (a < 0 || b < 0 || a >= c.types.size(1) || b >= c.types.size(1) || ac < 0
      || bc < 0 || ac >= c.connections.size(2) || bc >= c.connections.size(2)
      || c.types[p][a] < 0 || c.types[p][b] < 0) {
    c.status[p] = 2;
    return;
  }
  c.connections[p][a][ac] = {b, bc};
  c.connections[p][b][bc] = {a, ac};
}
template <typename Int, Device D>
TMOL_DEVICE_FUNC bool canonical_valid_connection(
    TView<Int, 2, D> types,
    TView<Vec<Int, 2>, 3, D> connections,
    int p,
    int r,
    int port) {
  auto other = connections[p][r][port];
  if (other[0] < 0) return true;
  if (other[0] >= types.size(1) || other[1] < 0
      || other[1] >= connections.size(2) || other[0] == r || types[p][r] < 0
      || types[p][other[0]] < 0)
    return false;
  auto reverse = connections[p][other[0]][other[1]];
  return reverse[0] == r && reverse[1] == port;
}
template <template <Device> class Ops, Device D, typename Int, class Context>
void launch_canonical_wiring(Context& mgr, CanonicalWiringContext<Int, D> c) {
  auto width = c.types.size(1);
  for (int d = 0; d < 2; ++d) {
    assert(c.polymeric.size(d) == c.types.size(d));
    assert(c.first.size(d) == c.types.size(d));
    assert(c.last.size(d) == c.types.size(d));
    assert(c.disconnected.size(d) == c.types.size(d));
    assert(c.connections.size(d) == c.types.size(d));
  }
  assert(c.disconnected.size(2) == 2);
  assert(
      c.up.size(0) == c.down.size(0)
      && c.up.size(0) == c.disulfide_port.size(0));
  assert(
      c.disulfides.size(1) == 3 && c.cycles.size(1) == 3
      && c.explicit_connections.size(1) == 5);
  assert(c.status.size(0) == c.types.size(0));
  LAUNCH_BOX_128;
  Ops<D>::template forall<launch_t>(
      mgr, c.types.size(0), [=] TMOL_DEVICE_FUNC(int p) {
        c.status[p] = 0;
        for (int r = 0; r < width; ++r)
          for (int port = 0; port < c.connections.size(2); ++port)
            c.connections[p][r][port] = {-1, -1};
        for (int r = 0; r < width; ++r) {
          if (c.types[p][r] < -1 || c.types[p][r] >= c.up.size(0)) {
            c.status[p] = 2;
            return;
          }
        }
        int previous = 0;
        for (int r = 0; r < width; ++r) {
          if (!canonical_sequential_endpoint(c, p, r, true)) continue;
          while (previous < width
                 && !canonical_sequential_endpoint(c, p, previous, false))
            ++previous;
          if (previous == width) {
            c.status[p] = 1;
            break;
          }
          canonical_join(
              c,
              p,
              Int(r),
              c.up[c.types[p][r]],
              Int(previous),
              c.down[c.types[p][previous]]);
          ++previous;
        }
        while (previous < width
               && !canonical_sequential_endpoint(c, p, previous, false))
          ++previous;
        if (previous < width) c.status[p] = 1;
        for (int kind = 0; kind < 2; ++kind) {
          auto rows = kind ? c.cycles : c.disulfides;
          for (int64_t i = 0; i < rows.size(0); ++i) {
            if (rows[i][0] < 0 || rows[i][0] >= c.types.size(0)) {
              c.status[p] = 2;
              continue;
            }
            if (rows[i][0] != p) continue;
            auto a = rows[i][1], b = rows[i][2];
            if (a < 0 || b < 0 || a >= width || b >= width || c.types[p][a] < 0
                || c.types[p][b] < 0) {
              c.status[p] = 2;
              continue;
            }
            auto ac =
                kind ? c.up[c.types[p][a]] : c.disulfide_port[c.types[p][a]];
            auto bc =
                kind ? c.down[c.types[p][b]] : c.disulfide_port[c.types[p][b]];
            canonical_join(c, p, a, ac, b, bc);
          }
        }
        for (int64_t i = 0; i < c.explicit_connections.size(0); ++i) {
          auto row = c.explicit_connections[i];
          if (row[0] < 0 || row[0] >= c.types.size(0)) {
            c.status[p] = 2;
            continue;
          }
          if (row[0] == p) canonical_join(c, p, row[1], row[2], row[3], row[4]);
        }
        for (int r = 0; r < width; ++r)
          for (int port = 0; port < c.connections.size(2); ++port)
            if (!canonical_valid_connection(c.types, c.connections, p, r, port))
              c.status[p] = 3;
      });
}
template <template <Device> class Ops, Device D, typename Int, class Context>
void launch_canonical_connection_validation(
    Context& mgr,
    TView<Int, 2, D> types,
    TView<Vec<Int, 2>, 3, D> connections,
    TView<bool, 3, D> bad) {
  for (int d = 0; d < 2; ++d) assert(types.size(d) == connections.size(d));
  for (int d = 0; d < 3; ++d) assert(bad.size(d) == connections.size(d));
  auto width = connections.size(1), ports = connections.size(2);
  auto n = score::common::checked_dispatch_product(
      score::common::checked_dispatch_product(
          connections.size(0), width, "canonical connection rows"),
      ports,
      "canonical connection validation");
  LAUNCH_BOX_128;
  Ops<D>::template forall<launch_t>(mgr, n, [=] TMOL_DEVICE_FUNC(int i) {
    auto port = i % ports, r = (i / ports) % width, p = i / (ports * width);
    bad[p][r][port] =
        !canonical_valid_connection(types, connections, p, r, port);
  });
}
template <template <Device> class Ops, Device D, typename Int>
struct CanonicalPolicies {
  static void variants(ContextManager&, CanonicalVariantContext<Int, D>);
  static void wiring(ContextManager&, CanonicalWiringContext<Int, D>);
  static void validate(
      ContextManager&,
      TView<Int, 2, D>,
      TView<Vec<Int, 2>, 3, D>,
      TView<bool, 3, D>);
};
template <template <Device> class Ops, Device D, typename Real, typename Int>
struct CanonicalCycles {
  static void f(ContextManager&, CanonicalCycleContext<Real, Int, D>);
};
}  // namespace tmol::io::details::compiled
