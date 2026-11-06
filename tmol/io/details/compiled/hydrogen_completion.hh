#pragma once
#include <tmol/utility/portable/matrix.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/counting.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>

namespace tmol::io::details::compiled {
using tmol::portable::Vec;
template <typename Real, typename Int, Device D>
struct HydrogenCompletionContext {
  TView<Vec<Real, 3>, 2, D> coordinates;
  TView<Int, 2, D> types, offsets;
  TView<Vec<Int, 2>, 3, D> connections;
  TView<Int, 2, D> connection_atoms;
  TView<bool, 3, D> rebuilt;
  TView<bool, 2, D> eligible;
  TView<int64_t, 2, D> parent;
  TView<int64_t, 3, D> neighbors;
  TView<int64_t, 2, D> n_neighbors;
  TView<Real, 2, D> distance;
  TView<int64_t, 3, D> parent_connections;
  TView<int64_t, 2, D> n_parent_connections;
};
// A normalized vector's Jacobian and directional derivative. No epsilon is
// introduced: degenerate real geometry retains the mature nonfinite behavior.
template <class Real>
TMOL_DEVICE_FUNC Vec<Real, 3> normalized_vjp(Vec<Real, 3> x, Vec<Real, 3> w) {
  Real length = x.norm();
  auto unit = x / length;
  return (w - unit * unit.dot(w)) / length;
}
template <class Real>
TMOL_DEVICE_FUNC Vec<Real, 3> normalized_vjp_jvp(
    Vec<Real, 3> x, Vec<Real, 3> w, Vec<Real, 3> direction) {
  Real length = x.norm();
  auto unit = x / length;
  auto dunit = normalized_vjp(x, direction);
  return -(dunit * unit.dot(w) + unit * dunit.dot(w)
           + normalized_vjp(x, w) * unit.dot(direction))
         / length;
}
struct HydrogenNeighbors {
  int64_t parent;
  int64_t atoms[6];
  int n_local, n_connected;
};
template <class Real, class Int, Device D>
TMOL_DEVICE_FUNC HydrogenNeighbors hydrogen_neighbors(
    HydrogenCompletionContext<Real, Int, D> c, int p, int b, int a, int type) {
  auto offset = c.offsets[p][b];
  HydrogenNeighbors result{offset + c.parent[type][a], {}, 0, 0};
  for (int k = 0; k < c.n_neighbors[type][a]; ++k)
    result.atoms[result.n_local++] = offset + c.neighbors[type][a][k];
  for (int k = 0; k < c.n_parent_connections[type][a]; ++k) {
    auto port = c.parent_connections[type][a][k];
    if (port < 0 || port >= c.connections.size(2)) continue;
    auto other = c.connections[p][b][port];
    if (other[0] < 0 || other[1] < 0 || other[0] >= c.types.size(1)
        || other[1] >= c.connection_atoms.size(1))
      continue;
    auto other_type = c.types[p][other[0]];
    if (other_type < 0 || other_type >= c.connection_atoms.size(0)) continue;
    auto atom = c.connection_atoms[other_type][other[1]];
    if (atom < 0) continue;
    result.atoms[result.n_local + result.n_connected++] =
        c.offsets[p][other[0]] + atom;
  }
  return result;
}
// Mode 0: forward. Mode 1: VJP in coordinate_output/distance_output.
// Mode 2: derivative of that VJP along coordinate/distance_direction, plus
// forward directional derivative in weight_output (the VJP weight transpose).
// Output buffers belong to the caller and are initialized by these launches.
template <
    int Mode,
    template <Device> class Ops,
    Device D,
    class Real,
    class Int,
    class Context>
void launch_hydrogen_completion(
    Context& mgr,
    HydrogenCompletionContext<Real, Int, D> c,
    TView<Vec<Real, 3>, 2, D> weights,
    TView<Vec<Real, 3>, 2, D> coordinate_direction,
    TView<Real, 2, D> distance_direction,
    TView<Vec<Real, 3>, 2, D> coordinate_output,
    TView<Real, 2, D> distance_output,
    TView<Vec<Real, 3>, 2, D> weight_output) {
  int poses = c.coordinates.size(0), atoms = c.coordinates.size(1),
      blocks = c.types.size(1), width = c.eligible.size(1);
  assert(c.neighbors.size(2) == 3 && c.parent_connections.size(2) == 3);
  for (int d = 0; d < 2; ++d) {
    assert(c.eligible.size(d) == c.parent.size(d));
    assert(c.eligible.size(d) == c.distance.size(d));
    assert(c.eligible.size(d) == c.n_neighbors.size(d));
    assert(c.eligible.size(d) == c.n_parent_connections.size(d));
  }
  assert(
      c.rebuilt.size(0) == poses && c.rebuilt.size(1) == blocks
      && c.rebuilt.size(2) == width);
  assert(
      c.types.size(0) == poses && c.offsets.size(0) == poses
      && c.offsets.size(1) == blocks);
  LAUNCH_BOX_128;
  auto ncoords = score::common::checked_dispatch_product(
      poses, atoms, "hydrogen completion coordinates");
  Ops<D>::template forall<launch_t>(
      mgr,
      ncoords,
      [atoms,
       c,
       coordinate_output,
       weights,
       weight_output,
       coordinate_direction] TMOL_DEVICE_FUNC(int i) {
        auto p = i / atoms, a = i % atoms;
        if constexpr (Mode == 0)
          coordinate_output[p][a] = c.coordinates[p][a];
        else if constexpr (Mode == 1)
          coordinate_output[p][a] = weights[p][a];
        else {
          coordinate_output[p][a] = {0, 0, 0};
          weight_output[p][a] = coordinate_direction[p][a];
        }
      });
  if constexpr (Mode != 0) {
    auto n = score::common::checked_dispatch_product(
        c.distance.size(0), width, "hydrogen completion distances");
    Ops<D>::template forall<launch_t>(
        mgr, n, [width, distance_output] TMOL_DEVICE_FUNC(int i) {
          distance_output[i / width][i % width] = 0;
        });
  }
  auto n = score::common::checked_dispatch_product(
      score::common::checked_dispatch_product(
          poses, blocks, "hydrogen completion blocks"),
      width,
      "hydrogen completion atoms");
  auto process =
      [c,
       width,
       blocks,
       coordinate_output,
       weights,
       distance_output,
       coordinate_direction,
       distance_direction,
       weight_output] TMOL_DEVICE_FUNC(int p, int b, int a, Int type) {
        auto h = c.offsets[p][b] + a;
        if constexpr (Mode == 1) {
          auto w = weights[p][h];
          // Packing can discard an unbuildable intermediate. Its zero
          // cotangent contributes nothing, even when its frame is undefined.
          if (w[0] == Real(0) && w[1] == Real(0) && w[2] == Real(0)) return;
        }
        auto nb = hydrogen_neighbors(c, p, b, a, type);
        auto parent = c.coordinates[p][nb.parent];
        Vec<Real, 3> delta[6], unit[6], sum_local{0, 0, 0},
            sum_connected{0, 0, 0};
        for (int k = 0; k < nb.n_local + nb.n_connected; ++k) {
          delta[k] = c.coordinates[p][nb.atoms[k]] - parent;
          unit[k] = delta[k] / delta[k].norm();
          if (k < nb.n_local)
            sum_local += unit[k];
          else
            sum_connected += unit[k];
        }
        auto sum = sum_local + sum_connected;
        auto direction = sum / sum.norm();
        Real distance = c.distance[type][a];
        if constexpr (Mode == 0)
          coordinate_output[p][h] = parent - distance * direction;
        else {
          auto w = weights[p][h];
          auto adjoint = -distance * normalized_vjp(sum, w);
          if constexpr (Mode == 1) {
            coordinate_output[p][h] = {0, 0, 0};
            Vec<Real, 3> parent_gradient = w;
            for (int k = 0; k < nb.n_local + nb.n_connected; ++k) {
              auto g = normalized_vjp(delta[k], adjoint);
              parent_gradient -= g;
              score::common::accumulate<D, Vec<Real, 3>>::add(
                  coordinate_output[p][nb.atoms[k]], g);
            }
            score::common::accumulate<D, Vec<Real, 3>>::add(
                coordinate_output[p][nb.parent], parent_gradient);
            score::common::accumulate<D, Real>::add(
                distance_output[type][a], -w.dot(direction));
          } else {
            auto vp = coordinate_direction[p][nb.parent];
            Vec<Real, 3> delta_dot[6], sum_dot{0, 0, 0};
            for (int k = 0; k < nb.n_local + nb.n_connected; ++k) {
              delta_dot[k] = coordinate_direction[p][nb.atoms[k]] - vp;
              sum_dot += normalized_vjp(delta[k], delta_dot[k]);
            }
            auto direction_dot = normalized_vjp(sum, sum_dot);
            Real distance_dot = distance_direction[type][a];
            auto adjoint_dot = -distance_dot * normalized_vjp(sum, w)
                               - distance * normalized_vjp_jvp(sum, w, sum_dot);
            Vec<Real, 3> parent_gradient{0, 0, 0};
            for (int k = 0; k < nb.n_local + nb.n_connected; ++k) {
              auto g = normalized_vjp_jvp(delta[k], adjoint, delta_dot[k])
                       + normalized_vjp(delta[k], adjoint_dot);
              parent_gradient -= g;
              score::common::accumulate<D, Vec<Real, 3>>::add(
                  coordinate_output[p][nb.atoms[k]], g);
            }
            score::common::accumulate<D, Vec<Real, 3>>::add(
                coordinate_output[p][nb.parent], parent_gradient);
            score::common::accumulate<D, Real>::add(
                distance_output[type][a], -w.dot(direction_dot));
            weight_output[p][h] =
                vp - distance_dot * direction - distance * direction_dot;
          }
        }
      };
  if constexpr (D == Device::CPU) {
    for (int p = 0; p < poses; ++p)
      for (int b = 0; b < blocks; ++b) {
        auto type = c.types[p][b];
        if (type < 0 || type >= c.eligible.size(0)) continue;
        for (int a = 0; a < width; ++a)
          if (c.eligible[type][a] && c.rebuilt[p][b][a]) process(p, b, a, type);
      }
  } else {
    Ops<D>::template forall<launch_t>(
        mgr, n, [c, width, blocks, process] TMOL_DEVICE_FUNC(int i) {
          int a = i % width, b = (i / width) % blocks, p = i / (width * blocks);
          auto type = c.types[p][b];
          if (type >= 0 && type < c.eligible.size(0) && c.rebuilt[p][b][a]
              && c.eligible[type][a])
            process(p, b, a, type);
        });
  }
}
template <template <Device> class Ops, Device D, class Real, class Int>
struct HydrogenCompletion {
  static void f(
      ContextManager&,
      int,
      HydrogenCompletionContext<Real, Int, D>,
      TView<Vec<Real, 3>, 2, D>,
      TView<Vec<Real, 3>, 2, D>,
      TView<Real, 2, D>,
      TView<Vec<Real, 3>, 2, D>,
      TView<Real, 2, D>,
      TView<Vec<Real, 3>, 2, D>);
};
}  // namespace tmol::io::details::compiled

namespace tmol::io::details::compiled {
template <class Real, class Int, Device D>
struct HydrogenInputContext {
  TView<Int, 2, D> types;
  TView<Vec<Real, 3>, 3, D> coordinates;
  TView<bool, 3, D> missing;
  TView<int64_t, 2, D> parent;
  TView<Real, 2, D> distance;
  Real tolerance;
  TView<bool, 3, D> retained;
};
template <
    template <Device> class Ops,
    Device D,
    class Real,
    class Int,
    class Context>
void launch_hydrogen_input(Context& mgr, HydrogenInputContext<Real, Int, D> c) {
  int width = c.coordinates.size(2), blocks = c.types.size(1);
  assert(c.parent.size(1) == width && c.distance.size(1) == width);
  for (int d = 0; d < 3; ++d) {
    assert(c.missing.size(d) == c.coordinates.size(d));
    assert(c.retained.size(d) == c.coordinates.size(d));
  }
  auto n = score::common::checked_dispatch_product(
      score::common::checked_dispatch_product(
          c.types.size(0), blocks, "hydrogen input blocks"),
      width,
      "hydrogen input atoms");
  LAUNCH_BOX_128;
  auto process = [c, width] TMOL_DEVICE_FUNC(int p, int b, int a, Int type) {
    bool keep = false;
    if (type >= 0 && type < c.parent.size(0)) {
      auto parent = c.parent[type][a];
      if (parent >= 0 && parent < width && !c.missing[p][b][a]
          && !c.missing[p][b][parent]) {
        auto delta =
            (c.coordinates[p][b][a] - c.coordinates[p][b][parent]).norm()
            - c.distance[type][a];
        keep = delta > -c.tolerance && delta < c.tolerance;
      }
    }
    c.retained[p][b][a] = keep;
  };
  if constexpr (D == Device::CPU) {
    for (int p = 0; p < c.types.size(0); ++p)
      for (int b = 0; b < blocks; ++b) {
        auto type = c.types[p][b];
        for (int a = 0; a < width; ++a) process(p, b, a, type);
      }
  } else {
    Ops<D>::template forall<launch_t>(
        mgr, n, [c, width, blocks, process] TMOL_DEVICE_FUNC(int i) {
          int a = i % width, b = (i / width) % blocks, p = i / (width * blocks);
          process(p, b, a, c.types[p][b]);
        });
  }
}
template <template <Device> class Ops, Device D, class Real, class Int>
struct HydrogenInput {
  static void f(ContextManager&, HydrogenInputContext<Real, Int, D>);
};
}  // namespace tmol::io::details::compiled
