#pragma once
#include <limits>
#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/score/na_torsion/potentials/na_torsion_pose_score.hh>
#include <tmol/score/na_torsion/potentials/na_rotamer_topology.hh>
namespace tmol::score::na_torsion::potentials {
template <template <tmol::Device> class Ops, tmol::Device D>
NARotamerTopologyOutput na_rotamer_topology_dispatch(
    ContextManager& mgr,
    tmol::native::Tensor types,
    tmol::native::Tensor poses,
    tmol::native::Tensor blocks,
    tmol::native::Tensor offsets,
    tmol::native::Tensor first,
    tmol::native::Tensor first_type,
    tmol::native::Tensor connections,
    tmol::native::Tensor downstream,
    tmol::native::Tensor bt_base,
    tmol::native::Tensor bt_uaids,
    tmol::native::Tensor bt_ring,
    tmol::native::Tensor bt_down) {
  NARotamerTopology<int64_t, D> ctx;
  ctx.types = TCAST(types);
  ctx.poses = TCAST(poses);
  ctx.blocks = TCAST(blocks);
  ctx.offsets = TCAST(offsets);
  ctx.first = TCAST(first);
  ctx.first_type = TCAST(first_type);
  ctx.connections = TCAST(connections);
  ctx.downstream = TCAST(downstream);
  ctx.bt_base = TCAST(bt_base);
  ctx.bt_uaids = TCAST(bt_uaids);
  ctx.bt_ring = TCAST(bt_ring);
  ctx.bt_down = TCAST(bt_down);

  int64_t const count = types.size(0), ntor = ctx.bt_uaids.size(1),
                nring = ctx.bt_ring.size(1);
  TORCH_CHECK(
      poses.size(0) == count && blocks.size(0) == count
          && offsets.size(0) == count,
      "candidate index lengths differ");
  TORCH_CHECK(
      ctx.first_type.size(0) == ctx.first.size(0)
          && ctx.first_type.size(1) == ctx.first.size(1),
      "background topology shapes differ");
  TORCH_CHECK(
      ctx.connections.size(0) == ctx.first.size(0)
          && ctx.connections.size(1) == ctx.first.size(1),
      "connection topology shape differs");
  TORCH_CHECK(
      ctx.bt_uaids.size(0) == ctx.bt_base.size(0)
          && ctx.bt_ring.size(0) == ctx.bt_base.size(0)
          && ctx.bt_down.size(0) == ctx.bt_base.size(0)
          && ctx.downstream.size(0) == ctx.bt_base.size(0),
      "block-type topology shapes differ");
  TORCH_CHECK(ctx.bt_uaids.size(2) == 4, "a torsion must contain four atoms");
  TORCH_CHECK(
      count <= std::numeric_limits<int>::max(),
      "candidate topology exceeds native dispatch limit");
  auto base = TPack<int64_t, 1, D>::empty({count});
  auto previous = TPack<int64_t, 1, D>::empty({count});
  auto is_na = TPack<bool, 1, D>::empty({count});
  auto torsions = TPack<int64_t, 3, D>::empty({count, ntor, 4});
  auto atoms_ok = TPack<bool, 3, D>::empty({count, ntor, 4});
  auto torsions_ok = TPack<bool, 2, D>::empty({count, ntor});
  auto rings = TPack<int64_t, 2, D>::empty({count, nring});
  auto ring_ok = TPack<bool, 2, D>::empty({count, nring});
  ctx.base = base.view;
  ctx.previous = previous.view;
  ctx.is_na = is_na.view;
  ctx.torsions = torsions.view;
  ctx.atoms_ok = atoms_ok.view;
  ctx.torsions_ok = torsions_ok.view;
  ctx.rings = rings.view;
  ctx.ring_ok = ring_ok.view;
  na_torsion_rotamer_topology_native<Ops, D, int64_t>(mgr, ctx);
  return {
      base.tensor,
      is_na.tensor,
      torsions.tensor,
      atoms_ok.tensor,
      torsions_ok.tensor,
      rings.tensor,
      ring_ok.tensor,
      previous.tensor};
}
}  // namespace tmol::score::na_torsion::potentials
