#pragma once
#include <cstdint>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/score/common/uaid_util.hh>
#include <tmol/score/common/launch_box_macros.hh>
namespace tmol::score::na_torsion::potentials {
// Static candidate topology; no coordinates, parameters, or framework storage.
// Connected atoms use the neighbor's FIRST/background conformer. Inactive atom
// indices are -1 and must be masked/clamped by the coordinate gather consumer.
template <typename Int, tmol::Device D>
struct NARotamerTopology {
  TView<Int, 1, D> types, poses, blocks, offsets, bt_base, bt_down;
  TView<Int, 2, D> first, first_type, bt_ring;
  TView<tmol::portable::Vec<Int, 2>, 3, D> connections;
  TView<Int, 3, D> downstream;
  TView<tmol::portable::Vec<Int, 3>, 3, D> bt_uaids;
  TView<int64_t, 1, D> base, previous;
  TView<bool, 1, D> is_na;
  TView<int64_t, 3, D> torsions;
  TView<bool, 3, D> atoms_ok;
  TView<bool, 2, D> torsions_ok, ring_ok;
  TView<int64_t, 2, D> rings;
};
template <template <tmol::Device> class Ops, tmol::Device D, typename Int>
void na_torsion_rotamer_topology_native(
    ContextManager& mgr, NARotamerTopology<Int, D> ctx) {
  LAUNCH_BOX_32;
  auto prepare = [ctx] TMOL_DEVICE_FUNC(int r) {
    Int const bt = ctx.types[r], pose = ctx.poses[r], block = ctx.blocks[r];
    bool const real = bt >= 0 && pose >= 0 && block >= 0;
    int64_t const base = real ? ctx.bt_base[bt] : -1;
    bool const na = base >= 0;
    ctx.base[r] = base;
    ctx.is_na[r] = na;
    ctx.previous[r] = -1;
    for (int t = 0; t < ctx.bt_uaids.size(1); ++t) {
      bool valid = na;
      for (int a = 0; a < 4; ++a) {
        int64_t index = -1;
        if (real) {
          auto const raw = ctx.bt_uaids[bt][t][a];
          // Resolved chemical records normally select exactly one atom/port.
          // Preserve the NA resolver's connection precedence for explicit IDs.
          UnresolvedAtomID<Int> uaid{
              raw[1] >= 0 ? Int(-1) : raw[0],
              raw[1],
              raw[2] >= 0 ? raw[2] : Int(0)};
          auto const atom = tmol::score::common::
              resolve_local_rotamer_atom_ind_from_uaid<Int, D>(
                  uaid,
                  r,
                  block,
                  pose,
                  ctx.first,
                  ctx.first_type,
                  ctx.connections,
                  ctx.downstream);
          Int const owner = std::get<0>(atom), local = std::get<1>(atom);
          if (owner >= 0 && local >= 0)
            index = int64_t(ctx.offsets[owner]) + local;
        }
        ctx.torsions[r][t][a] = index;
        ctx.atoms_ok[r][t][a] = index >= 0;
        valid = valid && index >= 0;
      }
      ctx.torsions_ok[r][t] = valid;
    }
    for (int a = 0; a < ctx.bt_ring.size(1); ++a) {
      Int const local = real ? ctx.bt_ring[bt][a] : -1;
      ctx.rings[r][a] =
          real && local >= 0 ? int64_t(ctx.offsets[r]) + local : -1;
      ctx.ring_ok[r][a] = na && local >= 0;
    }
    if (real) {
      Int const down = ctx.bt_down[bt];
      if (down >= 0) {
        Int const previous = ctx.connections[pose][block][down][0];
        if (previous >= 0) ctx.previous[r] = ctx.first[pose][previous];
      }
    }
  };
  Ops<D>::template forall<launch_t>(mgr, ctx.types.size(0), prepare);
}
}  // namespace tmol::score::na_torsion::potentials
