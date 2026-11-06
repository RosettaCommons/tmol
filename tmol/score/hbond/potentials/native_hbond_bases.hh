#pragma once
#include <tmol/score/common/tview_binder.hh>
#include <tmol/score/common/counting.hh>
#include <tmol/score/hbond/potentials/gen_hbond_bases_kernels.hh>

namespace tmol::score::hbond::potentials {
template <typename Real, typename Int, tmol::Device D>
using HBondBasesCtx = HBondBasesCtxT<
    tmol::score::common::TViewBinder<D>::template View,
    Real,
    Int>;

// Caller initializes derived coordinates to NaN and source indices to -1.
// In index-only mode, an unresolved active base is -2 (zero coordinates),
// distinct from untouched -1 slots (NaN coordinates). No allocation occurs.
template <
    template <Device> class DeviceOps,
    Device Dev,
    typename Real,
    typename Int,
    bool ComputeCoordinates = true,
    typename Context,
    typename Ctx>
void launch_hbond_bases(Context& mgr, Ctx ctx) {
  LAUNCH_BOX_32;
  auto f_basesgen = ([=] TMOL_DEVICE_FUNC(int rot_ind) {
    SHARED_MEMORY HBondBasesGenSharedData<Real, HBOND_BASES_GEN_TILE_SIZE>
        shared_m;
    hbond_bases_gen<DeviceOps, Dev, Real, Int, launch_t, ComputeCoordinates>(
        ctx, shared_m, rot_ind);
  });

  DeviceOps<Dev>::template foreach_workgroup<launch_t>(
      mgr, ctx.rot_coord_offset.size(0), f_basesgen);
}

template <typename Real, typename Ctx>
TMOL_DEVICE_FUNC void hbond_gather_base_coordinates(
    TMOL_THREAD Ctx const& ctx, int index) {
  int atom = index / 3, slot = index % 3;
  int source = ctx.derived_atom_inds[atom][slot];
  Vec<Real, 3> value{Real(NAN), Real(NAN), Real(NAN)};
  if (source >= 0) value = ctx.rot_coords[source];
  // An active but unresolved base historically loads a zero vector. Unused
  // slots retain NaNs; both negative index values are ignored by gradients.
  if (source == -2) value = Vec<Real, 3>{Real(0), Real(0), Real(0)};
  ctx.derived_coords[atom][slot] = value;
}

template <
    template <Device> class DeviceOps,
    Device Dev,
    typename Real,
    typename Context,
    typename Ctx>
void gather_hbond_bases(Context& mgr, Ctx ctx) {
  LAUNCH_BOX_128;
  DeviceOps<Dev>::template forall<launch_t>(
      mgr,
      common::checked_dispatch_product(
          ctx.derived_atom_inds.size(0), 3, "HB base gather"),
      [=] TMOL_DEVICE_FUNC(int index) {
        hbond_gather_base_coordinates<Real>(ctx, index);
      });
}

}  // namespace tmol::score::hbond::potentials
