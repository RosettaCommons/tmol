#pragma once
#include <tmol/score/common/tview_binder.hh>
#include <tmol/score/hbond/potentials/gen_hbond_bases_kernels.hh>

namespace tmol::score::hbond::potentials {
template <typename Real, typename Int, tmol::Device D>
using HBondBasesCtx = HBondBasesCtxT<
    tmol::score::common::TViewBinder<D>::template View,
    Real,
    Int>;

// Caller initializes derived coordinates to NaN and source indices to -1.
// No output or workspace allocation is performed here.
template <
    template <Device> class DeviceOps,
    Device Dev,
    typename Real,
    typename Int,
    typename Context,
    typename Ctx>
void launch_hbond_bases(Context& mgr, Ctx ctx) {
  LAUNCH_BOX_32;
  auto f_basesgen = ([=] TMOL_DEVICE_FUNC(int rot_ind) {
    SHARED_MEMORY HBondBasesGenSharedData<Real, HBOND_BASES_GEN_TILE_SIZE>
        shared_m;
    hbond_bases_gen<DeviceOps, Dev, Real, Int, launch_t>(
        ctx, shared_m, rot_ind);
  });

  DeviceOps<Dev>::template foreach_workgroup<launch_t>(
      mgr, ctx.rot_coord_offset.size(0), f_basesgen);
}
}  // namespace tmol::score::hbond::potentials
