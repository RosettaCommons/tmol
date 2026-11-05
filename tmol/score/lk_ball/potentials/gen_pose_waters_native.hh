#pragma once
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/score/common/tview_binder.hh>
#include <tmol/score/lk_ball/potentials/gen_pose_waters_kernels.hh>

// Caller initializes water coordinates to NaN and coordinate derivatives to
// zero. Lockstep rotamers and cross-residue water construction use the common
// topology.
namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {
template <typename Real, typename Int, tmol::Device D>
using WaterGenCtx =
    WaterGenCtxT<tmol::score::common::TViewBinder<D>::template View, Real, Int>;

template <typename Real, typename Int, tmol::Device D>
using WaterGenDerivCtx = WaterGenDerivCtxT<
    tmol::score::common::TViewBinder<D>::template View,
    Real,
    Int>;

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device Dev,
    typename Real,
    typename Int>
void water_gen_forward(ContextManager& mgr, WaterGenCtx<Real, Int, Dev> ctx) {
  LAUNCH_BOX_32;
  auto evaluate = [=] TMOL_DEVICE_FUNC(int rot) {
    SHARED_MEMORY WaterGenSharedData<Real, lk_tile_size> shared;
    water_gen_build_waters_for_rot<DeviceOps, Dev, Real, Int, launch_t>(
        ctx, shared, rot);
  };
  DeviceOps<Dev>::template foreach_workgroup<launch_t>(
      mgr, ctx.n_rots_total, evaluate);
}

template <typename Real, typename Int, tmol::Device Dev>
struct SeparateWaterDerivCtx : WaterGenCtx<Real, Int, Dev> {
  TView<Vec<Real, 3>, 2, Dev> dE_dWxyz;
  TView<Vec<Real, 3>, 1, Dev> dE_d_pose_coords;
};

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device Dev,
    typename Real,
    typename Int>
void water_gen_backward(
    ContextManager& mgr,
    WaterGenCtx<Real, Int, Dev> base,
    TView<Vec<Real, 3>, 2, Dev> dE_dWxyz,
    TView<Vec<Real, 3>, 1, Dev> dE_d_pose_coords) {
  LAUNCH_BOX_32;
  SeparateWaterDerivCtx<Real, Int, Dev> ctx;
  static_cast<WaterGenCtx<Real, Int, Dev>&>(ctx) = base;
  ctx.dE_dWxyz = dE_dWxyz;
  ctx.dE_d_pose_coords = dE_d_pose_coords;
  auto evaluate = [=] TMOL_DEVICE_FUNC(int rot) {
    SHARED_MEMORY WaterGenSharedData<Real, lk_tile_size> shared;
    water_gen_build_water_derivs_for_rot_impl<
        DeviceOps,
        Dev,
        Real,
        Int,
        launch_t,
        common::TViewBinder<Dev>::template View>(
        ctx,
        shared,
        rot,
        ctx.sp2_water_tors,
        ctx.sp3_water_tors,
        ctx.ring_water_tors);
  };
  DeviceOps<Dev>::template foreach_workgroup<launch_t>(
      mgr, ctx.n_rots_total, evaluate);
}
}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
