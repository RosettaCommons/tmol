#pragma once

// Allocation-free scientific state transitions for segmented Armijo searches.
// Frameworks own objective evaluation, storage, and loop synchronization.
#include <cmath>
#include <cstdint>
#include <tmol/utility/portable/device.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>

namespace tmol::optimization {

constexpr int64_t LS_DONE = 0;
constexpr int64_t LS_INCREASE = 1;
constexpr int64_t LS_BACKTRACK = 2;
constexpr int64_t LS_FAILED = 3;

template <
    template <Device> class DeviceDispatch,
    Device D,
    typename Real,
    typename Context,
    typename RealView,
    typename BoolView>
void launch_armijo_start(
    Context& mgr, BoolView searching, RealView alpha0, RealView alpha) {
  int const n_segments = alpha0.size(0);
  LAUNCH_BOX_128;
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, n_segments, [=] TMOL_DEVICE_FUNC(int i) {
        alpha[i] = searching[i] ? alpha0[i] : Real(0);
      });
}

template <
    template <Device> class DeviceDispatch,
    Device D,
    typename Real,
    typename Context,
    typename RealView,
    typename BoolView,
    typename IntView>
void launch_armijo_classify(
    Context& mgr,
    BoolView searching,
    RealView alpha,
    RealView phi,
    RealView phi0,
    RealView derphi0,
    Real sigma_increase,
    Real sigma_decrease,
    RealView accepted,
    RealView phi_accepted,
    IntView status,
    BoolView active) {
  int const n_segments = alpha.size(0);
  LAUNCH_BOX_128;
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, n_segments, [=] TMOL_DEVICE_FUNC(int i) {
        bool const linear =
            phi[i] <= phi0[i] + alpha[i] * sigma_increase * derphi0[i];
        bool const sufficient =
            phi[i] <= phi0[i] + alpha[i] * sigma_decrease * derphi0[i];
        bool const took = searching[i] && (linear || sufficient);

        accepted[i] = took ? alpha[i] : Real(0);
        phi_accepted[i] = took ? phi[i] : phi0[i];
        int64_t const status_i = !searching[i] ? LS_DONE
                                 : linear      ? LS_INCREASE
                                 : !sufficient ? LS_BACKTRACK
                                               : LS_DONE;
        status[i] = status_i;
        active[i] = status_i == LS_INCREASE || status_i == LS_BACKTRACK;
      });
}

template <
    template <Device> class DeviceDispatch,
    Device D,
    typename Real,
    typename Context,
    typename RealView,
    typename IntView>
void launch_armijo_trial(
    Context& mgr,
    IntView status,
    RealView alpha,
    RealView accepted,
    Real factor,
    RealView result) {
  int const n_segments = alpha.size(0);
  LAUNCH_BOX_128;
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, n_segments, [=] TMOL_DEVICE_FUNC(int i) {
        result[i] = status[i] == LS_INCREASE    ? alpha[i] / factor
                    : status[i] == LS_BACKTRACK ? alpha[i] * factor * factor
                                                : accepted[i];
      });
}

template <
    template <Device> class DeviceDispatch,
    Device D,
    typename Real,
    typename Context,
    typename RealView,
    typename BoolView,
    typename IntView>
void launch_armijo_update(
    Context& mgr,
    IntView status,
    RealView trial,
    RealView accepted,
    RealView phi_accepted,
    RealView phi,
    RealView phi_trial,
    RealView phi0,
    RealView derphi0,
    Real sigma_decrease,
    Real minstep,
    RealView next_accepted,
    RealView next_phi_accepted,
    IntView next_status,
    BoolView failed,
    BoolView active) {
  int const n_segments = trial.size(0);
  LAUNCH_BOX_128;
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, n_segments, [=] TMOL_DEVICE_FUNC(int i) {
        Real accepted_i = accepted[i];
        Real phi_accepted_i = phi_accepted[i];
        int64_t status_i = status[i];
        bool failed_i = false;

        if (status_i == LS_INCREASE) {
          if (phi_trial[i] < phi[i]) {
            accepted_i = trial[i];
            phi_accepted_i = phi_trial[i];
          }
          status_i = LS_DONE;
        } else if (status_i == LS_BACKTRACK) {
          bool const armijo =
              phi_trial[i] <= phi0[i] + trial[i] * sigma_decrease * derphi0[i];
          if (armijo) {
            accepted_i = trial[i];
            phi_accepted_i = phi_trial[i];
            status_i = LS_DONE;
          } else if (trial[i] < minstep) {
            if (phi_trial[i] < phi0[i]) {
              accepted_i = trial[i];
              phi_accepted_i = phi_trial[i];
              status_i = LS_DONE;
            } else {
              accepted_i = Real(0);
              phi_accepted_i = phi0[i];
              status_i = LS_FAILED;
              failed_i = true;
            }
          }
        }

        next_accepted[i] = accepted_i;
        next_phi_accepted[i] = phi_accepted_i;
        next_status[i] = status_i;
        failed[i] = failed_i;
        active[i] = status_i == LS_INCREASE || status_i == LS_BACKTRACK;
      });
}

template <
    template <Device> class DeviceDispatch,
    Device D,
    typename Real,
    typename Context,
    typename RealView,
    typename BoolView,
    typename IntView>
void launch_armijo_finalize(
    Context& mgr,
    IntView status,
    RealView derphi0,
    RealView accepted,
    RealView start,
    BoolView searching,
    Real minstep,
    RealView step,
    BoolView failed) {
  int const n_segments = accepted.size(0);
  LAUNCH_BOX_128;
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, n_segments, [=] TMOL_DEVICE_FUNC(int i) {
        bool const failed_i = status[i] == LS_FAILED;
        Real selected = accepted[i];
        if (failed_i) {
          Real magnitude = -derphi0[i];
          magnitude = magnitude < minstep ? minstep : magnitude;
          Real retry = Real(1) / sqrt(magnitude);
          selected = retry < Real(1) ? retry : Real(1);
        }
        step[i] = searching[i] ? selected : start[i];
        failed[i] = failed_i;
      });
}

}  // namespace tmol::optimization
