#pragma once

#include <tmol/optimization/armijo_compiled.hh>
#include <tmol/optimization/armijo.hh>

namespace tmol {
namespace optimization {

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real>
TPack<Real, 1, D> ArmijoCompiledDispatch<DeviceDispatch, D, Real>::start(
    ContextManager& mgr,
    TView<bool, 1, D> searching,
    TView<Real, 1, D> alpha0) {
  int const n_segments = alpha0.size(0);
  auto alpha_t = TPack<Real, 1, D>::empty({n_segments});
  auto alpha = alpha_t.view;

  launch_armijo_start<DeviceDispatch, D, Real>(mgr, searching, alpha0, alpha);
  return alpha_t;
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real>
std::tuple<
    TPack<Real, 1, D>,
    TPack<Real, 1, D>,
    TPack<int64_t, 1, D>,
    TPack<bool, 1, D>>
ArmijoCompiledDispatch<DeviceDispatch, D, Real>::classify(
    ContextManager& mgr,
    TView<bool, 1, D> searching,
    TView<Real, 1, D> alpha,
    TView<Real, 1, D> phi,
    TView<Real, 1, D> phi0,
    TView<Real, 1, D> derphi0,
    Real sigma_increase,
    Real sigma_decrease) {
  int const n_segments = alpha.size(0);
  auto accepted_t = TPack<Real, 1, D>::empty({n_segments});
  auto phi_accepted_t = TPack<Real, 1, D>::empty({n_segments});
  auto status_t = TPack<int64_t, 1, D>::empty({n_segments});
  auto active_t = TPack<bool, 1, D>::empty({n_segments});
  auto accepted = accepted_t.view;
  auto phi_accepted = phi_accepted_t.view;
  auto status = status_t.view;
  auto active = active_t.view;

  launch_armijo_classify<DeviceDispatch, D, Real>(
      mgr,
      searching,
      alpha,
      phi,
      phi0,
      derphi0,
      sigma_increase,
      sigma_decrease,
      accepted,
      phi_accepted,
      status,
      active);
  return {accepted_t, phi_accepted_t, status_t, active_t};
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real>
TPack<Real, 1, D> ArmijoCompiledDispatch<DeviceDispatch, D, Real>::trial(
    ContextManager& mgr,
    TView<int64_t, 1, D> status,
    TView<Real, 1, D> alpha,
    TView<Real, 1, D> accepted,
    Real factor) {
  int const n_segments = alpha.size(0);
  auto result_t = TPack<Real, 1, D>::empty({n_segments});
  auto result = result_t.view;

  launch_armijo_trial<DeviceDispatch, D, Real>(
      mgr, status, alpha, accepted, factor, result);
  return result_t;
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real>
std::tuple<
    TPack<Real, 1, D>,
    TPack<Real, 1, D>,
    TPack<int64_t, 1, D>,
    TPack<bool, 1, D>,
    TPack<bool, 1, D>>
ArmijoCompiledDispatch<DeviceDispatch, D, Real>::update(
    ContextManager& mgr,
    TView<int64_t, 1, D> status,
    TView<Real, 1, D> trial,
    TView<Real, 1, D> accepted,
    TView<Real, 1, D> phi_accepted,
    TView<Real, 1, D> phi,
    TView<Real, 1, D> phi_trial,
    TView<Real, 1, D> phi0,
    TView<Real, 1, D> derphi0,
    Real sigma_decrease,
    Real minstep) {
  int const n_segments = trial.size(0);
  auto next_accepted_t = TPack<Real, 1, D>::empty({n_segments});
  auto next_phi_accepted_t = TPack<Real, 1, D>::empty({n_segments});
  auto next_status_t = TPack<int64_t, 1, D>::empty({n_segments});
  auto failed_t = TPack<bool, 1, D>::empty({n_segments});
  auto active_t = TPack<bool, 1, D>::empty({n_segments});
  auto next_accepted = next_accepted_t.view;
  auto next_phi_accepted = next_phi_accepted_t.view;
  auto next_status = next_status_t.view;
  auto failed = failed_t.view;
  auto active = active_t.view;

  launch_armijo_update<DeviceDispatch, D, Real>(
      mgr,
      status,
      trial,
      accepted,
      phi_accepted,
      phi,
      phi_trial,
      phi0,
      derphi0,
      sigma_decrease,
      minstep,
      next_accepted,
      next_phi_accepted,
      next_status,
      failed,
      active);
  return {
      next_accepted_t, next_phi_accepted_t, next_status_t, failed_t, active_t};
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real>
std::tuple<TPack<Real, 1, D>, TPack<bool, 1, D>>
ArmijoCompiledDispatch<DeviceDispatch, D, Real>::finalize(
    ContextManager& mgr,
    TView<int64_t, 1, D> status,
    TView<Real, 1, D> derphi0,
    TView<Real, 1, D> accepted,
    TView<Real, 1, D> start,
    TView<bool, 1, D> searching,
    Real minstep) {
  int const n_segments = accepted.size(0);
  auto step_t = TPack<Real, 1, D>::empty({n_segments});
  auto failed_t = TPack<bool, 1, D>::empty({n_segments});
  auto step = step_t.view;
  auto failed = failed_t.view;

  launch_armijo_finalize<DeviceDispatch, D, Real>(
      mgr, status, derphi0, accepted, start, searching, minstep, step, failed);
  return {step_t, failed_t};
}

}  // namespace optimization
}  // namespace tmol
