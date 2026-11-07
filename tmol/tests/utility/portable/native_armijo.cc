#include <tmol/optimization/armijo.hh>
#include "native_test_ops.hh"
#include <cassert>
#include <limits>

using tmol_test::view;
using namespace tmol::optimization;

template <typename Real, typename Int>
void check() {
  tmol::ContextManager mgr;
  constexpr int n = 5;
  bool searching[n] = {true, true, true, false, true};
  Real alpha0[n] = {1, Real(.1), 4, 0, Real(4e-8)};
  Real alpha[n], phi[n], phi0[n], derivative[n];
  Real accepted[n], values[n], trial[n], next[n], next_values[n], step[n];
  Int status[n], next_status[n];
  bool active[n], failed[n];
  launch_armijo_start<tmol_test::SerialOps, tmol::Device::CPU, Real>(
      mgr, view(searching, {n}), view(alpha0, {n}), view(alpha, {n}));
  // Independent quadratic objective: x=2, direction=-2, f=.5*x*x.
  for (int i = 0; i < n; ++i) {
    Real x = 2 - 2 * alpha[i];
    phi[i] = Real(.5) * x * x;
    phi0[i] = 2;
    derivative[i] = -4;
  }
  phi[4] = std::numeric_limits<Real>::quiet_NaN();
  launch_armijo_classify<tmol_test::SerialOps, tmol::Device::CPU, Real>(
      mgr,
      view(searching, {n}),
      view(alpha, {n}),
      view(phi, {n}),
      view(phi0, {n}),
      view(derivative, {n}),
      Real(.8),
      Real(.1),
      view(accepted, {n}),
      view(values, {n}),
      view(status, {n}),
      view(active, {n}));
  Int expected[n] = {LS_DONE, LS_INCREASE, LS_BACKTRACK, LS_DONE, LS_BACKTRACK};
  for (int i = 0; i < n; ++i) assert(status[i] == expected[i]);
  launch_armijo_trial<tmol_test::SerialOps, tmol::Device::CPU, Real>(
      mgr,
      view(status, {n}),
      view(alpha, {n}),
      view(accepted, {n}),
      Real(.5),
      view(trial, {n}));
  Real trial_values[n];
  for (int i = 0; i < n; ++i) {
    Real x = 2 - 2 * trial[i];
    trial_values[i] = Real(.5) * x * x;
  }
  trial_values[4] = std::numeric_limits<Real>::infinity();
  launch_armijo_update<tmol_test::SerialOps, tmol::Device::CPU, Real>(
      mgr,
      view(status, {n}),
      view(trial, {n}),
      view(accepted, {n}),
      view(values, {n}),
      view(phi, {n}),
      view(trial_values, {n}),
      view(phi0, {n}),
      view(derivative, {n}),
      Real(.1),
      Real(1e-6),
      view(next, {n}),
      view(next_values, {n}),
      view(next_status, {n}),
      view(failed, {n}),
      view(active, {n}));
  for (int i = 0; i < 4; ++i) {
    assert(next_status[i] == LS_DONE);
    assert(!failed[i]);
  }
  assert(next_status[4] == LS_FAILED && failed[4] && next[4] == 0);
  assert(next[0] == 1 && next[1] == Real(.2) && next[2] == 1 && next[3] == 0);
  for (int i = 0; i < 3; ++i) assert(next_values[i] <= phi0[i]);
  launch_armijo_finalize<tmol_test::SerialOps, tmol::Device::CPU, Real>(
      mgr,
      view(next_status, {n}),
      view(derivative, {n}),
      view(next, {n}),
      view(alpha0, {n}),
      view(searching, {n}),
      Real(1e-6),
      view(step, {n}),
      view(failed, {n}));
  assert(step[4] == Real(.5) && failed[4]);
  for (int i = 0; i < 4; ++i) assert(step[i] == next[i] && !failed[i]);
}
int main() {
  check<float, int32_t>();
  check<float, int64_t>();
  check<double, int32_t>();
  check<double, int64_t>();
}
