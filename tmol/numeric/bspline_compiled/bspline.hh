#pragma once

#include "bspline_kernel.hh"
#include <tmol/utility/tensor/TensorPack.h>

namespace tmol {
namespace numeric {
namespace bspline {

template <
    std::size_t NDIM,
    std::size_t DEGREE,
    tmol::Device D,
    typename Real,
    typename Int>
struct ndspline : ndspline_kernel<NDIM, DEGREE, D, Real, Int> {
  using ndspline_kernel<NDIM, DEGREE, D, Real, Int>::interpolate;
  static auto TMOL_HOST_DEVICE_FUNC interpolate_tv(
      TView<Real, NDIM, D> coeffs,
      TView<tmol::portable::Vec<Real, NDIM>, 1, D> Xs) -> tmol::score::common::
      tuple<TPack<Real, 1, D>, TPack<tmol::portable::Vec<Real, NDIM>, 1, D> > {
    return interpolate(
        TensorAccessor<Real, NDIM, D>(coeffs),
        TensorAccessor<tmol::portable::Vec<Real, NDIM>, 1, D>(Xs));
  }

  static auto TMOL_HOST_DEVICE_FUNC interpolate(
      TensorAccessor<Real, NDIM, D> coeffs,
      TensorAccessor<tmol::portable::Vec<Real, NDIM>, 1, D> Xs)
      -> tmol::score::common::tuple<
          TPack<Real, 1, D>,
          TPack<tmol::portable::Vec<Real, NDIM>, 1, D> > {
    auto num_Vs = Xs.size(0);
    auto Vs_t = TPack<Real, 1, D>::empty({num_Vs});
    auto Vs = Vs_t.view;
    auto dV_dIs_t = TPack<tmol::portable::Vec<Real, NDIM>, 1, D>::empty(num_Vs);
    auto dV_dIs = dV_dIs_t.view;

    // fd - cuda specific code could be added here
    //  (current there are only CPU python bindings)
    for (int i = 0; i < num_Vs; ++i) {
      tmol::score::common::tie(Vs[i], dV_dIs[i]) = interpolate(coeffs, Xs[i]);
    }
    return {Vs_t, dV_dIs_t};
  }
};

}  // namespace bspline
}  // namespace numeric
}  // namespace tmol
