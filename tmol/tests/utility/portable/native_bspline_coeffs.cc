#include <tmol/numeric/bspline_compiled/bspline_coeffs.hh>
#include <cmath>
#include <vector>

template <typename Real, int Rank>
int check() {
  int64_t shape[Rank], stride[Rank];
  int total = 1;
  for (int d = Rank - 1; d >= 0; --d) {
    shape[d] = 5 + d;
    stride[d] = total;
    total *= shape[d];
  }
  std::vector<Real> original(total), coefficients(total);
  for (int i = 0; i < total; ++i) {
    double value = 2.0;
    for (int d = 0; d < Rank; ++d)
      value += std::sin(
          2.0 * std::acos(-1.0) * ((i / stride[d]) % shape[d]) / shape[d]);
    original[i] = coefficients[i] = Real(value);
  }
  using namespace tmol::numeric::bspline;
  tmol::TView<Real, Rank, tmol::Device::CPU> view(
      coefficients.data(), shape, stride);
  ndspline_coeffs<Rank, 3, tmol::Device::CPU, Real, int>::computeCoeffs(view);
  for (int i = 0; i < total; ++i) {
    tmol::portable::Vec<Real, Rank> point;
    for (int d = 0; d < Rank; ++d) point[d] = (i / stride[d]) % shape[d];
    Real value =
        ndspline_kernel<Rank, 3, tmol::Device::CPU, Real, int>::interpolate_V(
            view, point);
    if (std::abs(value - original[i]) > (sizeof(Real) == 4 ? 4e-6 : 2e-14))
      return Rank;
  }
  return 0;
}
int main() {
  return check<float, 2>() || check<double, 2>() || check<float, 3>()
         || check<double, 3>() || check<float, 4>() || check<double, 4>();
}
