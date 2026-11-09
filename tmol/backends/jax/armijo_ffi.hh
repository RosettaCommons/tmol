// Typed execution of the common segmented Armijo state transitions.
#pragma once
#include "graph_topology_ffi.hh"
#include <tmol/optimization/armijo.hh>

namespace tmol::jax {
struct ArmijoStage {
  static int fill(int64_t, size_t) { return -1; }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t extent,
      int64_t width,
      double first,
      double second) {
    constexpr size_t inputs[] = {2, 5, 3, 8, 5};
    constexpr size_t outputs[] = {1, 4, 1, 5, 2};
    if (op < 0 || op >= 5 || a.size() != inputs[op] || out.size() != outputs[op]
        || extent != 0 || width != 0 || !std::isfinite(first)
        || !std::isfinite(second))
      throw std::invalid_argument("invalid Armijo transition operands");
    auto real = a[1].element_type();
    if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
      throw std::invalid_argument("Armijo transition requires float32/64");
    require_shape(a[1], real, {-1});
    auto n = a[1].dimensions()[0];
    if (n > INT32_MAX)
      throw std::invalid_argument(
          "Armijo segment count exceeds native index range");
    auto integer = op == 1   ? out[2].element_type()
                   : op >= 2 ? a[0].element_type()
                             : ffi::DataType::S32;
    if (integer != ffi::DataType::S32 && integer != ffi::DataType::S64)
      throw std::invalid_argument("Armijo status requires int32/64");
    for (size_t k = 0; k < a.size(); ++k) {
      auto type = k == 0              ? (op < 2 ? ffi::DataType::PRED : integer)
                  : op == 4 && k == 4 ? ffi::DataType::PRED
                                      : real;
      require_shape(a[k], type, {n});
    }
    for (size_t k = 0; k < out.size(); ++k) {
      auto type = (op == 1 || op == 3) && k == 2 ? integer
                  : ((op == 1 || op == 3) && k >= 3) || (op == 4 && k == 1)
                      ? ffi::DataType::PRED
                      : real;
      require_shape(out[k], type, {n});
    }
  }

  template <typename Real, typename Int, template <Device> class Ops, Device D>
  static void launch(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      double first,
      double second) {
    using namespace tmol::optimization;
    ContextManager context;
#define IN(T, I) buffer_view<T, 1, D>(a[I])
#define OUT(T, I) buffer_view<T, 1, D>(out[I])
    switch (op) {
      case 0:
        launch_armijo_start<Ops, D, Real>(
            context, IN(bool, 0), IN(Real, 1), OUT(Real, 0));
        break;
      case 1:
        launch_armijo_classify<Ops, D, Real>(
            context,
            IN(bool, 0),
            IN(Real, 1),
            IN(Real, 2),
            IN(Real, 3),
            IN(Real, 4),
            Real(first),
            Real(second),
            OUT(Real, 0),
            OUT(Real, 1),
            OUT(Int, 2),
            OUT(bool, 3));
        break;
      case 2:
        launch_armijo_trial<Ops, D, Real>(
            context,
            IN(Int, 0),
            IN(Real, 1),
            IN(Real, 2),
            Real(first),
            OUT(Real, 0));
        break;
      case 3:
        launch_armijo_update<Ops, D, Real>(
            context,
            IN(Int, 0),
            IN(Real, 1),
            IN(Real, 2),
            IN(Real, 3),
            IN(Real, 4),
            IN(Real, 5),
            IN(Real, 6),
            IN(Real, 7),
            Real(first),
            Real(second),
            OUT(Real, 0),
            OUT(Real, 1),
            OUT(Int, 2),
            OUT(bool, 3),
            OUT(bool, 4));
        break;
      case 4:
        launch_armijo_finalize<Ops, D, Real>(
            context,
            IN(Int, 0),
            IN(Real, 1),
            IN(Real, 2),
            IN(Real, 3),
            IN(bool, 4),
            Real(first),
            OUT(Real, 0),
            OUT(bool, 1));
        break;
    }
#undef IN
#undef OUT
  }

  template <template <Device> class Ops, Device D>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t,
      int64_t,
      double first,
      double second) {
    bool wide = (op == 1 ? out[2].element_type() : a[0].element_type())
                == ffi::DataType::S64;
    if (a[1].element_type() == ffi::DataType::F32) {
      if (wide)
        launch<float, int64_t, Ops, D>(a, out, op, first, second);
      else
        launch<float, int32_t, Ops, D>(a, out, op, first, second);
    } else {
      if (wide)
        launch<double, int64_t, Ops, D>(a, out, op, first, second);
      else
        launch<double, int32_t, Ops, D>(a, out, op, first, second);
    }
  }
};
}  // namespace tmol::jax
