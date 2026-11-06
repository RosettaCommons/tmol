// Thin execution of the common torsion geometry used by constraint callbacks.
#pragma once
#include "genbonded_ffi.hh"
#include <tmol/score/common/geom.hh>

namespace tmol::jax {
template <bool DoublePrecision = false>
struct TorsionKernel {
  static void validate(
      const std::vector<ffi::AnyBuffer>& args,
      ffi::AnyBuffer energy,
      const ffi::AnyBuffer* gradient) {
    if (args.size() != 1)
      throw std::invalid_argument(
          "constraint torsion expects one coordinate operand");
    auto real = args[0].element_type();
    if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
      throw std::invalid_argument("constraint torsion requires float32/64");
    require_shape(args[0], real, {-1, 4, 3});
    auto count = args[0].dimensions()[0];
    require_shape(energy, real, {count});
    if (gradient) require_shape(*gradient, real, {count, 4, 3});
    if (count > std::numeric_limits<int32_t>::max())
      throw std::invalid_argument("constraint torsion count exceeds int32");
  }

  template <template <Device> class Ops, Device D, typename Real>
  static void execute(
      const std::vector<ffi::AnyBuffer>& args,
      ffi::AnyBuffer energy,
      const ffi::AnyBuffer* gradient) {
    auto xyz = buffer_view<Vec<Real, 3>, 2, D>(args[0]);
    auto value = buffer_view<Real, 1, D>(energy);
    TView<Vec<Real, 3>, 2, D> deriv;
    if (gradient) deriv = buffer_view<Vec<Real, 3>, 2, D>(*gradient);
    bool derivatives = gradient != nullptr;
    ContextManager context;
    LAUNCH_BOX_32;
    auto score = [=] TMOL_DEVICE_FUNC(int i) {
      using Compute = std::conditional_t<DoublePrecision, double, Real>;
      Vec<Compute, 3> local[4];
      for (int atom = 0; atom < 4; ++atom)
        for (int axis = 0; axis < 3; ++axis)
          local[atom][axis] = xyz[i][atom][axis];
      auto result = score::common::dihedral_angle<Compute>::V_dV(
          local[0], local[1], local[2], local[3]);
      value[i] = result.V;
      if (derivatives) {
        for (int axis = 0; axis < 3; ++axis) {
          deriv[i][0][axis] = result.dV_dI[axis];
          deriv[i][1][axis] = result.dV_dJ[axis];
          deriv[i][2][axis] = result.dV_dK[axis];
          deriv[i][3][axis] = result.dV_dL[axis];
        }
      }
    };
    Ops<D>::template forall<launch_t>(context, xyz.size(0), score);
  }
};
using ConstraintTorsionKernel = TorsionKernel<>;
using PreciseTorsionKernel = TorsionKernel<true>;
}  // namespace tmol::jax
