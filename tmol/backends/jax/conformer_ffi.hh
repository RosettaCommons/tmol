// Typed array execution of shared conformer DOF transfer kernels.
#pragma once
#include "workspace_ffi.hh"
#include <tmol/pack/rotamer/conformer_dofs.hh>
#include <tmol/score/common/tview_binder.hh>

namespace tmol::jax {
inline ffi::DataType validate_conformer_output(
    const std::vector<ffi::AnyBuffer>& args,
    ffi::AnyBuffer output,
    const ffi::AnyBuffer* derivative,
    size_t count) {
  if (args.size() != count || derivative)
    throw std::invalid_argument("incorrect conformer DOF arguments");
  auto real = args[0].element_type();
  if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
    throw std::invalid_argument("conformer DOFs require float32 or float64");
  require_shape(args[0], real, {-1, 9});
  require_shape(output, real, {args[0].dimensions()[0], 9});
  if (output.untyped_data() != args[0].untyped_data())
    throw std::invalid_argument("conformer FFI requires an aliased DOF result");
  return real;
}

struct CopyConformerDofsKernel {
  static constexpr bool zero_energy = false;
  static void validate(
      const std::vector<ffi::AnyBuffer>& args,
      ffi::AnyBuffer output,
      const ffi::AnyBuffer* derivative) {
    auto real = validate_conformer_output(args, output, derivative, 10);
    require_shape(args[1], real, {-1, 9});
    require_shape(args[2], ffi::DataType::S32, {-1});
    auto rows = args[2].dimensions()[0];
    for (int i = 3; i <= 5; ++i)
      require_shape(args[i], ffi::DataType::S32, {rows});
    require_shape(args[6], ffi::DataType::S32, {-1});
    auto types = args[6].dimensions()[0];
    require_shape(args[7], ffi::DataType::S32, {types, -1});
    auto width = args[7].dimensions()[1];
    require_shape(args[8], ffi::DataType::S32, {-1, -1, types, width});
    require_shape(args[9], ffi::DataType::S32, {types, -1});
  }

  template <template <Device> class Ops, Device D, typename Real>
  static void execute(
      const std::vector<ffi::AnyBuffer>& args,
      ffi::AnyBuffer output,
      const ffi::AnyBuffer*,
      int64_t sampler) {
    if (sampler < 0 || sampler >= args[8].dimensions()[0])
      throw std::invalid_argument("conformer sampler index out of range");
    using Context = pack::rotamer::MainchainDofCopy<
        score::common::TViewBinder<D>::template View,
        Real,
        int32_t>;
    Context c{
        buffer_view<int32_t, 1, D>(args[2]),
        buffer_view<int32_t, 1, D>(args[3]),
        buffer_view<int32_t, 1, D>(args[4]),
        buffer_view<int32_t, 1, D>(args[5]),
        buffer_view<int32_t, 1, D>(args[6]),
        buffer_view<int32_t, 2, D>(args[7]),
        buffer_view<int32_t, 4, D>(args[8]),
        buffer_view<int32_t, 2, D>(args[9]),
        buffer_view<Real, 2, D>(args[1]),
        buffer_view<Real, 2, D>(output),
        static_cast<int>(sampler)};
    ContextManager context;
    pack::rotamer::launch_copy_mainchain_dofs<Ops, D, Real>(context, c);
  }
};

struct AssignConformerChiKernel {
  static constexpr bool zero_energy = false;
  static void validate(
      const std::vector<ffi::AnyBuffer>& args,
      ffi::AnyBuffer output,
      const ffi::AnyBuffer* derivative) {
    auto real = validate_conformer_output(args, output, derivative, 8);
    require_shape(args[1], real, {-1, -1});
    auto rows = args[1].dimensions()[0], width = args[1].dimensions()[1];
    require_shape(args[2], ffi::DataType::S32, {-1});
    require_shape(args[3], ffi::DataType::S32, {args[2].dimensions()[0]});
    require_shape(args[4], ffi::DataType::S32, {rows});
    require_shape(args[5], ffi::DataType::S32, {rows, width});
    require_shape(args[6], ffi::DataType::S32, {-1, -1});
    require_shape(
        args[7], real, {args[6].dimensions()[0], args[6].dimensions()[1]});
  }

  template <template <Device> class Ops, Device D, typename Real>
  static void execute(
      const std::vector<ffi::AnyBuffer>& args,
      ffi::AnyBuffer output,
      const ffi::AnyBuffer*) {
    using Context = pack::rotamer::ChiDofAssignment<
        score::common::TViewBinder<D>::template View,
        Real,
        int32_t>;
    Context c{
        buffer_view<int32_t, 1, D>(args[2]),
        buffer_view<int32_t, 1, D>(args[3]),
        buffer_view<int32_t, 1, D>(args[4]),
        buffer_view<int32_t, 2, D>(args[5]),
        buffer_view<Real, 2, D>(args[1]),
        buffer_view<int32_t, 2, D>(args[6]),
        buffer_view<Real, 2, D>(args[7]),
        buffer_view<Real, 2, D>(output)};
    ContextManager context;
    pack::rotamer::launch_assign_chi_dofs<Ops, D, Real>(context, c);
  }
};
}  // namespace tmol::jax
