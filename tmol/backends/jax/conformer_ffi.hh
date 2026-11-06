// Typed array execution of shared conformer DOF transfer kernels.
#pragma once
#include "workspace_ffi.hh"
#include "graph_topology_ffi.hh"
#include <tmol/pack/rotamer/conformer_dofs.hh>
#include <tmol/score/common/tview_binder.hh>

namespace tmol::jax {
struct ConformerIndexStage {
  static int fill(int64_t, size_t) { return 0; }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t extent,
      int64_t width) {
    if ((op != 0 && op != 1) || a.size() != (op == 0 ? 3 : 5) || out.size() != 1
        || extent < 0 || width < 0)
      throw std::invalid_argument("invalid conformer index stage");
    auto i = ffi::DataType::S32;
    require_shape(a[0], i, {-1});
    require_shape(a[1], i, {a[0].dimensions()[0]});
    if (op == 0) {
      if (width > 1 || (extent && !a[0].dimensions()[0]))
        throw std::invalid_argument("invalid row-copy index layout");
      require_shape(a[2], i, {a[0].dimensions()[0] + 1});
    } else {
      require_shape(a[2], i, {-1});
      require_shape(a[3], i, {a[2].dimensions()[0], -1});
      require_shape(a[4], i, {-1, width});
      extent = score::common::checked_dispatch_product(
          a[3].dimensions()[0], a[3].dimensions()[1], "chi indices");
    }
    score::common::checked_dispatch_size(extent, "conformer indices");
    require_shape(out[0], ffi::DataType::U8, {2, extent, 8});
  }
  template <template <Device> class Ops, Device D>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t,
      int64_t width) {
    using namespace pack::rotamer;
    ContextManager mgr;
    if (op == 0) {
      ConformerRowIndexPlan<
          score::common::TViewBinder<D>::template View,
          int32_t>
          c{buffer_view<int32_t, 1, D>(a[0]),
            buffer_view<int32_t, 1, D>(a[1]),
            buffer_view<int32_t, 1, D>(a[2]),
            buffer_view<int64_t, 2, D>(out[0])};
      launch_conformer_row_indices<Ops, D>(mgr, c, static_cast<int>(width));
    } else {
      ChiDofIndexPlan<score::common::TViewBinder<D>::template View, int32_t> c{
          buffer_view<int32_t, 1, D>(a[0]),
          buffer_view<int32_t, 1, D>(a[1]),
          buffer_view<int32_t, 1, D>(a[2]),
          buffer_view<int32_t, 2, D>(a[3]),
          buffer_view<int32_t, 2, D>(a[4]),
          buffer_view<int64_t, 2, D>(out[0]),
          width};
      launch_chi_dof_indices<Ops, D>(mgr, c);
    }
  }
};

inline ffi::DataType validate_conformer_output(
    const std::vector<ffi::AnyBuffer>& args,
    ffi::AnyBuffer output,
    const ffi::AnyBuffer* derivative,
    size_t count,
    int64_t width = 9) {
  if (args.size() != count || derivative)
    throw std::invalid_argument("incorrect conformer DOF arguments");
  auto real = args[0].element_type();
  if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
    throw std::invalid_argument("conformer DOFs require float32 or float64");
  require_shape(args[0], real, {-1, width});
  require_shape(
      output, real, {args[0].dimensions()[0], args[0].dimensions()[1]});
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
struct CopyConformerRowsKernel {
  static constexpr bool zero_energy = false;
  static void validate(
      const std::vector<ffi::AnyBuffer>& args,
      ffi::AnyBuffer output,
      const ffi::AnyBuffer* derivative) {
    auto real = validate_conformer_output(args, output, derivative, 5, -1);
    require_shape(args[1], real, {-1, args[0].dimensions()[1]});
    require_shape(args[2], ffi::DataType::S32, {-1});
    auto count = args[2].dimensions()[0];
    require_shape(args[3], ffi::DataType::S32, {count});
    require_shape(args[4], ffi::DataType::S32, {count + 1});
  }

  template <template <Device> class Ops, Device D, typename Real>
  static void execute(
      const std::vector<ffi::AnyBuffer>& args,
      ffi::AnyBuffer output,
      const ffi::AnyBuffer*,
      int64_t n_rows,
      int64_t root_shift) {
    if (root_shift < 0 || root_shift > 1
        || (n_rows > 0 && args[2].dimensions()[0] == 0))
      throw std::invalid_argument("invalid conformer row-copy layout");
    using Context = pack::rotamer::ConformerRowCopy<
        score::common::TViewBinder<D>::template View,
        Real,
        int32_t>;
    Context c{
        buffer_view<int32_t, 1, D>(args[2]),
        buffer_view<int32_t, 1, D>(args[3]),
        buffer_view<int32_t, 1, D>(args[4]),
        buffer_view<Real, 2, D>(args[1]),
        buffer_view<Real, 2, D>(output)};
    ContextManager context;
    pack::rotamer::launch_copy_conformer_rows<Ops, D>(
        context, c, n_rows, static_cast<int>(root_shift));
  }
};
struct CorrectConformerChiKernel {
  static constexpr bool zero_energy = false;
  static void validate(
      const std::vector<ffi::AnyBuffer>& args,
      ffi::AnyBuffer output,
      const ffi::AnyBuffer* derivative) {
    auto real = validate_conformer_output(args, output, derivative, 9);
    require_shape(args[1], real, {-1, -1});
    require_shape(args[2], real, {-1, 3});
    auto rows = args[1].dimensions()[0], width = args[1].dimensions()[1];
    require_shape(args[3], ffi::DataType::S32, {-1});
    require_shape(args[4], ffi::DataType::S32, {args[3].dimensions()[0]});
    require_shape(args[5], ffi::DataType::S32, {rows});
    require_shape(args[6], ffi::DataType::S32, {rows, width});
    require_shape(args[7], ffi::DataType::S32, {-1, -1, 4});
    require_shape(
        args[8],
        ffi::DataType::S32,
        {args[7].dimensions()[0], args[7].dimensions()[1]});
  }

  template <template <Device> class Ops, Device D, typename Real>
  static void execute(
      const std::vector<ffi::AnyBuffer>& args,
      ffi::AnyBuffer output,
      const ffi::AnyBuffer*) {
    using Context = pack::rotamer::ChiDofCorrection<
        score::common::TViewBinder<D>::template View,
        Real,
        int32_t>;
    Context c{
        buffer_view<int32_t, 1, D>(args[3]),
        buffer_view<int32_t, 1, D>(args[4]),
        buffer_view<int32_t, 1, D>(args[5]),
        buffer_view<int32_t, 2, D>(args[6]),
        buffer_view<Real, 2, D>(args[1]),
        buffer_view<Real, 2, D>(args[2]),
        buffer_view<int32_t, 3, D>(args[7]),
        buffer_view<int32_t, 2, D>(args[8]),
        buffer_view<Real, 2, D>(output)};
    ContextManager context;
    pack::rotamer::launch_correct_chi_dofs<Ops, D, Real>(context, c);
  }
};
}  // namespace tmol::jax
