// Array execution for the common coordinate-to-DOF operation.
#pragma once
#include "workspace_ffi.hh"
#include <tmol/kinematics/compiled/native_kinematics.hh>

namespace tmol::jax {
struct InverseKinematicsKernel {
  static constexpr bool zero_energy = false;
  static constexpr bool zero_workspace = false;

  static void validate(
      const std::vector<ffi::AnyBuffer>& args,
      ffi::AnyBuffer dofs,
      const ffi::AnyBuffer* derivative) {
    if (args.size() != 6 || derivative)
      throw std::invalid_argument(
          "inverse kinematics expects coordinates and five forest arrays; "
          "coordinate derivatives are not implemented");
    auto real = args[0].element_type();
    if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
      throw std::invalid_argument("kinematics requires float32/64 coordinates");
    require_shape(args[0], real, {-1, 3});
    auto count = args[0].dimensions()[0];
    for (size_t i = 1; i != args.size(); ++i)
      require_shape(args[i], ffi::DataType::S32, {count});
    require_shape(dofs, real, {count, 9});
  }

  template <typename Real>
  static size_t workspace_bytes(int64_t count) {
    if (count < 0 || count > INT32_MAX)
      throw std::invalid_argument("kinematic node count exceeds int32");
    ByteWorkspace storage;
    storage.take<kinematics::KinTransform<Real>>(count);
    return storage.size_bytes();
  }

  static size_t workspace_bytes(const std::vector<ffi::AnyBuffer>& args, bool) {
    auto count = args[0].dimensions()[0];
    return args[0].element_type() == ffi::DataType::F32
               ? workspace_bytes<float>(count)
               : workspace_bytes<double>(count);
  }

  template <template <Device> class Ops, Device D, typename Real>
  static void execute(
      const std::vector<ffi::AnyBuffer>& args,
      ffi::AnyBuffer dofs,
      const ffi::AnyBuffer*,
      ffi::AnyBuffer workspace) {
    using Transform = kinematics::KinTransform<Real>;
    if (reinterpret_cast<uintptr_t>(workspace.untyped_data())
        % alignof(Transform))
      throw std::invalid_argument(
          "kinematic transform workspace is misaligned");
    auto count = args[0].dimensions()[0];
    ContextManager context;
    kinematics::inverse_kinematics_native<Ops, D, Real, int32_t>(
        context,
        buffer_view<kinematics::KinCoord<Real>, 1, D>(args[0]),
        buffer_view<int32_t, 1, D>(args[1]),
        buffer_view<int32_t, 1, D>(args[2]),
        buffer_view<int32_t, 1, D>(args[3]),
        buffer_view<int32_t, 1, D>(args[4]),
        buffer_view<int32_t, 1, D>(args[5]),
        contiguous_view<Transform, 1, D>(workspace.untyped_data(), {count}),
        buffer_view<kinematics::KinDofs<Real>, 1, D>(dofs));
  }
};
}  // namespace tmol::jax
