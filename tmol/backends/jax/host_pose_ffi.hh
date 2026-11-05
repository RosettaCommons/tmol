// Host setup ABI. Every topology operation is supplied by the shared planner.
#pragma once
#include "host_array_ffi.hh"
#include <tmol/kinematics/compiled/host_pose_kinematics.hh>
#include <tmol/kinematics/compiled/native_kinematics.hh>
#include <tmol/pack/rotamer/dunbrack/host_sampling.hh>
#include <cstdio>

namespace tmol::jax {
// Host allocation/dispatch only; the inverse operation is the runtime kernel.
template <typename Real>
auto host_inverse_kinematics(
    ContextManager& context,
    TView<kinematics::KinCoord<Real>, 1, Device::CPU> coords,
    TView<int32_t, 1, Device::CPU> parent,
    TView<int32_t, 1, Device::CPU> frame_x,
    TView<int32_t, 1, Device::CPU> frame_y,
    TView<int32_t, 1, Device::CPU> frame_z,
    TView<int32_t, 1, Device::CPU> kinds) {
  auto count = coords.size(0);
  for (auto index : {parent, frame_x, frame_y, frame_z, kinds})
    if (index.size(0) != count)
      throw std::invalid_argument(
          "inverse forest and coordinate extents differ");
  for (int32_t i = 0; i < count; ++i) {
    for (auto index : {parent, frame_x, frame_y, frame_z})
      if (index[i] < 0 || index[i] >= count)
        throw std::invalid_argument("inverse forest index is out of bounds");
    if (kinds[i] < 0 || kinds[i] > 2)
      throw std::invalid_argument("invalid inverse forest node type");
  }
  using Transform = kinematics::KinTransform<Real>;
  using Dofs = kinematics::KinDofs<Real>;
  auto scratch = OwnedHostTensor<Transform, 1, Device::CPU>::full(
      {count}, Transform::Zero());
  auto result =
      OwnedHostTensor<Dofs, 1, Device::CPU>::full({count}, Dofs::Zero());
  kinematics::inverse_kinematics_native<
      kinematics::HostPoseOperations,
      Device::CPU,
      Real,
      int32_t>(
      context,
      coords,
      parent,
      frame_x,
      frame_y,
      frame_z,
      kinds,
      scratch.view,
      result.view);
  return result;
}
}  // namespace tmol::jax

extern "C" size_t tmol_kinematic_host_array_size() {
  return sizeof(tmol::jax::HostArray);
}

extern "C" int tmol_kinematic_host_prepare(
    int32_t operation,
    const tmol::jax::HostArray* arrays,
    int64_t array_count,
    const int64_t* scalars,
    int64_t scalar_count,
    void** result,
    char* error,
    size_t error_size) {
  if (!result) return 1;
  *result = nullptr;
  try {
    using Planner = tmol::kinematics::HostPoseKinematics<>;
    tmol::ContextManager context;
    tmol::jax::HostArguments inputs(arrays, array_count, scalars, scalar_count);
    auto output = std::make_unique<tmol::jax::HostResult>();
    // The generic invoker derives argument types directly from each method.
#define TMOL_HOST_OPERATION(ID, NAME)                                 \
  case ID:                                                            \
    tmol::jax::invoke_host(&Planner::NAME, context, inputs, *output); \
    break
    switch (operation) {
      TMOL_HOST_OPERATION(0, get_kfo_indices_for_atoms);
      TMOL_HOST_OPERATION(1, get_kfo_atom_parents);
      TMOL_HOST_OPERATION(2, get_children);
      TMOL_HOST_OPERATION(3, get_id_and_frame_xyz);
      TMOL_HOST_OPERATION(4, calculate_ff_edge_delays);
      TMOL_HOST_OPERATION(5, get_jump_atom_indices);
      TMOL_HOST_OPERATION(6, get_block_parent_connectivity_from_toposort);
      TMOL_HOST_OPERATION(7, get_scans2);
      TMOL_HOST_OPERATION(8, create_minimizer_map);
      TMOL_HOST_OPERATION(10, assemble_pose);
      case 11:
        if (array_count < 1)
          throw std::invalid_argument("missing inverse coordinates");
        if (arrays[0].scalar_type == 3)
          tmol::jax::invoke_host(
              &tmol::jax::host_inverse_kinematics<float>,
              context,
              inputs,
              *output);
        else if (arrays[0].scalar_type == 4)
          tmol::jax::invoke_host(
              &tmol::jax::host_inverse_kinematics<double>,
              context,
              inputs,
              *output);
        else
          throw std::invalid_argument("inverse coordinates must be float32/64");
        break;
      case 12:
        if (array_count < 1)
          throw std::invalid_argument("missing sampling coordinates");
        if (arrays[0].scalar_type == 3)
          tmol::jax::invoke_host(
              &tmol::pack::rotamer::dunbrack::HostDunbrackSampling<float>::f,
              context,
              inputs,
              *output);
        else if (arrays[0].scalar_type == 4)
          tmol::jax::invoke_host(
              &tmol::pack::rotamer::dunbrack::HostDunbrackSampling<double>::f,
              context,
              inputs,
              *output);
        else
          throw std::invalid_argument(
              "sampling coordinates must be float32/64");
        break;
      default:
        throw std::invalid_argument("unknown native pose operation");
    }
#undef TMOL_HOST_OPERATION
    *result = output.release();
    return 0;
  } catch (const std::exception& exception) {
    if (error && error_size)
      std::snprintf(error, error_size, "%s", exception.what());
  } catch (...) {
    if (error && error_size)
      std::snprintf(error, error_size, "unknown native pose preparation error");
  }
  return 1;
}

extern "C" int64_t tmol_kinematic_host_result_count(const void* result) {
  if (!result) return -1;
  return static_cast<const tmol::jax::HostResult*>(result)->arrays.size();
}

extern "C" int tmol_kinematic_host_result_array(
    const void* result, int64_t index, tmol::jax::HostArray* array) {
  if (!result || !array || index < 0) return 1;
  const auto& values =
      static_cast<const tmol::jax::HostResult*>(result)->arrays;
  if (size_t(index) >= values.size()) return 1;
  *array = values[index].array;
  return 0;
}

extern "C" void tmol_kinematic_host_release(void* result) {
  delete static_cast<tmol::jax::HostResult*>(result);
}
