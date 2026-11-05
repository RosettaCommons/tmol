// Host setup ABI. Every topology operation is supplied by the shared planner.
#pragma once
#include "host_array_ffi.hh"
#include <tmol/kinematics/compiled/host_pose_kinematics.hh>
#include <cstdio>

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
