#pragma once
#include "native_pose_kinematics.impl.hh"
#include <tmol/score/common/serial_cpu_operations.hh>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/utility/tensor/owned_host_tensor.hh>

namespace tmol {
namespace kinematics {
template <Device D>
struct HostPoseOperations;
template <>
struct HostPoseOperations<Device::CPU> : score::common::SerialCPUOperations {};

// Topology is prepared once on CPU and then placed on the execution device.
// The returned packs own their arrays independently of input buffer lifetimes.
template <typename Int = int32_t>
using HostPoseKinematics = NativeKinForestFromStencil<
    HostPoseOperations,
    Device::CPU,
    Int,
    OwnedHostTensor,
    ContextManager>;
}  // namespace kinematics
}  // namespace tmol
