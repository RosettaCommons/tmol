#pragma once
#include "native_pose_kinematics.hh"
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorUtilEigen.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/score/unresolved_atom.hh>

namespace tmol {
namespace kinematics {
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Int>
using KinForestFromStencil =
    NativeKinForestFromStencil<DeviceDispatch, D, Int, TPack, ContextManager>;
}  // namespace kinematics
}  // namespace tmol
