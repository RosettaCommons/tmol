#include <tmol/utility/tensor/native_library.hh>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/tensor/preparation_binding.hh>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/device_operations.hh>

#include "common.hh"
#include "common_dispatch.hh"
#include "params.hh"

namespace tmol {
namespace kinematics {

// Cache the mgpu::standard_context_t objects
// so as to avoid re-initializing them at each
// kernel launch
ContextManager mgr;

using tmol::native::Tensor;
using tmol::native::autograd::AutogradContext;
using tmol::native::autograd::Function;
using tmol::native::autograd::tensor_list;

class KinematicOp : public tmol::native::autograd::Function<KinematicOp> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      Tensor dofs,
      Tensor nodes_f,
      Tensor scans_f,
      Tensor gens_f,
      Tensor nodes_b,
      Tensor scans_b,
      Tensor gens_b,
      Tensor kintree) {
    tmol::native::Tensor coords;
    tmol::native::Tensor HTs;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(dofs.options(), "forward_kin_op", ([&] {
                                    using Real = scalar_t;
                                    constexpr tmol::Device Dev = device_t;

                                    auto result =
                                        ForwardKinDispatch<Dev, Real, Int>::f(
                                            mgr,
                                            TCAST(dofs),
                                            TCAST(nodes_f),
                                            TCAST(scans_f),
                                            TCAST(gens_f),
                                            TCAST(kintree));

                                    coords = std::get<0>(result).tensor;
                                    HTs = std::get<1>(result).tensor;
                                  }));

    ctx->save_for_backward({HTs, dofs, nodes_b, scans_b, gens_b, kintree});

    return coords;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    int i = 0;
    auto HTs = saved[i++];
    auto dofs = saved[i++];
    auto nodes_b = saved[i++];
    auto scans_b = saved[i++];
    auto gens_b = saved[i++];
    auto kintree = saved[i++];

    tmol::native::Tensor dV_ddof;
    using Int = int32_t;
    auto dVdx = grad_outputs[0];
    TMOL_DISPATCH_FLOATING_DEVICE(HTs.options(), "kin_deriv_op", ([&] {
                                    using Real = scalar_t;
                                    constexpr tmol::Device Dev = device_t;

                                    auto result =
                                        KinDerivDispatch<Dev, Real, Int>::f(
                                            mgr,
                                            TCAST(dVdx),
                                            TCAST(HTs),
                                            TCAST(dofs),
                                            TCAST(nodes_b),
                                            TCAST(scans_b),
                                            TCAST(gens_b),
                                            TCAST(kintree));

                                    dV_ddof = result.tensor;
                                  }));

    return {
        dV_ddof,
        tmol::native::Tensor(),
        tmol::native::Tensor(),
        tmol::native::Tensor(),
        tmol::native::Tensor(),
        tmol::native::Tensor(),
        tmol::native::Tensor(),
        tmol::native::Tensor(),
    };
  }
};

Tensor kinematic_op(
    Tensor dofs,
    Tensor nodes_f,
    Tensor scans_f,
    Tensor gens_f,
    Tensor nodes_b,
    Tensor scans_b,
    Tensor gens_b,
    Tensor kintree) {
  Tensor retval = KinematicOp::apply(
      dofs, nodes_f, scans_f, gens_f, nodes_b, scans_b, gens_b, kintree);
  return retval;
}

Tensor forward_only_op(
    Tensor dofs,
    Tensor nodes_f,
    Tensor scans_f,
    Tensor gens_f,
    Tensor kintree) {
  tmol::native::Tensor coords;

  using Int = int32_t;

  TMOL_DISPATCH_FLOATING_DEVICE(dofs.options(), "forward_kin_only_op", ([&] {
                                  using Real = scalar_t;
                                  constexpr tmol::Device Dev = device_t;

                                  auto result =
                                      ForwardKinDispatch<Dev, Real, Int>::f(
                                          mgr,
                                          TCAST(dofs),
                                          TCAST(nodes_f),
                                          TCAST(scans_f),
                                          TCAST(gens_f),
                                          TCAST(kintree));

                                  coords = std::get<0>(result).tensor;
                                }));

  return coords;
};

std::tuple<Tensor, Tensor> forward_with_ht_op(
    Tensor dofs,
    Tensor nodes_f,
    Tensor scans_f,
    Tensor gens_f,
    Tensor kintree) {
  Tensor coords;
  Tensor HTs;

  using Int = int32_t;

  TMOL_DISPATCH_FLOATING_DEVICE(dofs.options(), "forward_kin_with_ht_op", ([&] {
                                  using Real = scalar_t;
                                  constexpr tmol::Device Dev = device_t;

                                  auto result =
                                      ForwardKinDispatch<Dev, Real, Int>::f(
                                          mgr,
                                          TCAST(dofs),
                                          TCAST(nodes_f),
                                          TCAST(scans_f),
                                          TCAST(gens_f),
                                          TCAST(kintree));

                                  coords = std::get<0>(result).tensor;
                                  HTs = std::get<1>(result).tensor;
                                }));

  return std::make_tuple(coords, HTs);
};

Tensor kin_deriv_op(
    Tensor dVdx,
    Tensor HTs,
    Tensor dofs,
    Tensor nodes_b,
    Tensor scans_b,
    Tensor gens_b,
    Tensor kintree) {
  Tensor dV_ddof;

  using Int = int32_t;

  TMOL_DISPATCH_FLOATING_DEVICE(HTs.options(), "kin_deriv_only_op", ([&] {
                                  using Real = scalar_t;
                                  constexpr tmol::Device Dev = device_t;

                                  auto result =
                                      KinDerivDispatch<Dev, Real, Int>::f(
                                          mgr,
                                          TCAST(dVdx),
                                          TCAST(HTs),
                                          TCAST(dofs),
                                          TCAST(nodes_b),
                                          TCAST(scans_b),
                                          TCAST(gens_b),
                                          TCAST(kintree));

                                  dV_ddof = result.tensor;
                                }));

  return dV_ddof;
};

// Scientific signatures live in the shared planner. Registration only selects
// the execution device and converts framework tensors to those typed views.
using PreparationCPU =
    KinForestFromStencil<score::common::DeviceOperations, Device::CPU, int32_t>;
#ifdef WITH_CUDA
using PreparationCUDA = KinForestFromStencil<
    score::common::DeviceOperations,
    Device::CUDA,
    int32_t>;
#else
using PreparationCUDA = PreparationCPU;
#endif
using assemble_pose_binding = native::PreparationOperator<
    &mgr,
    &PreparationCPU::assemble_pose,
    &PreparationCUDA::assemble_pose>;
using get_kfo_indices_for_atoms_binding = native::PreparationOperator<
    &mgr,
    &PreparationCPU::get_kfo_indices_for_atoms,
    &PreparationCUDA::get_kfo_indices_for_atoms>;
using get_kfo_atom_parents_binding = native::PreparationOperator<
    &mgr,
    &PreparationCPU::get_kfo_atom_parents,
    &PreparationCUDA::get_kfo_atom_parents>;
using get_children_binding = native::PreparationOperator<
    &mgr,
    &PreparationCPU::get_children,
    &PreparationCUDA::get_children>;
using get_id_and_frame_xyz_binding = native::PreparationOperator<
    &mgr,
    &PreparationCPU::get_id_and_frame_xyz,
    &PreparationCUDA::get_id_and_frame_xyz>;
using calculate_ff_edge_delays_binding = native::PreparationOperator<
    &mgr,
    &PreparationCPU::calculate_ff_edge_delays,
    &PreparationCUDA::calculate_ff_edge_delays>;
using get_jump_atom_indices_binding = native::PreparationOperator<
    &mgr,
    &PreparationCPU::get_jump_atom_indices,
    &PreparationCUDA::get_jump_atom_indices>;
using get_block_parent_connectivity_from_toposort_binding =
    native::PreparationOperator<
        &mgr,
        &PreparationCPU::get_block_parent_connectivity_from_toposort,
        &PreparationCUDA::get_block_parent_connectivity_from_toposort>;
using get_scans2_binding = native::PreparationOperator<
    &mgr,
    &PreparationCPU::get_scans2,
    &PreparationCUDA::get_scans2>;
using minimizer_map_from_movemap_binding = native::PreparationOperator<
    &mgr,
    &PreparationCPU::create_minimizer_map,
    &PreparationCUDA::create_minimizer_map>;

auto inv_kin_dispatch(
    Tensor coords,
    Tensor parent,
    Tensor frame_x,
    Tensor frame_y,
    Tensor frame_z,
    Tensor doftype) -> Tensor {
  tmol::native::Tensor dofs;

  using Int = int32_t;

  TMOL_DISPATCH_FLOATING_DEVICE(coords.options(), "inverse_kin_op", ([&] {
                                  using Real = scalar_t;
                                  constexpr tmol::Device Dev = device_t;

                                  auto result =
                                      InverseKinDispatch<Dev, Real, Int>::f(
                                          mgr,
                                          TCAST(coords),
                                          TCAST(parent),
                                          TCAST(frame_x),
                                          TCAST(frame_y),
                                          TCAST(frame_z),
                                          TCAST(doftype));

                                  dofs = result.tensor;
                                }));

  return dofs;
}

// See https://stackoverflow.com/a/3221914

TMOL_LIBRARY(tmol_kin, m) {
  m.def("assemble_pose", TMOL_FN(&assemble_pose_binding::call));
  m.def("forward_kin_op", TMOL_FN(&kinematic_op));
  m.def("forward_only_op", TMOL_FN(&forward_only_op));
  m.def("forward_with_ht_op", TMOL_FN(&forward_with_ht_op));
  m.def("kin_deriv_op", TMOL_FN(&kin_deriv_op));
  m.def(
      "get_kfo_indices_for_atoms",
      TMOL_FN(&get_kfo_indices_for_atoms_binding::call));
  m.def("get_kfo_atom_parents", TMOL_FN(&get_kfo_atom_parents_binding::call));
  m.def("get_children", TMOL_FN(&get_children_binding::call));
  m.def("get_id_and_frame_xyz", TMOL_FN(&get_id_and_frame_xyz_binding::call));
  m.def(
      "calculate_ff_edge_delays",
      TMOL_FN(&calculate_ff_edge_delays_binding::call));
  m.def("get_jump_atom_indices", TMOL_FN(&get_jump_atom_indices_binding::call));
  m.def(
      "get_block_parent_connectivity_from_toposort",
      TMOL_FN(&get_block_parent_connectivity_from_toposort_binding::call));
  m.def(
      "get_kinforest_scans_from_stencils", TMOL_FN(&get_scans2_binding::call));
  m.def(
      "get_kinforest_scans_from_stencils2", TMOL_FN(&get_scans2_binding::call));
  m.def(
      "minimizer_map_from_movemap",
      TMOL_FN(&minimizer_map_from_movemap_binding::call));
  m.def("inverse_kin", TMOL_FN(&inv_kin_dispatch));
}

}  // namespace kinematics
}  // namespace tmol
