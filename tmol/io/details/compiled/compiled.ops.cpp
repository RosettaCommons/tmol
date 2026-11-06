#include <tmol/utility/tensor/native_library.hh>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/device_operations.hh>

#include <tmol/io/details/compiled/gen_pose_leaf_atoms.hh>
#include <tmol/io/details/compiled/canonical_candidates.hh>
#include <tmol/io/details/compiled/canonical_topology.hh>
#include <tmol/io/details/compiled/canonical_policies.hh>
#include <tmol/io/details/compiled/hydrogen_completion.hh>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/io/details/compiled/resolve_his_taut.hh>

namespace tmol {
namespace io {
namespace details {
namespace compiled {

ContextManager mgr;

using tmol::native::Tensor;
using tmol::native::autograd::AutogradContext;
using tmol::native::autograd::Function;
using tmol::native::autograd::tensor_list;

class PoseLeafAtomGen
    : public tmol::native::autograd::Function<PoseLeafAtomGen> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      Tensor orig_coords,
      Tensor orig_coords_atom_missing,
      Tensor pose_stack_atom_missing,
      Tensor pose_stack_block_coord_offset,
      Tensor pose_stack_block_type,
      Tensor pose_stack_inter_block_connections,
      Tensor block_type_n_atoms,
      Tensor block_type_atom_downstream_of_conn,
      Tensor block_type_atom_ancestors,
      Tensor block_type_atom_icoors,
      Tensor block_type_atom_ancestors_backup,
      Tensor block_type_atom_icoors_backup) {
    tmol::native::Tensor new_coords;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        orig_coords.options(), "leaf_atom_gen_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result = GeneratePoseLeafAtoms<
              score::common::DeviceOperations,
              Dev,
              Real,
              Int>::
              forward(
                  mgr,
                  TCAST(orig_coords),
                  TCAST(orig_coords_atom_missing),
                  TCAST(pose_stack_atom_missing),
                  TCAST(pose_stack_block_coord_offset),
                  TCAST(pose_stack_block_type),
                  TCAST(pose_stack_inter_block_connections),
                  TCAST(block_type_n_atoms),
                  TCAST(block_type_atom_downstream_of_conn),
                  TCAST(block_type_atom_ancestors),
                  TCAST(block_type_atom_icoors),
                  TCAST(block_type_atom_ancestors_backup),
                  TCAST(block_type_atom_icoors_backup));

          new_coords = result.tensor;
        }));

    ctx->save_for_backward(
        {orig_coords,
         new_coords,
         orig_coords_atom_missing,
         pose_stack_atom_missing,
         pose_stack_block_coord_offset,
         pose_stack_block_type,
         pose_stack_inter_block_connections,
         block_type_n_atoms,
         block_type_atom_downstream_of_conn,
         block_type_atom_ancestors,
         block_type_atom_icoors,
         block_type_atom_ancestors_backup,
         block_type_atom_icoors_backup});

    return new_coords;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();

    int i = 0;

    auto orig_coords = saved[i++];
    auto new_coords = saved[i++];
    auto orig_coords_atom_missing = saved[i++];
    auto pose_stack_atom_missing = saved[i++];
    auto pose_stack_block_coord_offset = saved[i++];
    auto pose_stack_block_type = saved[i++];
    auto pose_stack_inter_block_connections = saved[i++];
    auto block_type_n_atoms = saved[i++];
    auto block_type_atom_downstream_of_conn = saved[i++];
    auto block_type_atom_ancestors = saved[i++];
    auto block_type_atom_icoors = saved[i++];
    auto block_type_atom_ancestors_backup = saved[i++];
    auto block_type_atom_icoors_backup = saved[i++];

    tmol::native::Tensor dE_d_orig_coords;

    using Int = int32_t;

    auto dE_d_new_coords = grad_outputs[0].contiguous();

    TMOL_DISPATCH_FLOATING_DEVICE(
        orig_coords.options(), "leaf_atom_gen_backward", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result = GeneratePoseLeafAtoms<
              score::common::DeviceOperations,
              Dev,
              Real,
              Int>::
              backward(
                  mgr,
                  TCAST(dE_d_new_coords),
                  TCAST(new_coords),
                  TCAST(orig_coords),
                  TCAST(orig_coords_atom_missing),
                  TCAST(pose_stack_atom_missing),
                  TCAST(pose_stack_block_coord_offset),
                  TCAST(pose_stack_block_type),
                  TCAST(pose_stack_inter_block_connections),
                  TCAST(block_type_n_atoms),
                  TCAST(block_type_atom_downstream_of_conn),
                  TCAST(block_type_atom_ancestors),
                  TCAST(block_type_atom_icoors),
                  TCAST(block_type_atom_ancestors_backup),
                  TCAST(block_type_atom_icoors_backup));
          dE_d_orig_coords = result.tensor;
        }));

    return {
        dE_d_orig_coords,
        tmol::native::Tensor(),
        tmol::native::Tensor(),
        tmol::native::Tensor(),
        tmol::native::Tensor(),

        tmol::native::Tensor(),
        tmol::native::Tensor(),
        tmol::native::Tensor(),
        tmol::native::Tensor(),
        tmol::native::Tensor(),

        tmol::native::Tensor(),
        tmol::native::Tensor()};
  };
};

Tensor pose_leaf_atom_gen_op(
    Tensor orig_coords,
    Tensor orig_coords_atom_missing,
    Tensor pose_stack_atom_missing,
    Tensor pose_stack_block_coord_offset,
    Tensor pose_stack_block_type,
    Tensor pose_stack_inter_block_connections,
    Tensor block_type_n_atoms,
    Tensor block_type_atom_downstream_of_conn,
    Tensor block_type_atom_ancestors,
    Tensor block_type_atom_icoors,
    Tensor block_type_atom_ancestors_backup,
    Tensor block_type_atom_icoors_backup) {
  return PoseLeafAtomGen::apply(
      orig_coords,
      orig_coords_atom_missing,
      pose_stack_atom_missing,
      pose_stack_block_coord_offset,
      pose_stack_block_type,
      pose_stack_inter_block_connections,
      block_type_n_atoms,
      block_type_atom_downstream_of_conn,
      block_type_atom_ancestors,
      block_type_atom_icoors,
      block_type_atom_ancestors_backup,
      block_type_atom_icoors_backup);
};

Tensor resolve_his_tautomerization(
    Tensor coords,
    Tensor res_types,
    Tensor res_type_variants,
    Tensor his_pose_ind,
    Tensor his_res_ind,
    Tensor atom_is_present,
    Tensor his_atom_inds,
    Tensor his_remapping_dst_index) {
  tmol::native::Tensor his_taut;
  TMOL_DISPATCH_FLOATING_DEVICE(coords.options(), "resolve_his_taut", ([&] {
                                  using Real = scalar_t;
                                  constexpr tmol::Device Dev = device_t;

                                  auto result = ResolveHisTaut<
                                      score::common::DeviceOperations,
                                      Dev,
                                      Real,
                                      int32_t>::
                                      f(mgr,
                                        TCAST(coords),
                                        TCAST(res_types),
                                        TCAST(res_type_variants),
                                        TCAST(his_pose_ind),
                                        TCAST(his_res_ind),
                                        TCAST(atom_is_present),
                                        TCAST(his_atom_inds),
                                        TCAST(his_remapping_dst_index));
                                  his_taut = result.tensor;
                                }));

  return his_taut;
}

std::tuple<Tensor, Tensor, Tensor> select_canonical_candidates(
    std::vector<Tensor> inputs) {
  TORCH_CHECK(inputs.size() == 10, "Canonical selection requires ten buffers");
  TORCH_CHECK(
      inputs[1].dim() == 2, "Canonical residue mask must have rank two");
  Tensor selected, score, candidate_index;
  TMOL_DISPATCH_INDEX_DEVICE(
      inputs[2].options(), "canonical_candidates", ([&] {
        using Int = scalar_t;
        auto shape =
            std::array<int64_t, 2>{inputs[1].size(0), inputs[1].size(1)};
        auto output = TPack<Int, 2, device_t>::empty(shape);
        auto scores = TPack<int64_t, 2, device_t>::empty(shape);
        auto indices = TPack<Int, 2, device_t>::empty(shape);
        CanonicalCandidateContext<Int, device_t> c{
            TCAST(inputs[0]),
            TCAST(inputs[1]),
            TCAST(inputs[2]),
            TCAST(inputs[3]),
            TCAST(inputs[4]),
            TCAST(inputs[5]),
            TCAST(inputs[6]),
            TCAST(inputs[7]),
            TCAST(inputs[8]),
            TCAST(inputs[9]),
            output.view,
            scores.view,
            indices.view};
        CanonicalCandidates<score::common::DeviceOperations, device_t, Int>::f(
            mgr, c);
        selected = output.tensor;
        score = scores.tensor;
        candidate_index = indices.tensor;
      }));
  return {selected, score, candidate_index};
}

std::tuple<Tensor, Tensor, Tensor, Tensor> canonical_chain_endings(
    std::vector<Tensor> inputs) {
  TORCH_CHECK(
      inputs.size() == 8, "Canonical chain endings require eight buffers");
  Tensor termini, first, last, polymeric;
  TMOL_DISPATCH_INDEX_DEVICE(
      inputs[1].options(), "canonical_chain_endings", ([&] {
        using Int = scalar_t;
        auto shape =
            std::array<int64_t, 2>{inputs[1].size(0), inputs[1].size(1)};
        auto a = TPack<Int, 2, device_t>::empty(shape);
        auto b = TPack<bool, 2, device_t>::empty(shape),
             c = TPack<bool, 2, device_t>::empty(shape),
             d = TPack<bool, 2, device_t>::empty(shape);
        CanonicalChainContext<Int, device_t> context{
            TCAST(inputs[0]),
            TCAST(inputs[1]),
            TCAST(inputs[2]),
            TCAST(inputs[3]),
            TCAST(inputs[4]),
            TCAST(inputs[5]),
            TCAST(inputs[6]),
            TCAST(inputs[7]),
            a.view,
            b.view,
            c.view,
            d.view};
        CanonicalTopology<score::common::DeviceOperations, device_t, Int>::
            chain_endings(mgr, context);
        termini = a.tensor;
        first = b.tensor;
        last = c.tensor;
        polymeric = d.tensor;
      }));
  return {termini, first, last, polymeric};
}
Tensor canonical_atom_termini(std::vector<Tensor> inputs) {
  TORCH_CHECK(
      inputs.size() == 9, "Canonical atom termini require nine buffers");
  Tensor output;
  TMOL_DISPATCH_INDEX_DEVICE(
      inputs[1].options(), "canonical_atom_termini", ([&] {
        using Int = scalar_t;
        auto shape =
            std::array<int64_t, 2>{inputs[1].size(0), inputs[1].size(1)};
        auto a = TPack<Int, 2, device_t>::empty(shape);
        auto counts = TPack<Int, 4, device_t>::empty(
            {shape[0], shape[1], 2, inputs[6].size(3)});
        CanonicalTerminusContext<Int, device_t> context{
            TCAST(inputs[0]),
            TCAST(inputs[1]),
            TCAST(inputs[2]),
            TCAST(inputs[3]),
            TCAST(inputs[4]),
            TCAST(inputs[5]),
            TCAST(inputs[6]),
            TCAST(inputs[7]),
            TCAST(inputs[8]),
            counts.view,
            a.view};
        CanonicalTopology<score::common::DeviceOperations, device_t, Int>::
            atom_termini(mgr, context);
        output = a.tensor;
      }));
  return output;
}

Tensor canonical_variants(std::vector<Tensor> inputs, int64_t neutral_base) {
  TORCH_CHECK(inputs.size() == 4, "Canonical variants require four buffers");
  Tensor result;
  TMOL_DISPATCH_INDEX_DEVICE(
      inputs[0].options(), "canonical_variants", ([&] {
        using Int = scalar_t;
        auto output = TPack<Int, 2, device_t>::empty(
            {inputs[0].size(0), inputs[0].size(1)});
        CanonicalVariantContext<Int, device_t> c{
            TCAST(inputs[0]),
            TCAST(inputs[1]),
            TCAST(inputs[2]),
            TCAST(inputs[3]),
            neutral_base,
            output.view};
        CanonicalPolicies<score::common::DeviceOperations, device_t, Int>::
            variants(mgr, c);
        result = output.tensor;
      }));
  return result;
}
Tensor canonical_cycles(std::vector<Tensor> inputs, double cutoff) {
  TORCH_CHECK(inputs.size() == 6, "Canonical cycles require six buffers");
  Tensor result;
  TMOL_DISPATCH_FLOATING_DEVICE(
      inputs[4].options(), "canonical_cycle_coordinates", ([&] {
        using Real = scalar_t;
        TMOL_DISPATCH_INDEX_DEVICE(
            inputs[0].options(), "canonical_cycle_indices", ([&] {
              using Int = scalar_t;
              auto output = TPack<Int, 2, device_t>::empty(
                  {inputs[0].size(0), inputs[0].size(1)});
              CanonicalCycleContext<Real, Int, device_t> c{
                  TCAST(inputs[0]),
                  TCAST(inputs[1]),
                  TCAST(inputs[2]),
                  TCAST(inputs[3]),
                  TCAST(inputs[4]),
                  TCAST(inputs[5]),
                  Real(cutoff),
                  output.view};
              CanonicalCycles<
                  score::common::DeviceOperations,
                  device_t,
                  Real,
                  Int>::f(mgr, c);
              result = output.tensor;
            }));
      }));
  return result;
}
std::tuple<Tensor, Tensor> canonical_wiring(
    std::vector<Tensor> inputs, int64_t max_ports) {
  TORCH_CHECK(inputs.size() == 11, "Canonical wiring requires eleven buffers");
  Tensor result, status;
  TMOL_DISPATCH_INDEX_DEVICE(
      inputs[0].options(), "canonical_wiring", ([&] {
        using Int = scalar_t;
        auto output = TPack<Vec<Int, 2>, 3, device_t>::empty(
            {inputs[0].size(0), inputs[0].size(1), max_ports});
        auto errors = TPack<Int, 1, device_t>::empty({inputs[0].size(0)});
        CanonicalWiringContext<Int, device_t> c{
            TCAST(inputs[0]),
            TCAST(inputs[1]),
            TCAST(inputs[2]),
            TCAST(inputs[3]),
            TCAST(inputs[4]),
            TCAST(inputs[5]),
            TCAST(inputs[6]),
            TCAST(inputs[7]),
            TCAST(inputs[8]),
            TCAST(inputs[9]),
            TCAST(inputs[10]),
            output.view,
            errors.view};
        CanonicalPolicies<score::common::DeviceOperations, device_t, Int>::
            wiring(mgr, c);
        result = output.tensor;
        status = errors.tensor;
      }));
  return {result, status};
}
Tensor canonical_connection_errors(Tensor types, Tensor connections) {
  Tensor result;
  TMOL_DISPATCH_INDEX_DEVICE(
      types.options(), "canonical_connection_errors", ([&] {
        using Int = scalar_t;
        auto bad = TPack<bool, 3, device_t>::empty(
            {connections.size(0), connections.size(1), connections.size(2)});
        CanonicalPolicies<score::common::DeviceOperations, device_t, Int>::
            validate(mgr, TCAST(types), TCAST(connections), bad.view);
        result = bad.tensor;
      }));
  return result;
}

std::tuple<Tensor, Tensor, Tensor> hydrogen_completion(
    std::vector<Tensor> inputs,
    Tensor weights,
    Tensor coordinate_direction,
    Tensor distance_direction,
    int64_t mode) {
  TORCH_CHECK(
      inputs.size() == 13 && mode >= 0 && mode <= 2,
      "Hydrogen completion requires thirteen inputs and mode0/1/2");
  Tensor result, dist_grad, weight_grad;
  TMOL_DISPATCH_FLOATING_DEVICE(
      inputs[0].options(), "hydrogen_completion_coordinates", ([&] {
        using Real = scalar_t;
        TMOL_DISPATCH_INDEX_DEVICE(
            inputs[1].options(), "hydrogen_completion_indices", ([&] {
              using Int = scalar_t;
              auto shape =
                  std::array<int64_t, 2>{inputs[0].size(0), inputs[0].size(1)};
              auto dshape = std::array<int64_t, 2>{
                  inputs[10].size(0), inputs[10].size(1)};
              auto empty = std::array<int64_t, 2>{0, 0};
              auto out = TPack<Vec<Real, 3>, 2, device_t>::empty(shape);
              auto gd = TPack<Real, 2, device_t>::empty(mode ? dshape : empty);
              auto gw = TPack<Vec<Real, 3>, 2, device_t>::empty(
                  mode == 2 ? shape : empty);
              HydrogenCompletionContext<Real, Int, device_t> c{
                  TCAST(inputs[0]),
                  TCAST(inputs[1]),
                  TCAST(inputs[2]),
                  TCAST(inputs[3]),
                  TCAST(inputs[4]),
                  TCAST(inputs[5]),
                  TCAST(inputs[6]),
                  TCAST(inputs[7]),
                  TCAST(inputs[8]),
                  TCAST(inputs[9]),
                  TCAST(inputs[10]),
                  TCAST(inputs[11]),
                  TCAST(inputs[12])};
              HydrogenCompletion<
                  score::common::DeviceOperations,
                  device_t,
                  Real,
                  Int>::
                  f(mgr,
                    mode,
                    c,
                    TCAST(weights),
                    TCAST(coordinate_direction),
                    TCAST(distance_direction),
                    out.view,
                    gd.view,
                    gw.view);
              result = out.tensor;
              dist_grad = gd.tensor;
              weight_grad = gw.tensor;
            }));
      }));
  return {result, dist_grad, weight_grad};
}
Tensor hydrogen_input(std::vector<Tensor> inputs, double tolerance) {
  TORCH_CHECK(inputs.size() == 5, "Hydrogen input requires five buffers");
  Tensor result;
  TMOL_DISPATCH_FLOATING_DEVICE(
      inputs[1].options(), "hydrogen_input_coordinates", ([&] {
        using Real = scalar_t;
        TMOL_DISPATCH_INDEX_DEVICE(
            inputs[0].options(), "hydrogen_input_indices", ([&] {
              using Int = scalar_t;
              auto out = TPack<bool, 3, device_t>::empty(
                  {inputs[2].size(0), inputs[2].size(1), inputs[2].size(2)});
              HydrogenInputContext<Real, Int, device_t> c{
                  TCAST(inputs[0]),
                  TCAST(inputs[1]),
                  TCAST(inputs[2]),
                  TCAST(inputs[3]),
                  TCAST(inputs[4]),
                  Real(tolerance),
                  out.view};
              HydrogenInput<
                  score::common::DeviceOperations,
                  device_t,
                  Real,
                  Int>::f(mgr, c);
              result = out.tensor;
            }));
      }));
  return result;
}

TMOL_LIBRARY(tmol_io, m) {
  m.def("hydrogen_completion", TMOL_FN(&hydrogen_completion));
  m.def("hydrogen_input", TMOL_FN(&hydrogen_input));
  m.def("canonical_variants", TMOL_FN(&canonical_variants));
  m.def("canonical_cycles", TMOL_FN(&canonical_cycles));
  m.def("canonical_wiring", TMOL_FN(&canonical_wiring));
  m.def("canonical_connection_errors", TMOL_FN(&canonical_connection_errors));

  m.def("canonical_chain_endings", TMOL_FN(&canonical_chain_endings));
  m.def("canonical_atom_termini", TMOL_FN(&canonical_atom_termini));
  m.def("select_canonical_candidates", TMOL_FN(&select_canonical_candidates));
  m.def("gen_pose_leaf_atoms", TMOL_FN(&pose_leaf_atom_gen_op));
  m.def("resolve_his_taut", TMOL_FN(&resolve_his_tautomerization));
}

}  // namespace compiled
}  // namespace details
}  // namespace io
}  // namespace tmol
