#include <tmol/utility/tensor/native_library.hh>
#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/function_dispatch/aten.hh>
#include <tmol/score/common/device_operations.hh>
#include <tmol/pack/rotamer/conformer_ops.hh>
#include <tmol/pack/rotamer/opth_sampling_torch.hh>
#include <tmol/pack/rotamer/pose_imposition_torch.hh>
namespace tmol::pack::rotamer {
using tmol::native::Tensor;
ContextManager conformer_context;
void copy_mainchain_dofs(
    Tensor source_types,
    Tensor source_offsets,
    Tensor target_types,
    Tensor target_offsets,
    Tensor source_fingerprint,
    Tensor source_atoms,
    Tensor target_atoms,
    Tensor kinematic_index,
    Tensor source_dofs,
    Tensor target_dofs,
    int64_t sampler) {
  TORCH_CHECK(
      target_dofs.dim() == 2 && target_dofs.size(1) == 9,
      "conformer DOFs must have nine components");
  TMOL_DISPATCH_FLOATING_DEVICE(
      target_dofs.options(), "copy_mainchain_dofs", ([&] {
        using Kernels = ConformerDofKernels<
            score::common::DeviceOperations,
            device_t,
            scalar_t,
            int64_t>;
        Kernels::Copy c;
        c.source_types = TCAST(source_types);
        c.source_offsets = TCAST(source_offsets);
        c.target_types = TCAST(target_types);
        c.target_offsets = TCAST(target_offsets);
        c.source_fingerprint = TCAST(source_fingerprint);
        c.source_atoms = TCAST(source_atoms);
        c.target_atoms = TCAST(target_atoms);
        c.kinematic_index = TCAST(kinematic_index);
        c.source_dofs = TCAST(source_dofs);
        c.target_dofs = TCAST(target_dofs);
        c.sampler = sampler;
        Kernels::copy(conformer_context, c);
      }));
}
void assign_chi_dofs(
    Tensor types,
    Tensor offsets,
    Tensor conformers,
    Tensor atoms,
    Tensor chi,
    Tensor kinematic_index,
    Tensor ring_corrections,
    Tensor dofs) {
  TORCH_CHECK(
      dofs.dim() == 2 && dofs.size(1) == 9,
      "conformer DOFs must have nine components");
  TMOL_DISPATCH_FLOATING_DEVICE(dofs.options(), "assign_chi_dofs", ([&] {
                                  using Kernels = ConformerDofKernels<
                                      score::common::DeviceOperations,
                                      device_t,
                                      scalar_t,
                                      int64_t>;
                                  Kernels::Chi c;
                                  c.types = TCAST(types);
                                  c.offsets = TCAST(offsets);
                                  c.conformers = TCAST(conformers);
                                  c.atoms = TCAST(atoms);
                                  c.chi = TCAST(chi);
                                  c.kinematic_index = TCAST(kinematic_index);
                                  c.ring_corrections = TCAST(ring_corrections);
                                  c.dofs = TCAST(dofs);
                                  Kernels::assign(conformer_context, c);
                                }));
}
void copy_conformer_rows(
    Tensor source_offsets,
    Tensor target_offsets,
    Tensor packed_offsets,
    Tensor source_values,
    Tensor target_values,
    int64_t n_rows,
    int64_t root_shift) {
  TORCH_CHECK(
      source_values.dim() == 2 && target_values.dim() == 2
          && source_values.size(1) == target_values.size(1),
      "Conformer row widths must match");
  TORCH_CHECK(
      root_shift == 0 || root_shift == 1, "Invalid conformer root shift");
  TORCH_CHECK(
      n_rows == 0 || source_offsets.numel() > 0,
      "Nonempty conformer copy requires segments");
  TORCH_CHECK(
      source_offsets.numel() == target_offsets.numel()
          && packed_offsets.numel() == source_offsets.numel() + 1,
      "Conformer copy offsets have inconsistent lengths");
  TMOL_DISPATCH_FLOATING_DEVICE(
      target_values.options(), "copy_conformer_rows", ([&] {
        using Kernels = ConformerDofKernels<
            score::common::DeviceOperations,
            device_t,
            scalar_t,
            int64_t>;
        Kernels::Rows c;
        c.source_offsets = TCAST(source_offsets);
        c.target_offsets = TCAST(target_offsets);
        c.packed_offsets = TCAST(packed_offsets);
        c.source_values = TCAST(source_values);
        c.target_values = TCAST(target_values);
        Kernels::rows(conformer_context, c, n_rows, root_shift);
      }));
}

void correct_chi_dofs(
    Tensor types,
    Tensor offsets,
    Tensor conformers,
    Tensor atoms,
    Tensor chi,
    Tensor trial_coords,
    Tensor chi_atoms,
    Tensor kinematic_index,
    Tensor dofs) {
  TORCH_CHECK(
      dofs.dim() == 2 && dofs.size(1) == 9,
      "conformer DOFs require nine components");
  TMOL_DISPATCH_FLOATING_DEVICE(dofs.options(), "correct_chi_dofs", ([&] {
                                  using Kernels = ConformerDofKernels<
                                      score::common::DeviceOperations,
                                      device_t,
                                      scalar_t,
                                      int64_t>;
                                  Kernels::Correction c;
                                  c.types = TCAST(types);
                                  c.offsets = TCAST(offsets);
                                  c.conformers = TCAST(conformers);
                                  c.atoms = TCAST(atoms);
                                  c.chi = TCAST(chi);
                                  c.trial_coords = TCAST(trial_coords);
                                  c.chi_atoms = TCAST(chi_atoms);
                                  c.kinematic_index = TCAST(kinematic_index);
                                  c.dofs = TCAST(dofs);
                                  Kernels::correct(conformer_context, c);
                                }));
}

Tensor select_pose_rotamers(std::vector<Tensor> inputs) {
  TORCH_CHECK(!inputs.empty(), "Pose imposition requires inputs");
  Tensor output;
  TMOL_DISPATCH_INDEX_DEVICE(
      inputs[0].options(), "select_pose_rotamers", ([&] {
        output =
            PoseImpositionTorch<score::common::DeviceOperations, device_t>::
                select(conformer_context, inputs);
      }));
  return output;
}
std::vector<Tensor> plan_pose_imposition(
    std::vector<Tensor> inputs, int64_t capacity, int64_t ncoords) {
  TORCH_CHECK(!inputs.empty(), "Pose imposition requires inputs");
  std::vector<Tensor> output;
  TMOL_DISPATCH_INDEX_DEVICE(
      inputs[0].options(), "plan_pose_imposition", ([&] {
        output =
            PoseImpositionTorch<score::common::DeviceOperations, device_t>::
                plan(conformer_context, inputs, capacity, ncoords);
      }));
  return output;
}
std::tuple<Tensor, Tensor> opth_sampling_counts(
    std::vector<Tensor> inputs, bool flip) {
  TORCH_CHECK(!inputs.empty(), "OptH sampler needs input buffers");
  std::tuple<Tensor, Tensor> output;
  TMOL_DISPATCH_INDEX_DEVICE(
      inputs[0].options(), "opth_sampling_counts", ([&] {
        output = OptHSamplingTorch<score::common::DeviceOperations, device_t>::
            counts(conformer_context, inputs, flip);
      }));
  return output;
}
std::tuple<Tensor, Tensor, Tensor, Tensor> opth_sampling_fill(
    std::vector<Tensor> inputs, Tensor counts, int64_t nrot) {
  TORCH_CHECK(!inputs.empty(), "OptH sampler needs input buffers");
  std::tuple<Tensor, Tensor, Tensor, Tensor> output;
  TMOL_DISPATCH_INDEX_DEVICE(
      inputs[0].options(), "opth_sampling_fill", ([&] {
        output =
            OptHSamplingTorch<score::common::DeviceOperations, device_t>::fill(
                conformer_context, inputs, counts, nrot);
      }));
  return output;
}
std::tuple<Tensor, Tensor, Tensor> opth_sampling_proton(
    std::vector<Tensor> inputs, int64_t width) {
  TORCH_CHECK(!inputs.empty(), "OptH sampler needs input buffers");
  std::tuple<Tensor, Tensor, Tensor> output;
  TMOL_DISPATCH_INDEX_DEVICE(
      inputs[0].options(), "opth_sampling_proton", ([&] {
        output = OptHSamplingTorch<score::common::DeviceOperations, device_t>::
            proton(conformer_context, inputs, width);
      }));
  return output;
}
Tensor conformer_row_indices(
    Tensor source,
    Tensor target,
    Tensor packed,
    int64_t n_rows,
    int64_t shift) {
  TORCH_CHECK(shift == 0 || shift == 1, "Invalid conformer root shift");
  TORCH_CHECK(
      source.numel() == target.numel() && packed.numel() == source.numel() + 1,
      "Conformer row index axes differ");
  TORCH_CHECK(
      n_rows == 0 || source.numel() > 0,
      "Nonempty conformer rows require segments");
  int n = score::common::checked_dispatch_size(n_rows, "conformer row indices");
  Tensor result;
  TMOL_DISPATCH_INDEX_DEVICE(
      source.options(), "conformer_row_indices", ([&] {
        using Kernels = ConformerDofKernels<
            score::common::DeviceOperations,
            device_t,
            float,
            int64_t>;
        auto indices = TPack<int64_t, 2, device_t>::empty({2, n});
        typename Kernels::RowIndices c{
            TCAST(source), TCAST(target), TCAST(packed), indices.view};
        Kernels::row_indices(conformer_context, c, shift);
        result = indices.tensor;
      }));
  return result;
}
Tensor chi_dof_indices(
    Tensor types,
    Tensor offsets,
    Tensor conformers,
    Tensor atoms,
    Tensor kinematic_index,
    int64_t correction_width) {
  TORCH_CHECK(
      atoms.dim() == 2 && atoms.size(0) == conformers.numel(),
      "Conformer chi index axes differ");
  TORCH_CHECK(
      correction_width >= 0, "Conformer correction width must be nonnegative");
  int n = score::common::checked_dispatch_product(
      atoms.size(0), atoms.size(1), "conformer chi indices");
  Tensor result;
  TMOL_DISPATCH_INDEX_DEVICE(types.options(), "chi_dof_indices", ([&] {
                               using Kernels = ConformerDofKernels<
                                   score::common::DeviceOperations,
                                   device_t,
                                   float,
                                   int64_t>;
                               auto indices =
                                   TPack<int64_t, 2, device_t>::empty({2, n});
                               typename Kernels::ChiIndices c{
                                   TCAST(types),
                                   TCAST(offsets),
                                   TCAST(conformers),
                                   TCAST(atoms),
                                   TCAST(kinematic_index),
                                   indices.view,
                                   correction_width};
                               Kernels::chi_indices(conformer_context, c);
                               result = indices.tensor;
                             }));
  return result;
}
Tensor conformer_copy_plan(std::vector<Tensor> inputs) {
  TORCH_CHECK(inputs.size() == 7, "Conformer copy plan requires seven buffers");
  TORCH_CHECK(
      inputs[0].dim() == 2, "Original block types must be pose by block");
  int n = score::common::checked_dispatch_size(
      inputs[0].numel(), "conformer original blocks");
  int selected = score::common::checked_dispatch_size(
      inputs[6].numel(), "conformer selected blocks");
  Tensor result;
  TMOL_DISPATCH_INDEX_DEVICE(
      inputs[0].options(), "conformer_copy_plan", ([&] {
        using Kernels = ConformerDofKernels<
            score::common::DeviceOperations,
            device_t,
            float,
            int64_t>;
        auto counts = TPack<int64_t, 1, device_t>::empty({n});
        auto offsets = TPack<int64_t, 1, device_t>::empty({n});
        auto segments = TPack<int64_t, 2, device_t>::empty({3, selected});
        typename Kernels::CopyPlan c{
            TCAST(inputs[0]),
            TCAST(inputs[1]),
            TCAST(inputs[2]),
            TCAST(inputs[3]),
            TCAST(inputs[4]),
            TCAST(inputs[5]),
            TCAST(inputs[6]),
            counts.view,
            offsets.view,
            segments.view};
        Kernels::copy_plan(conformer_context, c);
        result = segments.tensor;
      }));
  return result;
}
Tensor opth_reset_plan(
    std::vector<Tensor> inputs, int64_t ideal_width, bool flip) {
  TORCH_CHECK(
      inputs.size() == 6 && ideal_width >= 0,
      "OptH reset requires six buffers and nonnegative ideal width");
  TORCH_CHECK(
      inputs[3].dim() == 2 && inputs[3].size(0) == inputs[2].numel(),
      "OptH reset chi axes differ");
  int n = score::common::checked_dispatch_product(
      inputs[2].numel(), inputs[4].size(1), "OptH reset indices");
  Tensor result;
  TMOL_DISPATCH_INDEX_DEVICE(inputs[0].options(), "opth_reset_plan", ([&] {
                               using Kernels = ConformerDofKernels<
                                   score::common::DeviceOperations,
                                   device_t,
                                   float,
                                   int64_t>;
                               auto indices =
                                   TPack<int64_t, 2, device_t>::empty({2, n});
                               typename Kernels::ResetPlan c{
                                   TCAST(inputs[0]),
                                   TCAST(inputs[1]),
                                   TCAST(inputs[2]),
                                   TCAST(inputs[3]),
                                   TCAST(inputs[4]),
                                   TCAST(inputs[5]),
                                   indices.view,
                                   ideal_width,
                                   flip};
                               Kernels::opth_reset(conformer_context, c);
                               result = indices.tensor;
                             }));
  return result;
}
TMOL_LIBRARY(tmol_conformer_dofs, m) {
  m.def(
      "conformer_copy_plan(Tensor[] inputs) -> Tensor",
      TMOL_FN(&conformer_copy_plan));
  m.def(
      "opth_reset_plan(Tensor[] inputs, int ideal_width, bool flip) -> Tensor",
      TMOL_FN(&opth_reset_plan));
  m.def(
      "chi_dof_indices(Tensor types, Tensor offsets, Tensor conformers, Tensor "
      "atoms, Tensor kinematic_index, int correction_width) -> Tensor",
      TMOL_FN(&chi_dof_indices));
  m.def(
      "conformer_row_indices(Tensor source, Tensor target, Tensor packed, int "
      "n_rows, int shift) -> Tensor",
      TMOL_FN(&conformer_row_indices));
  m.def(
      "opth_sampling_counts(Tensor[] inputs, bool flip) -> (Tensor, Tensor)",
      TMOL_FN(&opth_sampling_counts));
  m.def(
      "opth_sampling_fill(Tensor[] inputs, Tensor counts, int nrot) -> "
      "(Tensor, Tensor, Tensor, Tensor)",
      TMOL_FN(&opth_sampling_fill));
  m.def(
      "opth_sampling_proton(Tensor[] inputs, int width) -> (Tensor, Tensor, "
      "Tensor)",
      TMOL_FN(&opth_sampling_proton));

  m.def(
      "select_pose_rotamers(Tensor[] inputs) -> Tensor",
      TMOL_FN(&select_pose_rotamers));
  m.def(
      "plan_pose_imposition(Tensor[] inputs, int capacity, int ncoords) -> "
      "Tensor[]",
      TMOL_FN(&plan_pose_imposition));
  m.def(
      "correct_chi_dofs(Tensor types, Tensor offsets, Tensor conformers, "
      "Tensor atoms, Tensor chi, Tensor trial_coords, Tensor chi_atoms, Tensor "
      "kinematic_index, Tensor(a!) dofs) -> ()",
      TMOL_FN(&correct_chi_dofs));

  m.def(
      "copy_conformer_rows(Tensor source_offsets, Tensor target_offsets, "
      "Tensor packed_offsets, Tensor source_values, Tensor(a!) target_values, "
      "int n_rows, int root_shift) -> ()",
      TMOL_FN(&copy_conformer_rows));
  m.def(
      "copy_mainchain_dofs(Tensor source_types, Tensor source_offsets, Tensor "
      "target_types, Tensor target_offsets, Tensor source_fingerprint, Tensor "
      "source_atoms, Tensor target_atoms, Tensor kinematic_index, Tensor "
      "source_dofs, Tensor(a!) target_dofs, int sampler) -> ()",
      TMOL_FN(&copy_mainchain_dofs));
  m.def(
      "assign_chi_dofs(Tensor types, Tensor offsets, Tensor conformers, Tensor "
      "atoms, Tensor chi, Tensor kinematic_index, Tensor ring_corrections, "
      "Tensor(a!) dofs) -> ()",
      TMOL_FN(&assign_chi_dofs));
}
}  // namespace tmol::pack::rotamer
