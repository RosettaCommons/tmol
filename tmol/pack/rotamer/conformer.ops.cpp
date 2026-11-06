#include <tmol/utility/tensor/native_library.hh>
#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>
#include <tmol/score/common/device_operations.hh>
#include <tmol/pack/rotamer/conformer_ops.hh>
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

TMOL_LIBRARY(tmol_conformer_dofs, m) {
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
