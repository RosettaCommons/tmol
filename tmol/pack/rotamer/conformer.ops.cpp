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
TMOL_LIBRARY(tmol_conformer_dofs, m) {
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
