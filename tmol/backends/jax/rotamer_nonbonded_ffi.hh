// Sparse values and coordinate VJPs from the canonical nonbonded kernels.
#pragma once
#include "graph_topology_ffi.hh"
#include "energy_accumulator_ffi.hh"
#include <tmol/score/common/counting.hh>
#include <tmol/score/ljlk/potentials/ljlk_pose_kernels.hh>
#include <tmol/score/elec/potentials/elec_pose_kernels.hh>

namespace tmol::jax {
inline size_t rotamer_elec_gradient_bytes(size_t atoms) {
  ByteWorkspace storage;
  storage.take<double>(score::common::checked_dispatch_product(
      atoms, 3, "sparse electrostatic gradient values"));
  return storage.size_bytes();
}

template <bool Electrostatic, bool PreciseGradient = false>
struct RotamerNonbondedStage {
  static_assert(!PreciseGradient || Electrostatic);
  static constexpr int channels = Electrostatic ? 1 : 3;
  static int fill(int64_t, size_t output) { return output == 0 ? 0 : -1; }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t operation,
      int64_t,
      int64_t,
      double reach) {
    if ((operation != 0 && operation != 1) || a.size() != 18 + operation
        || out.size() != 1 + int(PreciseGradient) || !std::isfinite(reach)
        || reach <= 0)
      throw std::invalid_argument("invalid sparse nonbonded operands");
    auto real = a[0].element_type(), integer = ffi::DataType::S32;
    if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
      throw std::invalid_argument("sparse coordinates require float32/64");
    require_shape(a[0], real, {-1, 3});
    require_shape(a[1], integer, {-1});
    auto atoms = a[0].dimensions()[0], rots = a[1].dimensions()[0];
    for (int i : {2, 3, 4}) require_shape(a[i], integer, {rots});
    require_shape(a[5], integer, {-1, -1});
    auto poses = a[5].dimensions()[0], blocks = a[5].dimensions()[1];
    require_shape(a[6], integer, {poses, blocks});
    require_shape(a[7], integer, {poses, blocks, -1, 2});
    auto slots = a[7].dimensions()[2];
    require_shape(a[8], ffi::DataType::S8, {poses, blocks, slots, -1, -1});
    auto ports = a[8].dimensions()[3];
    if (!slots || ports > 12 || ports != a[8].dimensions()[4])
      throw std::invalid_argument("invalid sparse connection layout");
    require_shape(a[9], integer, {-1});
    auto types = a[9].dimensions()[0];
    require_shape(a[10], integer, {types});
    require_shape(a[11], integer, {types, ports});
    require_shape(a[12], integer, {types});
    require_shape(a[13], Electrostatic ? real : integer, {types, -1});
    auto width = a[13].dimensions()[1];
    require_shape(a[14], integer, {types, width, width});
    if constexpr (Electrostatic) {
      require_shape(a[15], integer, {types, width, width});
      require_shape(a[16], real, {1, 11});
    } else {
      require_shape(a[15], real, {-1, 13});
      require_shape(a[16], real, {1, 5});
    }
    require_shape(a[17], integer, {3, -1});
    auto entries = a[17].dimensions()[1];
    if (operation) {
      require_shape(a[18], real, {channels, entries});
      require_shape(out[0], real, {atoms, 3});
    } else
      require_shape(out[0], real, {channels, entries});
    if constexpr (PreciseGradient) {
      auto bytes = operation && real == ffi::DataType::F32
                       ? rotamer_elec_gradient_bytes(atoms)
                       : 0;
      require_shape(out[1], ffi::DataType::U8, {static_cast<int64_t>(bytes)});
      if (reinterpret_cast<uintptr_t>(out[1].untyped_data()) % alignof(double))
        throw std::invalid_argument(
            "electrostatic gradient workspace alignment mismatch");
    }
  }
  template <bool Backward, template <Device> class Ops, Device D, typename Real>
  static void launch(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      double reach) {
    using namespace score;
    using Accumulator = EnergyAccumulator<Ops, D, Real>;
    using GradReal = std::conditional_t<
        PreciseGradient && Backward,
        typename Accumulator::Value,
        Real>;
    using Ctx = std::conditional_t<
        Electrostatic,
        elec::potentials::ElecCtx<Real, int32_t, D, 2, GradReal>,
        ljlk::potentials::LJLKCtx<Real, int32_t, D, 2>>;
    Ctx c;
#define IN(T, R, I) buffer_view<T, R, D>(a[I])
    c.rot_coords = buffer_view<Vec<Real, 3>, 1, D>(a[0]);
    c.rot_coord_offset = IN(int32_t, 1, 1);
    c.block_ind_for_rot = IN(int32_t, 1, 2);
    c.block_type_ind_for_rot = IN(int32_t, 1, 4);
    c.rot_offset_for_block = IN(int32_t, 2, 6);
    c.pose_stack_near_blocks = IN(int32_t, 4, 7);
    c.pose_stack_inter_block_bondsep = IN(int8_t, 5, 8);
    c.block_type_n_atoms = IN(int32_t, 1, 9);
    c.block_type_n_interblock_bonds = IN(int32_t, 1, 10);
    c.block_type_atoms_forming_chemical_bonds = IN(int32_t, 2, 11);
    c.block_type_all_atoms_ligand_typed = IN(int32_t, 1, 12);
    c.dispatch_indices = IN(int32_t, 2, 17);
    c.max_n_blocks = a[5].dimensions()[1];
    if constexpr (!Electrostatic) c.max_dis = Real(reach);
    if constexpr (Electrostatic) {
      c.block_type_partial_charge = IN(Real, 2, 13);
      c.block_type_inter_repr_path_distance = IN(int32_t, 3, 14);
      c.block_type_intra_repr_path_distance = IN(int32_t, 3, 15);
      c.global_params = IN(elec::potentials::ElecGlobalParams<Real>, 1, 16);
    } else {
      c.pose_ind_for_rot = IN(int32_t, 1, 3);
      c.first_rot_block_type = IN(int32_t, 2, 5);
      c.block_type_atom_types = IN(int32_t, 2, 13);
      c.block_type_path_distance = IN(int32_t, 3, 14);
      c.type_params = IN(ljlk::potentials::LJLKTypeParams<Real>, 1, 15);
      c.global_params = IN(ljlk::potentials::LJGlobalParams<Real>, 1, 16);
    }
    ContextManager context;
    if constexpr (Backward) {
      c.dTdV = IN(Real, 2, 18);
      std::optional<Accumulator> derivative;
      if constexpr (PreciseGradient)
        derivative.emplace(
            context, out[0], static_cast<double*>(out[1].untyped_data()));
      auto* gradient_data =
          derivative
              ? derivative->template view<2>({a[0].dimensions()[0], 3}).data()
              : out[0].untyped_data();
      c.dV_dcoords = contiguous_view<Vec<GradReal, 3>, 2, D>(
          gradient_data, {1, a[0].dimensions()[0]});
      if constexpr (Electrostatic)
        elec::potentials::
            elec_rotamer_backward_native<Ops, D, Real, int32_t, GradReal>(
                context, c);
      else
        ljlk::potentials::ljlk_backward_native<true, Ops, D, Real, int32_t>(
            context, c, a[5].dimensions()[0]);
      if (derivative) derivative->finish(context);
    } else {
      c.output = buffer_view<Real, 2, D>(out[0]);
      c.output_block_pair_energies = true;
      if constexpr (Electrostatic)
        elec::potentials::elec_rotamer_forward_native<Ops, D, Real, int32_t>(
            context, c);
      else
        ljlk::potentials::ljlk_rotamer_forward_native<Ops, D, Real, int32_t>(
            context, c);
    }
#undef IN
  }
  template <template <Device> class Ops, Device D>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t operation,
      int64_t,
      int64_t,
      double reach) {
    if (a[0].element_type() == ffi::DataType::F32) {
      if (operation)
        launch<true, Ops, D, float>(a, out, reach);
      else
        launch<false, Ops, D, float>(a, out, reach);
    } else {
      if (operation)
        launch<true, Ops, D, double>(a, out, reach);
      else
        launch<false, Ops, D, double>(a, out, reach);
    }
  }
};
}  // namespace tmol::jax
