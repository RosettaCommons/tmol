// Independent LJ/LK and electrostatics use only their own parameter families.
#pragma once
#include "workspace_ffi.hh"
#include <tmol/score/common/sphere_overlap_preparation.hh>
#include <tmol/score/ljlk/potentials/ljlk_pose_kernels.hh>
#include <tmol/score/elec/potentials/elec_pose_kernels.hh>

namespace tmol::jax {
template <typename Real, bool Electrostatic>
struct IndependentNonbondedWorkspace {
  ByteWorkspace storage;
  Real* spheres;
  int32_t* neighbors;
  double* accumulation;
  double* gradient_accumulation;
  size_t neighbor_capacity;
  IndependentNonbondedWorkspace(
      size_t poses,
      size_t blocks,
      void* data = nullptr,
      size_t gradient_values = 0)
      : storage(data) {
    auto pairs = score::common::checked_triangular_size(
        blocks, true, "independent nonbonded neighbors");
    neighbor_capacity =
        blocks == 1 ? 0
                    : 1
                          + score::common::checked_dispatch_product(
                              poses, pairs, "independent nonbonded neighbors");
    spheres = storage.take<Real>(blocks == 1 ? 0 : poses * blocks * 4);
    neighbors = storage.take<int32_t>(neighbor_capacity);
    accumulation = storage.take<double>(Electrostatic ? 0 : 3 * poses);
    gradient_accumulation = storage.take<double>(gradient_values);
  }
};

template <bool Electrostatic, bool PreciseGradient = false>
struct IndependentNonbondedKernel {
  static_assert(!Electrostatic || !PreciseGradient);
  static constexpr int channels = Electrostatic ? 1 : 3;
  static size_t gradient_values(
      int64_t poses, int64_t atoms, int bits, bool derivative) {
    if (!PreciseGradient || bits != 32 || !derivative) return 0;
    auto rows = score::common::checked_dispatch_product(
        poses, atoms, "independent nonbonded gradient atoms");
    return score::common::checked_dispatch_product(
        rows, channels * 3, "independent nonbonded gradient values");
  }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      ffi::AnyBuffer energy,
      const ffi::AnyBuffer* gradient) {
    if (a.size() != 16)
      throw std::invalid_argument("independent nonbonded expects 16 operands");
    auto real = a[0].element_type();
    if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
      throw std::invalid_argument("nonbonded coordinates must be float32/64");
    auto integer = ffi::DataType::S32;
    require_shape(a[0], real, {-1, -1, 3});
    auto poses = a[0].dimensions()[0], atoms = a[0].dimensions()[1];
    require_shape(a[5], integer, {poses, -1});
    auto blocks = a[5].dimensions()[1], rots = poses * blocks;
    for (int i : {1, 3, 4}) require_shape(a[i], integer, {rots});
    require_shape(a[2], integer, {poses, blocks});
    require_shape(a[6], integer, {poses, blocks, -1, 2});
    auto slots = a[6].dimensions()[2];
    require_shape(a[7], ffi::DataType::S8, {poses, blocks, slots, -1, -1});
    auto ports = a[7].dimensions()[3];
    if (!slots || ports > 12 || a[7].dimensions()[4] != ports)
      throw std::invalid_argument(
          "invalid independent nonbonded connection layout");
    require_shape(a[8], integer, {-1});
    auto types = a[8].dimensions()[0];
    require_shape(a[9], integer, {types});
    require_shape(a[10], integer, {types, ports});
    require_shape(a[11], integer, {types});
    require_shape(a[12], Electrostatic ? real : integer, {types, -1});
    auto type_atoms = a[12].dimensions()[1];
    require_shape(a[13], integer, {types, type_atoms, type_atoms});
    if constexpr (Electrostatic) {
      require_shape(a[14], integer, {types, type_atoms, type_atoms});
      require_shape(a[15], real, {1, 11});
      require_shape(energy, real, {poses});
      if (gradient) require_shape(*gradient, real, {poses, atoms, 3});
    } else {
      require_shape(a[14], real, {-1, 13});
      require_shape(a[15], real, {1, 5});
      require_shape(energy, real, {3, poses});
      if (gradient) require_shape(*gradient, real, {3, poses, atoms, 3});
    }
    if (poses * atoms > INT32_MAX || rots > INT32_MAX)
      throw std::invalid_argument(
          "independent nonbonded flattened layout exceeds int32");
  }

  static size_t workspace_bytes(
      const std::vector<ffi::AnyBuffer>& a, bool derivative) {
    auto poses = a[5].dimensions()[0], blocks = a[5].dimensions()[1];
    return a[0].element_type() == ffi::DataType::F32
               ? IndependentNonbondedWorkspace<float, Electrostatic>(
                     poses,
                     blocks,
                     nullptr,
                     gradient_values(
                         poses, a[0].dimensions()[1], 32, derivative))
                     .storage.size_bytes()
               : IndependentNonbondedWorkspace<double, Electrostatic>(
                     poses, blocks)
                     .storage.size_bytes();
  }

  template <template <Device> class Ops, Device D, typename Real>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      ffi::AnyBuffer energy,
      const ffi::AnyBuffer* gradient,
      ffi::AnyBuffer workspace,
      double reach) {
    using I = int32_t;
    constexpr bool precise = PreciseGradient && std::is_same_v<Real, float>;
    using GradReal = std::conditional_t<precise, double, Real>;
    if (!std::isfinite(reach) || reach <= 0)
      throw std::invalid_argument("neighbor reach must be finite and positive");
    auto poses = a[0].dimensions()[0], atoms = a[0].dimensions()[1];
    auto blocks = a[5].dimensions()[1], rots = poses * blocks;
    IndependentNonbondedWorkspace<Real, Electrostatic> work(
        poses,
        blocks,
        workspace.untyped_data(),
        gradient_values(poses, atoms, sizeof(Real) * 8, gradient != nullptr));
    auto xyz = contiguous_view<Vec<Real, 3>, 1, D>(
        a[0].untyped_data(), {poses * atoms});
    auto types = contiguous_view<I, 1, D>(a[5].untyped_data(), {rots});
    auto spheres =
        contiguous_view<Real, 3, D>(work.spheres, {poses, blocks, 4});
    auto neighbors = contiguous_view<I, 1, D>(
        work.neighbors, {int64_t(work.neighbor_capacity)});
    ContextManager context;
    if (blocks != 1) {
      score::common::sphere_overlap::
          compute_block_spheres_native<Ops, D, Real, I>(
              context,
              xyz,
              buffer_view<I, 1, D>(a[1]),
              buffer_view<I, 1, D>(a[3]),
              buffer_view<I, 1, D>(a[4]),
              types,
              buffer_view<I, 1, D>(a[8]),
              spheres);
      score::common::sphere_overlap::
          detect_compact_block_neighbors<Ops, D, Real, I>::f(
              context,
              buffer_view<I, 2, D>(a[5]),
              spheres,
              neighbors,
              Real(reach));
    }
    using Ctx = std::conditional_t<
        Electrostatic,
        score::elec::potentials::ElecCtx<Real, I, D>,
        score::ljlk::potentials::LJLKCtx<Real, I, D, 4, GradReal>>;
    Ctx ctx{};
    ctx.rot_coords = xyz;
    ctx.rot_coord_offset = buffer_view<I, 1, D>(a[1]);
    ctx.rot_offset_for_block = buffer_view<I, 2, D>(a[2]);
    ctx.block_ind_for_rot = buffer_view<I, 1, D>(a[3]);
    ctx.block_type_ind_for_rot = types;
    ctx.pose_stack_near_blocks = buffer_view<I, 4, D>(a[6]);
    ctx.pose_stack_inter_block_bondsep = buffer_view<int8_t, 5, D>(a[7]);
    ctx.block_type_n_atoms = buffer_view<I, 1, D>(a[8]);
    ctx.block_type_n_interblock_bonds = buffer_view<I, 1, D>(a[9]);
    ctx.block_type_atoms_forming_chemical_bonds = buffer_view<I, 2, D>(a[10]);
    ctx.block_type_all_atoms_ligand_typed = buffer_view<I, 1, D>(a[11]);
    ctx.output = contiguous_view<Real, 4, D>(
        energy.untyped_data(), {channels, poses, 1, 1});
    auto* gradient_data = precise && gradient
                              ? static_cast<void*>(work.gradient_accumulation)
                              : (gradient ? gradient->untyped_data() : nullptr);
    ctx.dV_dcoords = contiguous_view<Vec<GradReal, 3>, 2, D>(
        gradient_data, {channels, gradient ? poses * atoms : 0});
    ctx.max_n_blocks = blocks;
    ctx.max_n_upper_triangle_inds =
        score::common::checked_triangular_size(blocks, true, "nonbonded pairs");
    ctx.use_shared_compact_block_neighbors = true;
    if constexpr (Electrostatic) {
      using Global = score::elec::potentials::ElecGlobalParams<Real>;
      static_assert(sizeof(Global) == 11 * sizeof(Real));
      ctx.block_type_partial_charge = buffer_view<Real, 2, D>(a[12]);
      ctx.block_type_inter_repr_path_distance = buffer_view<I, 3, D>(a[13]);
      ctx.block_type_intra_repr_path_distance = buffer_view<I, 3, D>(a[14]);
      ctx.global_params = buffer_view<Global, 1, D>(a[15]);
      score::elec::potentials::elec_pose_forward_native<Ops, D, Real, I>(
          context, ctx, neighbors, poses, gradient != nullptr);
    } else {
      using Type = score::ljlk::potentials::LJLKTypeParams<Real>;
      using Global = score::ljlk::potentials::LJGlobalParams<Real>;
      static_assert(sizeof(Type) == 13 * sizeof(Real));
      static_assert(sizeof(Global) == 5 * sizeof(Real));
      ctx.pose_ind_for_rot = buffer_view<I, 1, D>(a[4]);
      ctx.first_rot_block_type = buffer_view<I, 2, D>(a[5]);
      ctx.shared_compact_block_neighbors = neighbors;
      ctx.block_type_atom_types = buffer_view<I, 2, D>(a[12]);
      ctx.block_type_path_distance = buffer_view<I, 3, D>(a[13]);
      ctx.type_params = buffer_view<Type, 1, D>(a[14]);
      ctx.global_params = buffer_view<Global, 1, D>(a[15]);
      ctx.require_gradient = gradient != nullptr;
      ctx.max_dis = Real(reach);
      score::ljlk::potentials::ljlk_pose_forward_native<Ops, D, Real, I>(
          context, ctx, neighbors, poses, work.accumulation);
    }
    if constexpr (precise) {
      if (gradient) {
        // Shared pair physics remains Real; round the accumulated derivative
        // once.
        auto* accumulated = work.gradient_accumulation;
        auto* result = static_cast<Real*>(gradient->untyped_data());
        LAUNCH_BOX_128;
        Ops<D>::template forall<launch_t>(
            context,
            channels * poses * atoms * 3,
            [=] TMOL_DEVICE_FUNC(int64_t i) {
              result[i] = Real(accumulated[i]);
            });
      }
    }
  }
};
}  // namespace tmol::jax
