// Shared fused nonbonded physics with live canonical spatial neighbors.
#pragma once
#include "workspace_ffi.hh"
#include <tmol/score/common/sphere_overlap_preparation.hh>
#include <tmol/score/ljlk/potentials/ljlk_elec_pose_kernels.hh>

namespace tmol::jax {
template <typename Real, bool Weighted = false>
struct NonbondedWorkspace {
  ByteWorkspace storage;
  Real* spheres;
  int32_t* neighbors;
  double* accumulation;
  size_t neighbor_capacity;
  NonbondedWorkspace(size_t poses, size_t blocks, void* data = nullptr)
      : storage(data) {
    auto pairs = score::common::checked_triangular_size(
        blocks, true, "JAX native neighbors");
    neighbor_capacity = 1
                        + score::common::checked_dispatch_product(
                            poses, pairs, "JAX native neighbors");
    spheres = storage.take<Real>(poses * blocks * 4);
    neighbors = storage.take<int32_t>(neighbor_capacity);
    accumulation = storage.take<double>((Weighted ? 1 : 4) * poses);
  }
};

template <bool Weighted = false>
struct NonbondedKernel {
  static constexpr int channels = Weighted ? 1 : 4;
  // Every pose/channel is assigned from the zeroed double accumulator.
  static constexpr bool zero_energy = false;
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      ffi::AnyBuffer energy,
      const ffi::AnyBuffer* gradient) {
    if (a.size() != (Weighted ? 21 : 20))
      throw std::invalid_argument("invalid native nonbonded operand count");
    auto real = a[0].element_type();
    if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
      throw std::invalid_argument("nonbonded coordinates must be float32/64");
    auto integer = ffi::DataType::S32;
    require_shape(a[0], real, {-1, -1, 3});
    auto poses = a[0].dimensions()[0], atoms = a[0].dimensions()[1];
    require_shape(a[2], integer, {poses, -1});
    auto blocks = a[2].dimensions()[1], rots = poses * blocks;
    require_shape(a[1], integer, {rots});
    require_shape(a[3], integer, {rots});
    require_shape(a[4], integer, {rots});
    require_shape(a[5], integer, {poses, blocks});
    require_shape(a[6], integer, {poses, blocks, -1, 2});
    auto slots = a[6].dimensions()[2];
    require_shape(a[7], ffi::DataType::S8, {poses, blocks, slots, -1, -1});
    auto ports = a[7].dimensions()[3];
    if (!slots || a[7].dimensions()[4] != ports || ports > 12)
      throw std::invalid_argument("invalid native nonbonded connection layout");
    require_shape(a[8], integer, {-1});
    auto types = a[8].dimensions()[0];
    require_shape(a[9], integer, {types, -1});
    auto type_atoms = a[9].dimensions()[1];
    require_shape(a[10], integer, {types});
    require_shape(a[11], integer, {types, ports});
    require_shape(a[12], integer, {types, type_atoms, type_atoms});
    require_shape(a[13], integer, {types});
    require_shape(a[14], real, {-1, 13});
    require_shape(a[15], real, {1, 5});
    require_shape(a[16], real, {types, type_atoms});
    require_shape(a[17], integer, {types, type_atoms, type_atoms});
    require_shape(a[18], integer, {types, type_atoms, type_atoms});
    require_shape(a[19], real, {1, 11});
    if constexpr (Weighted) {
      require_shape(a[20], real, {4});
      require_shape(energy, real, {poses});
      if (gradient) require_shape(*gradient, real, {poses, atoms, 3});
    } else {
      require_shape(energy, real, {4, poses});
      if (gradient) require_shape(*gradient, real, {4, poses, atoms, 3});
    }
    if (poses * atoms > INT32_MAX || rots > INT32_MAX)
      throw std::invalid_argument("nonbonded flattened layout exceeds int32");
  }

  static size_t workspace_bytes(const std::vector<ffi::AnyBuffer>& a, bool) {
    auto poses = a[2].dimensions()[0], blocks = a[2].dimensions()[1];
    return a[0].element_type() == ffi::DataType::F32
               ? NonbondedWorkspace<float, Weighted>(poses, blocks)
                     .storage.size_bytes()
               : NonbondedWorkspace<double, Weighted>(poses, blocks)
                     .storage.size_bytes();
  }

  template <
      template <Device> class Ops,
      Device D,
      typename Real,
      bool Derivative>
  static void launch(
      const std::vector<ffi::AnyBuffer>& a,
      ffi::AnyBuffer energy,
      const ffi::AnyBuffer* gradient,
      ffi::AnyBuffer workspace,
      double reach) {
    using I = int32_t;
    using LJType = score::ljlk::potentials::LJLKTypeParams<Real>;
    using LJGlobal = score::ljlk::potentials::LJGlobalParams<Real>;
    using ElecGlobal = score::elec::potentials::ElecGlobalParams<Real>;
    static_assert(sizeof(LJType) == 13 * sizeof(Real));
    static_assert(sizeof(LJGlobal) == 5 * sizeof(Real));
    static_assert(sizeof(ElecGlobal) == 11 * sizeof(Real));
    auto poses = a[0].dimensions()[0], atoms = a[0].dimensions()[1];
    auto blocks = a[2].dimensions()[1], rots = poses * blocks;
    auto xyz = contiguous_view<Vec<Real, 3>, 1, D>(
        a[0].untyped_data(), {poses * atoms});
    auto types = contiguous_view<I, 1, D>(a[5].untyped_data(), {rots});
    NonbondedWorkspace<Real, Weighted> work(
        poses, blocks, workspace.untyped_data());
    auto spheres =
        contiguous_view<Real, 3, D>(work.spheres, {poses, blocks, 4});
    auto neighbors = contiguous_view<I, 1, D>(
        work.neighbors, {int64_t(work.neighbor_capacity)});
    ContextManager context;
    if (blocks == 1) {
      score::common::sphere_overlap::
          single_block_neighbor_candidates_native<Ops, D, I>(
              context, poses, neighbors);
    } else {
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
    // Whole-pose scoring has no rotamer dispatch or upstream-gradient operand.
    // Empty optional views are the canonical native mode contract.
    score::ljlk::potentials::
        ljlk_elec_native_forward<Derivative, Weighted, false, Ops, D, Real, I>(
            context,
            xyz,
            buffer_view<I, 1, D>(a[1]),
            buffer_view<I, 2, D>(a[2]),
            buffer_view<I, 1, D>(a[3]),
            types,
            buffer_view<I, 2, D>(a[2]),
            buffer_view<I, 4, D>(a[6]),
            buffer_view<int8_t, 5, D>(a[7]),
            buffer_view<I, 1, D>(a[8]),
            buffer_view<I, 2, D>(a[9]),
            buffer_view<I, 1, D>(a[10]),
            buffer_view<I, 2, D>(a[11]),
            buffer_view<I, 3, D>(a[12]),
            buffer_view<I, 1, D>(a[13]),
            buffer_view<LJType, 1, D>(a[14]),
            buffer_view<LJGlobal, 1, D>(a[15]),
            buffer_view<Real, 2, D>(a[16]),
            buffer_view<I, 3, D>(a[17]),
            buffer_view<I, 3, D>(a[18]),
            buffer_view<ElecGlobal, 1, D>(a[19]),
            neighbors,
            contiguous_view<I, 2, D>(nullptr, {0, 0}),
            Weighted ? buffer_view<Real, 1, D>(a[20])
                     : contiguous_view<Real, 1, D>(nullptr, {0}),
            contiguous_view<Real, 1, D>(nullptr, {0}),
            I(rots),
            contiguous_view<Real, 4, D>(
                energy.untyped_data(), {channels, poses, 1, 1}),
            contiguous_view<Vec<Real, 3>, 2, D>(
                gradient ? gradient->untyped_data() : nullptr,
                {channels, gradient ? poses * atoms : 0}),
            work.accumulation);
  }

  template <template <Device> class Ops, Device D, typename Real>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      ffi::AnyBuffer energy,
      const ffi::AnyBuffer* gradient,
      ffi::AnyBuffer workspace,
      double reach) {
    if (!std::isfinite(reach) || reach <= 0)
      throw std::invalid_argument(
          "nonbonded neighbor reach must be finite and positive");
    if (gradient)
      launch<Ops, D, Real, true>(a, energy, gradient, workspace, reach);
    else
      launch<Ops, D, Real, false>(a, energy, nullptr, workspace, reach);
  }
};
}  // namespace tmol::jax
