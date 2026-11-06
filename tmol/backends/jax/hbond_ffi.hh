// Typed XLA storage and dispatch for canonical hydrogen-bond physics.
#pragma once
#include "workspace_ffi.hh"
#include <tmol/score/common/sphere_overlap_preparation.hh>
#include <tmol/score/hbond/potentials/native_hbond_pose.hh>
#include <tmol/score/hbond/potentials/native_hbond_bases.hh>

namespace tmol::jax {
template <typename Real>
struct HBondWorkspace {
  ByteWorkspace storage;
  Real *spheres, *derived_coords;
  int32_t *neighbors, *derived_indices;
  size_t neighbor_capacity;
  HBondWorkspace(
      size_t poses, size_t blocks, size_t atoms, void* data = nullptr)
      : storage(data) {
    auto pairs =
        score::common::checked_triangular_size(blocks, true, "HB neighbors");
    neighbor_capacity =
        1
        + score::common::checked_dispatch_product(poses, pairs, "HB neighbors");
    spheres = storage.take<Real>(poses * blocks * 4);
    neighbors = storage.take<int32_t>(neighbor_capacity);
    derived_coords = storage.take<Real>(poses * atoms * 9);
    derived_indices = storage.take<int32_t>(poses * atoms * 3);
  }
};

struct HBondKernel {
  // Every live scratch element is overwritten below; padding spheres and
  // unused compact-list entries are never read by the common dispatch.
  static constexpr bool zero_workspace = false;
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      ffi::AnyBuffer energy,
      const ffi::AnyBuffer* gradient) {
    if (a.size() != 27)
      throw std::invalid_argument("HBond expects 27 operands");
    auto real = a[0].element_type();
    if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
      throw std::invalid_argument("HBond coordinates must be float32/64");
    auto integer = ffi::DataType::S32;
    require_shape(a[0], real, {-1, -1, 3});
    auto poses = a[0].dimensions()[0], atoms = a[0].dimensions()[1];
    require_shape(a[3], integer, {poses, -1});
    auto blocks = a[3].dimensions()[1], rots = poses * blocks;
    require_shape(a[2], integer, {poses, blocks});
    for (int i : {1, 4, 5}) require_shape(a[i], integer, {rots});
    require_shape(a[6], integer, {poses, blocks, -1, 2});
    auto ports = a[6].dimensions()[2];
    if (ports > score::hbond::potentials::MAX_N_CONN)
      throw std::invalid_argument("HBond native connection capacity exceeded");
    require_shape(a[7], integer, {poses, blocks, -1, 2});
    auto slots = a[7].dimensions()[2];
    if (!slots)
      throw std::invalid_argument("HBond near-block sentinel is missing");
    require_shape(
        a[8], ffi::DataType::S8, {poses, blocks, slots, ports, ports});
    require_shape(a[9], integer, {-1});
    auto types = a[9].dimensions()[0];
    for (int i : {10, 12}) require_shape(a[i], integer, {types});
    require_shape(a[11], integer, {types, ports});
    require_shape(a[13], integer, {types, -1, 3});
    require_shape(a[14], integer, {types, -1, 2});
    auto type_atoms = a[14].dimensions()[1];
    require_shape(a[15], integer, {types, type_atoms});
    require_shape(a[16], integer, {types, type_atoms, type_atoms});
    require_shape(a[17], integer, {types, -1});
    auto tiles = a[17].dimensions()[1];
    require_shape(a[18], integer, {types, tiles});
    for (int i = 19; i <= 23; ++i)
      require_shape(a[i], integer, {types, tiles, 32});
    require_shape(a[24], real, {-1, -1, 3});
    // Exact native double coefficients survive a JAX process with x64 disabled.
    require_shape(
        a[25],
        ffi::DataType::U8,
        {a[24].dimensions()[0], a[24].dimensions()[1], 48 * 8});
    if (reinterpret_cast<uintptr_t>(a[25].untyped_data()) % alignof(double))
      throw std::invalid_argument(
          "HBond polynomial storage is not double-aligned");
    require_shape(a[26], real, {1, 6});
    require_shape(energy, real, {poses});
    if (gradient) require_shape(*gradient, real, {poses, atoms, 3});
    if (poses * atoms > INT32_MAX / 9 || rots > INT32_MAX)
      throw std::invalid_argument("HBond flattened layout exceeds int32");
  }

  static size_t workspace_bytes(const std::vector<ffi::AnyBuffer>& a, bool) {
    auto poses = a[0].dimensions()[0], atoms = a[0].dimensions()[1];
    auto blocks = a[3].dimensions()[1];
    return a[0].element_type() == ffi::DataType::F32
               ? HBondWorkspace<float>(poses, blocks, atoms)
                     .storage.size_bytes()
               : HBondWorkspace<double>(poses, blocks, atoms)
                     .storage.size_bytes();
  }

  template <template <Device> class Ops, Device D, typename Real>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      ffi::AnyBuffer energy,
      const ffi::AnyBuffer* gradient,
      ffi::AnyBuffer workspace) {
    using I = int32_t;
    using namespace score::hbond::potentials;
    static_assert(sizeof(HBondPolynomials<double>) == 48 * sizeof(double));
    static_assert(sizeof(HBondPairParams<Real>) == 3 * sizeof(Real));
    static_assert(sizeof(HBondGlobalParams<Real>) == 6 * sizeof(Real));
    auto poses = a[0].dimensions()[0], atoms = a[0].dimensions()[1];
    auto blocks = a[3].dimensions()[1], rots = poses * blocks,
         count = poses * atoms;
    HBondWorkspace<Real> work(poses, blocks, atoms, workspace.untyped_data());
    auto xyz =
        contiguous_view<Vec<Real, 3>, 1, D>(a[0].untyped_data(), {count});
    auto types = contiguous_view<I, 1, D>(a[3].untyped_data(), {rots});
    auto spheres =
        contiguous_view<Real, 3, D>(work.spheres, {poses, blocks, 4});
    auto neighbors = contiguous_view<I, 1, D>(
        work.neighbors, {int64_t(work.neighbor_capacity)});
    auto derived =
        contiguous_view<Vec<Real, 3>, 2, D>(work.derived_coords, {count, 3});
    auto indices = contiguous_view<I, 2, D>(work.derived_indices, {count, 3});
    ContextManager context;
    LAUNCH_BOX_32;
    auto coordinates = work.derived_coords;
    auto sources = work.derived_indices;
    auto neighbor_data = work.neighbors;
    Ops<D>::template forall<launch_t>(
        context, std::max<int64_t>(count * 9, 1), [=] TMOL_DEVICE_FUNC(int i) {
          if (i == 0) neighbor_data[0] = 0;
          if (i < count * 9)
            coordinates[i] = std::numeric_limits<Real>::quiet_NaN();
          if (i < count * 3) sources[i] = -1;
        });
    score::common::sphere_overlap::
        compute_block_spheres_native<Ops, D, Real, I>(
            context,
            xyz,
            buffer_view<I, 1, D>(a[1]),
            buffer_view<I, 1, D>(a[4]),
            buffer_view<I, 1, D>(a[5]),
            types,
            buffer_view<I, 1, D>(a[9]),
            spheres);
    score::common::sphere_overlap::
        detect_compact_block_neighbors<Ops, D, Real, I>::f(
            context,
            buffer_view<I, 2, D>(a[3]),
            spheres,
            neighbors,
            Real(hbond_neighbor_reach()));

    HBondBasesCtxT<score::common::TViewBinder<D>::template View, Real, I>
        bases{};
    bases.rot_coords = xyz;
    bases.rot_coord_offset = buffer_view<I, 1, D>(a[1]);
    bases.first_rot_for_block = buffer_view<I, 2, D>(a[2]);
    bases.first_rot_block_type = buffer_view<I, 2, D>(a[3]);
    bases.block_ind_for_rot = buffer_view<I, 1, D>(a[4]);
    bases.pose_ind_for_rot = buffer_view<I, 1, D>(a[5]);
    bases.block_type_ind_for_rot = types;
    bases.pose_stack_inter_residue_connections =
        buffer_view<Vec<I, 2>, 3, D>(a[6]);
    bases.block_type_n_atoms = buffer_view<I, 1, D>(a[9]);
    bases.block_type_n_interblock_bonds = buffer_view<I, 1, D>(a[10]);
    bases.block_type_atoms_forming_chemical_bonds = buffer_view<I, 2, D>(a[11]);
    bases.block_type_n_all_bonds = buffer_view<I, 1, D>(a[12]);
    bases.block_type_all_bonds = buffer_view<Vec<I, 3>, 2, D>(a[13]);
    bases.block_type_atom_all_bond_ranges = buffer_view<Vec<I, 2>, 2, D>(a[14]);
    bases.block_type_atom_is_hydrogen = buffer_view<I, 2, D>(a[15]);
    bases.block_type_tile_n_donH = buffer_view<I, 2, D>(a[17]);
    bases.block_type_tile_n_acc = buffer_view<I, 2, D>(a[18]);
    bases.block_type_tile_donH_inds = buffer_view<I, 3, D>(a[19]);
    bases.block_type_tile_acc_inds = buffer_view<I, 3, D>(a[20]);
    bases.block_type_tile_hybridization = buffer_view<I, 3, D>(a[23]);
    bases.derived_coords = derived;
    bases.derived_atom_inds = indices;
    launch_hbond_bases<Ops, D, Real, I>(context, bases);

    HBondCtx<Real, I, D, 4, 3, 2> score{};
    score.rot_coords = xyz;
    score.rot_coord_offset = bases.rot_coord_offset;
    score.block_ind_for_rot = bases.block_ind_for_rot;
    score.block_type_ind_for_rot = types;
    score.rot_offset_for_block = bases.first_rot_for_block;
    score.pose_stack_near_blocks = buffer_view<I, 4, D>(a[7]);
    score.pose_stack_inter_block_bondsep = buffer_view<int8_t, 5, D>(a[8]);
    score.block_type_n_atoms = bases.block_type_n_atoms;
    score.block_type_n_interblock_bonds = bases.block_type_n_interblock_bonds;
    score.block_type_atoms_forming_chemical_bonds =
        bases.block_type_atoms_forming_chemical_bonds;
    score.block_type_path_distance = buffer_view<I, 3, D>(a[16]);
    score.block_type_tile_n_donH = bases.block_type_tile_n_donH;
    score.block_type_tile_n_acc = bases.block_type_tile_n_acc;
    score.block_type_tile_donH_inds = bases.block_type_tile_donH_inds;
    score.block_type_tile_acc_inds = bases.block_type_tile_acc_inds;
    score.block_type_tile_donor_type = buffer_view<I, 3, D>(a[21]);
    score.block_type_tile_acceptor_type = buffer_view<I, 3, D>(a[22]);
    score.block_type_tile_hybridization = bases.block_type_tile_hybridization;
    score.pair_params = buffer_view<HBondPairParams<Real>, 2, D>(a[24]);
    score.pair_polynomials = contiguous_view<HBondPolynomials<double>, 2, D>(
        a[25].untyped_data(), {a[24].dimensions()[0], a[24].dimensions()[1]});
    score.global_params = buffer_view<HBondGlobalParams<Real>, 1, D>(a[26]);
    score.derived_coords = derived;
    score.derived_atom_inds = indices;
    score.output =
        contiguous_view<Real, 4, D>(energy.untyped_data(), {1, poses, 1, 1});
    score.dTdV = score.output;
    if (gradient)
      score.dV_dcoords = contiguous_view<Vec<Real, 3>, 2, D>(
          gradient->untyped_data(), {1, count});
    score.max_n_blocks = blocks;
    score.output_block_pair_energies = false;
    score.accumulate_derivs = gradient != nullptr;
    score.max_n_upper_triangle_inds =
        score::common::checked_triangular_size(blocks, true, "HB pairs");
    score.use_shared_compact_block_neighbors = true;
    auto launch = [&](auto evaluate) {
      score::common::sphere_overlap::
          launch_precomputed_block_neighbors<Ops, D, launch_t, I>(
              context, neighbors, poses, blocks, evaluate);
    };
    launch_hbond_pose<Ops, D, Real, I>(
        score, poses, rots, work.neighbor_capacity, true, launch);
  }
};
}  // namespace tmol::jax
