// Native water generation, LK-ball scoring, and their composed derivatives.
#pragma once
#include "workspace_ffi.hh"
#include <tmol/score/common/sphere_overlap_preparation.hh>
#include <tmol/score/lk_ball/potentials/gen_pose_waters_native.hh>
#include <tmol/score/lk_ball/potentials/lk_ball_pose_kernels.hh>

namespace tmol::jax {
template <typename Real, bool Weighted = false>
struct LKBallWorkspace {
  ByteWorkspace storage;
  Real *spheres, *waters, *water_gradient, *weights, *channel_energy;
  int32_t *neighbors, *lockstep;
  size_t neighbor_capacity;
  LKBallWorkspace(
      size_t poses,
      size_t blocks,
      size_t atoms,
      bool derivative,
      void* data = nullptr)
      : storage(data) {
    auto pairs = score::common::checked_triangular_size(
        blocks, true, "LK-ball neighbors");
    neighbor_capacity = 1
                        + score::common::checked_dispatch_product(
                            poses, pairs, "LK-ball neighbors");
    spheres = storage.take<Real>(poses * blocks * 4);
    neighbors = storage.take<int32_t>(neighbor_capacity);
    lockstep = storage.take<int32_t>(poses * blocks);
    waters = storage.take<Real>(poses * atoms * 4 * 3);
    water_gradient = storage.take<Real>(derivative ? poses * atoms * 4 * 3 : 0);
    weights = storage.take<Real>(derivative ? poses * 4 : 0);
    channel_energy = storage.take<Real>(Weighted ? poses * 4 : 0);
  }
};

template <bool Weighted = false>
struct LKBallKernelT {
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      ffi::AnyBuffer energy,
      const ffi::AnyBuffer* gradient) {
    if (a.size() != (Weighted ? 35 : 34))
      throw std::invalid_argument("LK-ball operand count mismatch");
    auto real = a[0].element_type();
    if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
      throw std::invalid_argument("LK-ball requires float32/64 coordinates");
    auto integer = ffi::DataType::S32;
    require_shape(a[0], real, {-1, -1, 3});
    auto poses = a[0].dimensions()[0], atoms = a[0].dimensions()[1];
    require_shape(a[4], integer, {poses, -1});
    auto blocks = a[4].dimensions()[1], rots = poses * blocks;
    for (int i : {1, 2, 3}) require_shape(a[i], integer, {rots});
    require_shape(a[5], integer, {poses, blocks});
    require_shape(a[6], integer, {poses, blocks, -1, 2});
    auto ports = a[6].dimensions()[2];
    if (ports > score::lk_ball::potentials::lk_max_n_conn)
      throw std::invalid_argument(
          "LK-ball native connection capacity exceeded");
    require_shape(a[7], integer, {-1});
    auto types = a[7].dimensions()[0];
    require_shape(a[8], integer, {types});
    require_shape(a[9], integer, {types, ports});
    require_shape(a[10], integer, {types});
    require_shape(a[11], integer, {types, -1, 3});
    require_shape(a[12], integer, {types, -1, 2});
    auto type_atoms = a[12].dimensions()[1];
    require_shape(a[13], integer, {types, type_atoms});
    require_shape(a[14], integer, {types, -1});
    auto tiles = a[14].dimensions()[1];
    require_shape(a[15], integer, {types, tiles});
    for (int i = 16; i <= 21; ++i)
      require_shape(a[i], integer, {types, tiles, 32});
    require_shape(a[22], real, {1, 4});
    for (int i = 23; i <= 25; ++i) require_shape(a[i], real, {-1});
    require_shape(a[26], integer, {poses, blocks, -1, 2});
    auto slots = a[26].dimensions()[2];
    if (!slots)
      throw std::invalid_argument("LK-ball near-block sentinel is missing");
    require_shape(
        a[27], ffi::DataType::S8, {poses, blocks, slots, ports, ports});
    require_shape(a[28], integer, {types, type_atoms, type_atoms});
    for (int i : {29, 30}) require_shape(a[i], integer, {types, tiles});
    require_shape(a[31], integer, {types, tiles, 32});
    require_shape(a[32], real, {types, tiles, 32, 9});
    require_shape(a[33], real, {1, 5});
    if constexpr (Weighted) {
      require_shape(a[34], real, {4});
      require_shape(energy, real, {poses});
      if (gradient) require_shape(*gradient, real, {poses, atoms, 3});
    } else {
      require_shape(energy, real, {4, poses});
      if (gradient) require_shape(*gradient, real, {4, poses, atoms, 3});
    }
    if (poses * atoms > INT32_MAX / 12 || rots > INT32_MAX
        || poses > INT32_MAX / 4)
      throw std::invalid_argument(
          "LK-ball native flattened layout exceeds int32");
  }

  static size_t workspace_bytes(
      const std::vector<ffi::AnyBuffer>& a, bool derivative) {
    auto poses = a[0].dimensions()[0], atoms = a[0].dimensions()[1];
    auto blocks = a[4].dimensions()[1];
    return a[0].element_type() == ffi::DataType::F32
               ? LKBallWorkspace<float, Weighted>(
                     poses, blocks, atoms, derivative)
                     .storage.size_bytes()
               : LKBallWorkspace<double, Weighted>(
                     poses, blocks, atoms, derivative)
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
    using namespace score::lk_ball::potentials;
    static_assert(sizeof(LKBallTypeParams<Real>) == 9 * sizeof(Real));
    static_assert(sizeof(LKBallGlobalParams<Real>) == 5 * sizeof(Real));
    static_assert(sizeof(LKBallWaterGenGlobalParams<Real>) == 4 * sizeof(Real));
    if (!std::isfinite(reach) || reach <= 0)
      throw std::invalid_argument(
          "LK-ball neighbor reach must be finite and positive");
    auto poses = a[0].dimensions()[0], atoms = a[0].dimensions()[1];
    auto blocks = a[4].dimensions()[1], rots = poses * blocks,
         count = poses * atoms;
    LKBallWorkspace<Real, Weighted> work(
        poses, blocks, atoms, gradient != nullptr, workspace.untyped_data());
    auto xyz =
        contiguous_view<Vec<Real, 3>, 1, D>(a[0].untyped_data(), {count});
    auto types = contiguous_view<I, 1, D>(a[4].untyped_data(), {rots});
    auto spheres =
        contiguous_view<Real, 3, D>(work.spheres, {poses, blocks, 4});
    auto neighbors = contiguous_view<I, 1, D>(
        work.neighbors, {int64_t(work.neighbor_capacity)});
    auto no_lockstep = contiguous_view<I, 2, D>(work.lockstep, {poses, blocks});
    auto waters = contiguous_view<Vec<Real, 3>, 2, D>(
        work.waters, {count, lk_max_n_water});
    ContextManager context;
    LAUNCH_BOX_32;
    auto lockstep = work.lockstep;
    auto water_data = work.waters;
    Ops<D>::template forall<launch_t>(
        context, rots, [=] TMOL_DEVICE_FUNC(int i) { lockstep[i] = -1; });
    Ops<D>::template forall<launch_t>(
        context, count * lk_max_n_water * 3, [=] TMOL_DEVICE_FUNC(int i) {
          water_data[i] = std::numeric_limits<Real>::quiet_NaN();
        });
    score::common::sphere_overlap::
        compute_block_spheres_native<Ops, D, Real, I>(
            context,
            xyz,
            buffer_view<I, 1, D>(a[1]),
            buffer_view<I, 1, D>(a[3]),
            buffer_view<I, 1, D>(a[2]),
            types,
            buffer_view<I, 1, D>(a[7]),
            spheres);
    score::common::sphere_overlap::
        detect_compact_block_neighbors<Ops, D, Real, I>::f(
            context,
            buffer_view<I, 2, D>(a[4]),
            spheres,
            neighbors,
            Real(reach));

    WaterGenCtx<Real, I, D> water{};
    water.rot_coords = xyz;
    water.rot_coord_offset = buffer_view<I, 1, D>(a[1]);
    water.pose_ind_for_rot = buffer_view<I, 1, D>(a[2]);
    water.block_ind_for_rot = buffer_view<I, 1, D>(a[3]);
    water.block_type_ind_for_rot = types;
    water.first_rot_for_block = buffer_view<I, 2, D>(a[5]);
    water.first_rot_block_type = buffer_view<I, 2, D>(a[4]);
    water.lockstep_group_for_block = no_lockstep;
    water.rot_offset_for_block = water.first_rot_for_block;
    water.pose_stack_inter_residue_connections =
        buffer_view<Vec<I, 2>, 3, D>(a[6]);
    water.block_type_n_atoms = buffer_view<I, 1, D>(a[7]);
    water.block_type_n_interblock_bonds = buffer_view<I, 1, D>(a[8]);
    water.block_type_atoms_forming_chemical_bonds = buffer_view<I, 2, D>(a[9]);
    water.block_type_n_all_bonds = buffer_view<I, 1, D>(a[10]);
    water.block_type_all_bonds = buffer_view<Vec<I, 3>, 2, D>(a[11]);
    water.block_type_atom_all_bond_ranges = buffer_view<Vec<I, 2>, 2, D>(a[12]);
    water.block_type_atom_is_hydrogen = buffer_view<I, 2, D>(a[13]);
    water.block_type_tile_n_donH = buffer_view<I, 2, D>(a[14]);
    water.block_type_tile_n_acc = buffer_view<I, 2, D>(a[15]);
    water.block_type_tile_donH_inds = buffer_view<I, 3, D>(a[16]);
    water.block_type_tile_don_hvy_inds = buffer_view<I, 3, D>(a[17]);
    water.block_type_tile_which_donH_for_hvy = buffer_view<I, 3, D>(a[18]);
    water.block_type_tile_acc_inds = buffer_view<I, 3, D>(a[19]);
    water.block_type_tile_hybridization = buffer_view<I, 3, D>(a[20]);
    water.block_type_tile_acc_n_attached_H = buffer_view<I, 3, D>(a[21]);
    water.global_params =
        buffer_view<LKBallWaterGenGlobalParams<Real>, 1, D>(a[22]);
    water.sp2_water_tors = buffer_view<Real, 1, D>(a[23]);
    water.sp3_water_tors = buffer_view<Real, 1, D>(a[24]);
    water.ring_water_tors = buffer_view<Real, 1, D>(a[25]);
    water.water_coords = waters;
    water.n_rots_total = rots;
    water_gen_forward<Ops, D, Real, I>(context, water);

    LKBallCtx<Real, I, D> score{};
    score.rot_coords = xyz;
    score.water_coords = waters;
    score.rot_coord_offset = water.rot_coord_offset;
    score.rot_offset_for_block = water.first_rot_for_block;
    score.block_type_ind_for_rot = types;
    score.block_ind_for_rot = water.block_ind_for_rot;
    score.pose_stack_inter_residue_connections =
        water.pose_stack_inter_residue_connections;
    score.pose_stack_near_blocks = buffer_view<I, 4, D>(a[26]);
    score.pose_stack_inter_block_bondsep = buffer_view<int8_t, 5, D>(a[27]);
    score.block_type_n_atoms = water.block_type_n_atoms;
    score.block_type_n_interblock_bonds = water.block_type_n_interblock_bonds;
    score.block_type_atoms_forming_chemical_bonds =
        water.block_type_atoms_forming_chemical_bonds;
    score.block_type_path_distance = buffer_view<I, 3, D>(a[28]);
    score.block_type_tile_n_polar_atoms = buffer_view<I, 2, D>(a[29]);
    score.block_type_tile_n_occluder_atoms = buffer_view<I, 2, D>(a[30]);
    score.block_type_tile_pol_occ_inds = buffer_view<I, 3, D>(a[31]);
    score.block_type_tile_lk_ball_params =
        buffer_view<LKBallTypeParams<Real>, 3, D>(a[32]);
    score.global_params = buffer_view<LKBallGlobalParams<Real>, 1, D>(a[33]);
    score.output = contiguous_view<Real, 4, D>(
        Weighted ? work.channel_energy : energy.untyped_data(),
        {4, poses, 1, 1});
    score.max_n_blocks = blocks;
    score.n_block_pairs = score::common::checked_triangular_size(
        blocks, true, "LK-ball block pairs");
    score.use_shared_compact_block_neighbors = true;
    lk_ball_pose_native<false, Ops, D, Real, I>(
        context, score, neighbors, poses);
    auto user_weights = Weighted ? buffer_view<Real, 1, D>(a[34])
                                 : contiguous_view<Real, 1, D>(nullptr, {0});
    if constexpr (Weighted) {
      auto channels = score.output;
      auto total = buffer_view<Real, 1, D>(energy);
      Ops<D>::template forall<launch_t>(
          context, poses, [=] TMOL_DEVICE_FUNC(int p) {
            Real value = 0;
            for (int c = 0; c < 4; ++c)
              value += user_weights[c] * channels[c][p][0][0];
            total[p] = value;
          });
    }
    if (!gradient) return;
    score.dTdV = contiguous_view<Real, 4, D>(work.weights, {4, poses, 1, 1});
    score.dV_d_water_coords = contiguous_view<Vec<Real, 3>, 2, D>(
        work.water_gradient, {count, lk_max_n_water});
    auto weights = work.weights;
    auto water_gradient = work.water_gradient;
    for (int channel = 0; channel < (Weighted ? 1 : 4); ++channel) {
      Ops<D>::template forall<launch_t>(
          context, 4 * poses, [=] TMOL_DEVICE_FUNC(int i) {
            weights[i] =
                Weighted ? user_weights[i / poses] : Real(i / poses == channel);
          });
      Ops<D>::template forall<launch_t>(
          context, count * lk_max_n_water * 3, [=] TMOL_DEVICE_FUNC(int i) {
            water_gradient[i] = 0;
          });
      score.dV_d_pose_coords = contiguous_view<Vec<Real, 3>, 1, D>(
          static_cast<Real*>(gradient->untyped_data()) + channel * count * 3,
          {count});
      lk_ball_pose_native<true, Ops, D, Real, I>(
          context, score, neighbors, poses);
      water_gen_backward<Ops, D, Real, I>(
          context, water, score.dV_d_water_coords, score.dV_d_pose_coords);
    }
  }
};
using LKBallKernel = LKBallKernelT<false>;
using LKBallWeightedKernel = LKBallKernelT<true>;
}  // namespace tmol::jax
