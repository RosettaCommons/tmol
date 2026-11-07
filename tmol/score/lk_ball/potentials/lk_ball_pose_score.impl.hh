#pragma once

#include <tmol/utility/portable/reduce_ops.hh>
#include <tmol/utility/portable/math.hh>
#include <tmol/utility/portable/matrix.hh>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/counting.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/sphere_overlap.impl.hh>
#include <tmol/score/common/tile_atom_pair_evaluation.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/upper_triangle_indices.hh>
#include <tmol/score/common/warp_segreduce.hh>
#include <tmol/score/common/warp_stride_reduce.hh>

#include <tmol/score/lk_ball/potentials/lk_ball.hh>
#include <tmol/score/lk_ball/potentials/lk_ball_ctx.hh>
#include <tmol/score/common/tview_binder.hh>
#include <tmol/score/lk_ball/potentials/lk_ball_pose_kernels.hh>
#include <tmol/backends/metal/score/common/mps_launch.hh>
#ifdef WITH_MPS
#include <tmol/backends/metal/score/lk_ball/potentials/lk_ball_mps_kernels_source.h>
#endif
#include <tmol/score/lk_ball/potentials/params.hh>

// Operator definitions; safe for CPU compilation

#include <chrono>
#include <type_traits>

// Definitions for lk_tile_size and MAX_N_WATERS
// Shared between this file and gen_pose_waters.impl.hh
#include <tmol/score/lk_ball/potentials/constants.hh>

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
class LKBallPoseScoreDispatch {
 public:
  static auto forward(
      ContextManager& mgr,
      TView<Vec<Real, 3>, 1, Dev> rot_coords,
      TView<Int, 1, Dev> rot_coord_offset,
      TView<Int, 1, Dev> pose_ind_for_atom,
      TView<Int, 2, Dev> first_rot_for_block,
      TView<Int, 2, Dev> first_rot_block_type,
      TView<Int, 1, Dev> block_ind_for_rot,
      TView<Int, 1, Dev> pose_ind_for_rot,
      TView<Int, 1, Dev> block_type_ind_for_rot,
      TView<Int, 1, Dev> n_rots_for_pose,
      TView<Int, 1, Dev> rot_offset_for_pose,
      TView<Int, 2, Dev> n_rots_for_block,
      TView<Int, 2, Dev> rot_offset_for_block,
      Int max_n_rots_per_pose,

      // For determining which atoms to retrieve from neighboring
      // residues we have to know how the blocks in the Pose
      // are connected
      TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_residue_connections,

      // InterBlockBondsep: [pose, block1, slot, (block2, min separation)]
      // and [pose, block1, slot, conn1, conn2]
      TView<Int, 4, Dev> pose_stack_near_blocks,
      TView<int8_t, 5, Dev> pose_stack_inter_block_bondsep,

      //////////////////////
      // Chemical properties
      // how many atoms for a given block
      // Dimsize n_block_types
      TView<Int, 1, Dev> block_type_n_atoms,

      // how many inter-block chemical bonds are there
      // Dimsize: n_block_types
      TView<Int, 1, Dev> block_type_n_interblock_bonds,

      // what atoms form the inter-block chemical bonds
      // Dimsize: n_block_types x max_n_interblock_bonds
      // TO DO: Rename since lots of atoms form chemical bonds
      TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds,

      TView<Int, 2, Dev> block_type_tile_n_polar_atoms,
      TView<Int, 2, Dev> block_type_tile_n_occluder_atoms,
      TView<Int, 3, Dev> block_type_tile_pol_occ_inds,
      TView<LKBallTypeParams<Real>, 3, Dev> block_type_tile_lk_ball_params,

      // How many chemical bonds separate all pairs of atoms
      // within each block type?
      // Dimsize: n_block_types x max_n_atoms x max_n_atoms
      TView<Int, 3, Dev> block_type_path_distance,

      //////////////////////

      // LKBall potential parameters
      TView<LKBallGlobalParams<Real>, 1, Dev> global_params,
      Real max_dis,
      TView<Vec<Real, 3>, 2, Dev> water_coords,
      TView<Int, 1, Dev> shared_compact_block_neighbors,
      bool output_block_pair_energies)
      -> std::tuple<TPack<Real, 4, Dev>, TPack<Int, 3, Dev>> {
    using tmol::score::common::accumulate;
    using Accum = std::conditional_t<
        Dev == Device::CUDA && std::is_same_v<Real, float>,
        double,
        Real>;
    using Real3 = Vec<Real, 3>;

    int const n_rots = rot_coord_offset.size(0);
    int const max_n_blocks = first_rot_for_block.size(1);
    int const n_poses = n_rots_for_pose.size(0);
    int const max_n_pose_atoms = water_coords.size(1);
    int const max_n_conn = pose_stack_inter_residue_connections.size(2);
    int const n_block_types = block_type_n_atoms.size(0);
    int const max_n_block_atoms = block_type_path_distance.size(1);
    int const max_n_interblock_bonds =
        block_type_atoms_forming_chemical_bonds.size(1);
    int const max_n_tiles = block_type_tile_pol_occ_inds.size(1);
    int const n_block_pairs = score::common::checked_triangular_size(
        max_n_blocks, true, "LK-ball block-pair dispatch");

    bool const use_shared_compact_block_neighbors =
        shared_compact_block_neighbors.size(0) != 0;
    TPack<Real, 3, Dev> scratch_rot_spheres_t;
    auto scratch_rot_neighbors_t = TPack<Int, 3, Dev>::empty({0, 0, 0});
    TView<Real, 3, Dev> scratch_rot_spheres;
    TView<Int, 3, Dev> scratch_rot_neighbors;
    if (!use_shared_compact_block_neighbors) {
      scratch_rot_spheres_t =
          Dev == Device::CPU
              ? TPack<Real, 3, Dev>::zeros({n_poses, max_n_blocks, 4})
              : TPack<Real, 3, Dev>::empty({n_poses, max_n_blocks, 4});
      scratch_rot_neighbors_t =
          Dev == Device::CPU
              ? TPack<Int, 3, Dev>::zeros({n_poses, max_n_blocks, max_n_blocks})
              : TPack<Int, 3, Dev>::empty(
                    {n_poses, max_n_blocks, max_n_blocks});
      scratch_rot_spheres = scratch_rot_spheres_t.view;
      scratch_rot_neighbors = scratch_rot_neighbors_t.view;
    }

    TPack<Accum, 4, Dev> output_t;
    if (output_block_pair_energies) {
      output_t = TPack<Accum, 4, Dev>::zeros(
          {n_lk_ball_score_types, n_poses, max_n_blocks, max_n_blocks});
    } else {
      output_t =
          TPack<Accum, 4, Dev>::zeros({n_lk_ball_score_types, n_poses, 1, 1});
    }
    auto output = output_t.view;

    // A 16-warp occupancy target balances register pressure and occupancy
    // for these scoring kernels.
    LAUNCH_BOX_32_OCC(16);
    // Define nt and reduce_t
    CTA_REAL_REDUCE_T_TYPEDEF;

    // State the kernel body reads, named explicitly rather than captured
    // implicitly by [=]. The lambda below closes over this one object, and
    // the Metal launch binds the same fields positionally; see lk_ball_ctx.hh.
    LKBallCtx<Real, Int, Dev, 4, Real, Accum> ctx;
    ctx.rot_coords = rot_coords;
    ctx.water_coords = water_coords;
    ctx.rot_coord_offset = rot_coord_offset;
    ctx.rot_offset_for_block = rot_offset_for_block;
    ctx.block_type_ind_for_rot = block_type_ind_for_rot;
    ctx.pose_stack_inter_residue_connections =
        pose_stack_inter_residue_connections;
    ctx.pose_stack_near_blocks = pose_stack_near_blocks;
    ctx.pose_stack_inter_block_bondsep = pose_stack_inter_block_bondsep;
    ctx.scratch_rot_neighbors = scratch_rot_neighbors;
    ctx.block_type_n_atoms = block_type_n_atoms;
    ctx.block_type_n_interblock_bonds = block_type_n_interblock_bonds;
    ctx.block_type_atoms_forming_chemical_bonds =
        block_type_atoms_forming_chemical_bonds;
    ctx.block_type_path_distance = block_type_path_distance;
    ctx.block_type_tile_n_polar_atoms = block_type_tile_n_polar_atoms;
    ctx.block_type_tile_n_occluder_atoms = block_type_tile_n_occluder_atoms;
    ctx.block_type_tile_pol_occ_inds = block_type_tile_pol_occ_inds;
    ctx.block_type_tile_lk_ball_params = block_type_tile_lk_ball_params;
    ctx.global_params = global_params;
    ctx.output = output;
    ctx.max_n_blocks = max_n_blocks;
    ctx.n_block_pairs = n_block_pairs;
    ctx.output_block_pair_energies = output_block_pair_energies;
    ctx.use_shared_compact_block_neighbors = use_shared_compact_block_neighbors;

    auto eval_energies_by_block =
        lk_ball_pose_evaluator<false, DeviceDispatch, Dev, Real, Int, launch_t>(
            ctx);

    // Build block neighbors before dispatching LK-ball pair scoring.
    if (!use_shared_compact_block_neighbors) {
      score::common::sphere_overlap::
          compute_block_spheres<DeviceDispatch, Dev, Real, Int>::f(
              mgr,
              rot_coords,
              rot_coord_offset,
              block_ind_for_rot,
              pose_ind_for_rot,
              block_type_ind_for_rot,
              block_type_n_atoms,
              scratch_rot_spheres);

      score::common::sphere_overlap::
          detect_block_neighbors<DeviceDispatch, Dev, Real, Int>::f(
              mgr,
              first_rot_block_type,
              scratch_rot_spheres,
              scratch_rot_neighbors,
              max_dis);
    }
#ifdef __NVCC__
    if (!output_block_pair_energies && use_shared_compact_block_neighbors) {
      launch_lk_ball_compact_pose_pair_workgroups<
          DeviceDispatch,
          Dev,
          launch_t,
          Int>(
          mgr,
          shared_compact_block_neighbors,
          n_poses,
          max_n_blocks,
          eval_energies_by_block);
    } else if (
        !output_block_pair_energies
        && score::common::sphere_overlap::should_compact_block_neighbors(
            n_poses, max_n_blocks, false)) {
      auto eval_all = ([=] TMOL_DEVICE_FUNC(int cta) {
        eval_energies_by_block(
            cta, TilePairModeTag<common::TilePairMode::InterAndIntra>{});
      });
      score::common::sphere_overlap::
          launch_compact_block_neighbors<DeviceDispatch, Dev, launch_t, Int>(
              mgr, scratch_rot_neighbors, eval_all);
    } else {
      launch_lk_ball_pose_pair_workgroups<DeviceDispatch, Dev, launch_t>(
          mgr, n_poses, max_n_blocks, eval_energies_by_block);
    }
#else
    if (Dev == tmol::Device::MPS) {
      // Metal cannot launch a device lambda -- there is nowhere for a closure
      // to live, and entry points are compiled by a separate toolchain -- so
      // the same kernel body is reached through a named entry rather than
      // through eval_energies_by_block. The context that lambda closes over
      // is the same object either way.
      //
      // This is tested ahead of the compact-block-neighbor path because that
      // launches through DeviceDispatch, which on this backend would run the
      // kernel on the CPU.
      tmol::score::common::mps_launch_if<Dev>(
          TMOL_MPS_SOURCE(kLKBallMetalShaderSource),
          "lk_ball_eval_energies_by_block_f32",
          ctx,
          int64_t(n_poses) * int64_t(n_block_pairs),
          true);
    } else if (
        !output_block_pair_energies && use_shared_compact_block_neighbors) {
      score::common::sphere_overlap::launch_precomputed_block_neighbors<
          DeviceDispatch,
          Dev,
          launch_t,
          Int>(
          mgr,
          shared_compact_block_neighbors,
          n_poses,
          max_n_blocks,
          eval_energies_by_block);
    } else {
      launch_lk_ball_pose_pair_workgroups<DeviceDispatch, Dev, launch_t>(
          mgr, n_poses, max_n_blocks, eval_energies_by_block);
    }
#endif

    return {output_t.template cast<Real>(), scratch_rot_neighbors_t};
  }

  static auto backward(
      ContextManager& mgr,
      TView<Vec<Real, 3>, 1, Dev> rot_coords,
      TView<Vec<Real, 3>, 2, Dev> water_coords,
      TView<Int, 1, Dev> rot_coord_offset,
      TView<Int, 1, Dev> pose_ind_for_atom,
      TView<Int, 2, Dev> first_rot_for_block,
      TView<Int, 2, Dev> first_rot_block_type,
      TView<Int, 1, Dev> block_ind_for_rot,
      TView<Int, 1, Dev> pose_ind_for_rot,
      TView<Int, 1, Dev> block_type_ind_for_rot,
      TView<Int, 1, Dev> n_rots_for_pose,
      TView<Int, 1, Dev> rot_offset_for_pose,
      TView<Int, 2, Dev> n_rots_for_block,
      TView<Int, 2, Dev> rot_offset_for_block,
      Int max_n_rots_per_pose,

      // For determining which atoms to retrieve from neighboring
      // residues we have to know how the blocks in the Pose
      // are connected
      TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_residue_connections,

      // InterBlockBondsep: [pose, block1, slot, (block2, min separation)]
      // and [pose, block1, slot, conn1, conn2]
      TView<Int, 4, Dev> pose_stack_near_blocks,
      TView<int8_t, 5, Dev> pose_stack_inter_block_bondsep,

      //////////////////////
      // Chemical properties
      // how many atoms for a given block
      // Dimsize n_block_types
      TView<Int, 1, Dev> block_type_n_atoms,

      // how many inter-block chemical bonds are there
      // Dimsize: n_block_types
      TView<Int, 1, Dev> block_type_n_interblock_bonds,

      // what atoms form the inter-block chemical bonds
      // Dimsize: n_block_types x max_n_interblock_bonds
      TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds,

      TView<Int, 2, Dev> block_type_tile_n_polar_atoms,
      TView<Int, 2, Dev> block_type_tile_n_occluder_atoms,
      TView<Int, 3, Dev> block_type_tile_pol_occ_inds,
      TView<LKBallTypeParams<Real>, 3, Dev> block_type_tile_lk_ball_params,

      // How many chemical bonds separate all pairs of atoms
      // within each block type?
      // Dimsize: n_block_types x max_n_atoms x max_n_atoms
      TView<Int, 3, Dev> block_type_path_distance,

      //////////////////////

      // LKBall potential parameters
      TView<LKBallGlobalParams<Real>, 1, Dev> global_params,
      TView<Int, 3, Dev> scratch_rot_neighbors,  // from forward pass
      TView<Int, 1, Dev> compact_block_neighbors,
      TView<Real, 4, Dev> dTdV,
      bool block_pair_scoring,
      bool allow_split_pairs)
      -> std::tuple<TPack<Vec<Real, 3>, 1, Dev>, TPack<Vec<Real, 3>, 2, Dev>> {
    // std::cout << "d lkball start" << std::endl;
    using tmol::score::common::accumulate;
    using Accum = std::conditional_t<
        Dev == Device::CUDA && std::is_same_v<Real, float>,
        double,
        Real>;
    using Real3 = Vec<Real, 3>;

    int const n_rots = rot_coord_offset.size(0);
    int const n_poses = n_rots_for_pose.size(0);
    int const max_n_blocks = first_rot_for_block.size(1);
    int const n_atoms = water_coords.size(0);
    int const max_n_conn = pose_stack_inter_residue_connections.size(2);
    int const n_block_types = block_type_n_atoms.size(0);
    int const max_n_block_atoms = block_type_path_distance.size(1);
    int const max_n_interblock_bonds =
        block_type_atoms_forming_chemical_bonds.size(1);
    int const max_n_tiles = block_type_tile_pol_occ_inds.size(1);
    int const n_block_pairs = score::common::checked_triangular_size(
        max_n_blocks, true, "LK-ball derivative block-pair dispatch");
    bool const use_compact_block_neighbors =
        compact_block_neighbors.size(0) != 0;

    auto dV_d_pose_coords_t = TPack<Vec<Accum, 3>, 1, Dev>::zeros({n_atoms});
    auto dV_d_pose_coords = dV_d_pose_coords_t.view;

    auto dV_d_water_coords_t =
        TPack<Vec<Accum, 3>, 2, Dev>::zeros({n_atoms, lk_max_n_water});
    auto dV_d_water_coords = dV_d_water_coords_t.view;

    LKBallCtx<Real, Int, Dev, 4, Accum> ctx;
    ctx.rot_coords = rot_coords;
    ctx.water_coords = water_coords;
    ctx.rot_coord_offset = rot_coord_offset;
    ctx.rot_offset_for_block = rot_offset_for_block;
    ctx.block_type_ind_for_rot = block_type_ind_for_rot;
    ctx.block_ind_for_rot = block_ind_for_rot;
    ctx.pose_stack_inter_residue_connections =
        pose_stack_inter_residue_connections;
    ctx.pose_stack_near_blocks = pose_stack_near_blocks;
    ctx.pose_stack_inter_block_bondsep = pose_stack_inter_block_bondsep;
    ctx.scratch_rot_neighbors = scratch_rot_neighbors;
    ctx.block_type_n_atoms = block_type_n_atoms;
    ctx.block_type_n_interblock_bonds = block_type_n_interblock_bonds;
    ctx.block_type_atoms_forming_chemical_bonds =
        block_type_atoms_forming_chemical_bonds;
    ctx.block_type_path_distance = block_type_path_distance;
    ctx.block_type_tile_n_polar_atoms = block_type_tile_n_polar_atoms;
    ctx.block_type_tile_n_occluder_atoms = block_type_tile_n_occluder_atoms;
    ctx.block_type_tile_pol_occ_inds = block_type_tile_pol_occ_inds;
    ctx.block_type_tile_lk_ball_params = block_type_tile_lk_ball_params;
    ctx.global_params = global_params;
    ctx.dTdV = dTdV;
    ctx.dV_d_pose_coords = dV_d_pose_coords;
    ctx.dV_d_water_coords = dV_d_water_coords;
    ctx.max_n_blocks = max_n_blocks;
    ctx.n_block_pairs = n_block_pairs;
    ctx.block_pair_scoring = block_pair_scoring;
    // The derivative kernel neither reads nor writes energies.
    ctx.output_block_pair_energies = false;
    ctx.use_shared_compact_block_neighbors = use_compact_block_neighbors;
    if (Dev == tmol::Device::MPS) {
      // Metal has nowhere for a closure to live, so the identical body is
      // reached through a named entry point instead. Tested ahead of the
      // compaction paths below, which launch through DeviceDispatch and would
      // run the kernel on the CPU for this backend.
      tmol::score::common::mps_launch_if<Dev>(
          TMOL_MPS_SOURCE(kLKBallMetalShaderSource),
          "lk_ball_eval_derivs_f32",
          ctx,
          int64_t(n_poses) * int64_t(n_block_pairs),
          true);
      return {
          dV_d_pose_coords_t.template cast<Vec<Real, 3>>(),
          dV_d_water_coords_t.template cast<Vec<Real, 3>>()};
    }

    lk_ball_pose_native<true, DeviceDispatch, Dev, Real, Int>(
        mgr, ctx, compact_block_neighbors, n_poses, allow_split_pairs);

    return {
        dV_d_pose_coords_t.template cast<Vec<Real, 3>>(),
        dV_d_water_coords_t.template cast<Vec<Real, 3>>()};
  }
};

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
class LKBallRotamerScoreDispatch {
 public:
  static auto forward(
      ContextManager& mgr,
      TView<Vec<Real, 3>, 1, Dev> rot_coords,
      TView<Int, 1, Dev> rot_coord_offset,
      TView<Int, 1, Dev> pose_ind_for_atom,
      TView<Int, 2, Dev> first_rot_for_block,
      TView<Int, 2, Dev> first_rot_block_type,
      TView<Int, 1, Dev> block_ind_for_rot,
      TView<Int, 1, Dev> pose_ind_for_rot,
      TView<Int, 1, Dev> block_type_ind_for_rot,
      TView<Int, 1, Dev> n_rots_for_pose,
      TView<Int, 1, Dev> rot_offset_for_pose,
      TView<Int, 2, Dev> n_rots_for_block,
      TView<Int, 2, Dev> rot_offset_for_block,
      // [n_poses, max_n_blocks]; blocks sharing an id >= 0 move in
      // lockstep, so only matching rotamer indices ever coexist
      TView<Int, 2, Dev> lockstep_group_for_block,
      Int max_n_rots_per_pose,

      // For determining which atoms to retrieve from neighboring
      // residues we have to know how the blocks in the Pose
      // are connected
      TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_residue_connections,

      // InterBlockBondsep: [pose, block1, slot, (block2, min separation)]
      // and [pose, block1, slot, conn1, conn2]
      TView<Int, 4, Dev> pose_stack_near_blocks,
      TView<int8_t, 5, Dev> pose_stack_inter_block_bondsep,

      //////////////////////
      // Chemical properties
      // how many atoms for a given block
      // Dimsize n_block_types
      TView<Int, 1, Dev> block_type_n_atoms,

      // how many inter-block chemical bonds are there
      // Dimsize: n_block_types
      TView<Int, 1, Dev> block_type_n_interblock_bonds,

      // what atoms form the inter-block chemical bonds
      // Dimsize: n_block_types x max_n_interblock_bonds
      // TO DO: Rename since lots of atoms form chemical bonds
      TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds,

      TView<Int, 2, Dev> block_type_tile_n_polar_atoms,
      TView<Int, 2, Dev> block_type_tile_n_occluder_atoms,
      TView<Int, 3, Dev> block_type_tile_pol_occ_inds,
      TView<LKBallTypeParams<Real>, 3, Dev> block_type_tile_lk_ball_params,

      // How many chemical bonds separate all pairs of atoms
      // within each block type?
      // Dimsize: n_block_types x max_n_atoms x max_n_atoms
      TView<Int, 3, Dev> block_type_path_distance,

      //////////////////////

      // LKBall potential parameters
      TView<LKBallGlobalParams<Real>, 1, Dev> global_params,
      Real max_dis,
      TView<Vec<Real, 3>, 2, Dev> water_coords,
      bool output_block_pair_energies,
      TPack<Int, 2, Dev> shared_dispatch_indices)
      -> std::tuple<TPack<Real, 2, Dev>, TPack<Int, 2, Dev>> {
    using tmol::score::common::accumulate;
    using Accum = std::conditional_t<
        Dev == Device::CUDA && std::is_same_v<Real, float>,
        double,
        Real>;
    using Real3 = Vec<Real, 3>;

    int const n_atoms = rot_coords.size(0);
    int const n_rots = rot_coord_offset.size(0);
    int const n_poses = n_rots_for_pose.size(0);
    int const max_n_blocks = first_rot_for_block.size(1);
    int const max_n_pose_atoms = water_coords.size(1);
    int const max_n_conn = pose_stack_inter_residue_connections.size(2);
    int const n_block_types = block_type_n_atoms.size(0);
    int const max_n_block_atoms = block_type_path_distance.size(1);
    int const max_n_interblock_bonds =
        block_type_atoms_forming_chemical_bonds.size(1);
    int const max_n_tiles = block_type_tile_pol_occ_inds.size(1);

    assert(max_n_interblock_bonds <= lk_max_n_conn);

    assert(first_rot_block_type.size(0) == n_poses);
    assert(first_rot_block_type.size(1) == max_n_blocks);

    assert(block_ind_for_rot.size(0) == n_rots);
    assert(pose_ind_for_rot.size(0) == n_rots);
    assert(block_type_ind_for_rot.size(0) == n_rots);

    assert(n_rots_for_pose.size(0) == n_poses);
    assert(rot_offset_for_pose.size(0) == n_poses);

    assert(n_rots_for_block.size(0) == n_poses);
    assert(n_rots_for_block.size(1) == max_n_blocks);

    assert(rot_offset_for_block.size(0) == n_poses);
    assert(rot_offset_for_block.size(1) == max_n_blocks);

    assert(pose_stack_inter_residue_connections.size(0) == n_poses);
    assert(pose_stack_inter_residue_connections.size(1) == max_n_blocks);

    assert(pose_stack_near_blocks.size(0) == n_poses);
    assert(pose_stack_near_blocks.size(1) == max_n_blocks);
    assert(pose_stack_near_blocks.size(3) == 2);

    assert(pose_stack_inter_block_bondsep.size(0) == n_poses);
    assert(pose_stack_inter_block_bondsep.size(1) == max_n_blocks);
    assert(
        pose_stack_inter_block_bondsep.size(2)
        == pose_stack_near_blocks.size(2));
    assert(pose_stack_inter_block_bondsep.size(3) == max_n_interblock_bonds);
    assert(pose_stack_inter_block_bondsep.size(4) == max_n_interblock_bonds);

    assert(water_coords.size(0) == n_atoms);

    assert(block_type_n_interblock_bonds.size(0) == n_block_types);

    assert(block_type_atoms_forming_chemical_bonds.size(0) == n_block_types);

    assert(block_type_tile_n_polar_atoms.size(0) == n_block_types);
    assert(block_type_tile_n_polar_atoms.size(1) == max_n_tiles);
    assert(block_type_tile_n_occluder_atoms.size(0) == n_block_types);
    assert(block_type_tile_n_occluder_atoms.size(1) == max_n_tiles);
    assert(block_type_tile_pol_occ_inds.size(0) == n_block_types);
    assert(block_type_tile_pol_occ_inds.size(1) == max_n_tiles);
    assert(block_type_tile_pol_occ_inds.size(2) == lk_tile_size);
    assert(block_type_tile_lk_ball_params.size(0) == n_block_types);
    assert(block_type_tile_lk_ball_params.size(1) == max_n_tiles);
    assert(block_type_tile_lk_ball_params.size(2) == lk_tile_size);

    assert(block_type_path_distance.size(0) == n_block_types);
    assert(block_type_path_distance.size(1) == max_n_block_atoms);
    assert(block_type_path_distance.size(2) == max_n_block_atoms);

    TPack<Int, 2, Dev> dispatch_indices_t;
    if (shared_dispatch_indices.size(0) == 3) {
      dispatch_indices_t = shared_dispatch_indices;
    } else {
      auto scratch_rot_spheres_t =
          Dev == Device::CPU ? TPack<Real, 2, Dev>::zeros({n_rots, 4})
                             : TPack<Real, 2, Dev>::empty({n_rots, 4});
      auto scratch_rot_spheres = scratch_rot_spheres_t.view;

      auto scratch_block_spheres_t =
          Dev == Device::CPU
              ? TPack<Real, 3, Dev>::zeros({n_poses, max_n_blocks, 4})
              : TPack<Real, 3, Dev>::empty({n_poses, max_n_blocks, 4});
      auto scratch_block_spheres = scratch_block_spheres_t.view;

      score::common::sphere_overlap::
          compute_rot_spheres<DeviceDispatch, Dev, Real, Int>::f(
              mgr,
              rot_coords,
              rot_coord_offset,
              block_type_ind_for_rot,
              block_type_n_atoms,
              scratch_rot_spheres);

      score::common::sphere_overlap::compute_block_spheres_from_rot_spheres<
          DeviceDispatch,
          Dev,
          Real,
          Int>::
          f(mgr,
            scratch_rot_spheres,
            n_rots_for_block,
            rot_offset_for_block,
            scratch_block_spheres);

      dispatch_indices_t = score::common::sphere_overlap::
          rot_neighbor_indices_from_block_neighbors<
              DeviceDispatch,
              Dev,
              Real,
              Int>::
              f(mgr,
                first_rot_block_type,
                scratch_block_spheres,
                n_rots_for_block,
                rot_offset_for_block,
                scratch_rot_spheres,
                lockstep_group_for_block,
                max_dis);
    }
    auto dispatch_indices = dispatch_indices_t.view;

    TPack<Accum, 2, Dev> output_t;
    if (output_block_pair_energies) {
      output_t = TPack<Accum, 2, Dev>::zeros(
          {n_lk_ball_score_types, dispatch_indices.size(1)});
    } else {
      output_t = TPack<Accum, 2, Dev>::zeros({n_lk_ball_score_types, n_poses});
    }
    auto output = output_t.view;

    // Optimal launch box on v100 and a100 is nt=32, vt=1
    LAUNCH_BOX_32;
    // Define nt and reduce_t
    CTA_REAL_REDUCE_T_TYPEDEF;

    // State the kernel body reads, named explicitly rather than captured
    // implicitly by [=], so the Metal entry point can be handed the same
    // fields; see lk_ball_ctx.hh.
    LKBallCtx<Real, Int, Dev, 2, Real, Accum> ctx;
    ctx.rot_coords = rot_coords;
    ctx.water_coords = water_coords;
    ctx.rot_coord_offset = rot_coord_offset;
    ctx.rot_offset_for_block = rot_offset_for_block;
    ctx.block_type_ind_for_rot = block_type_ind_for_rot;
    ctx.block_ind_for_rot = block_ind_for_rot;
    ctx.pose_stack_inter_residue_connections =
        pose_stack_inter_residue_connections;
    ctx.pose_stack_near_blocks = pose_stack_near_blocks;
    ctx.pose_stack_inter_block_bondsep = pose_stack_inter_block_bondsep;
    ctx.dispatch_indices = dispatch_indices;
    ctx.block_type_n_atoms = block_type_n_atoms;
    ctx.block_type_n_interblock_bonds = block_type_n_interblock_bonds;
    ctx.block_type_atoms_forming_chemical_bonds =
        block_type_atoms_forming_chemical_bonds;
    ctx.block_type_path_distance = block_type_path_distance;
    ctx.block_type_tile_n_polar_atoms = block_type_tile_n_polar_atoms;
    ctx.block_type_tile_n_occluder_atoms = block_type_tile_n_occluder_atoms;
    ctx.block_type_tile_pol_occ_inds = block_type_tile_pol_occ_inds;
    ctx.block_type_tile_lk_ball_params = block_type_tile_lk_ball_params;
    ctx.global_params = global_params;
    ctx.output = output;
    ctx.max_n_blocks = max_n_blocks;
    ctx.output_block_pair_energies = output_block_pair_energies;
    // Unused by the rotamer-pair body -- it walks the dispatch list rather
    // than the block-pair upper triangle -- but the Metal launcher copies
    // every scalar slot, so none may be left uninitialized.
    ctx.n_block_pairs = 0;
    ctx.use_shared_compact_block_neighbors = false;

    // Evaluate the prepared rotamer-pair dispatch list.
    if (Dev == tmol::Device::MPS) {
      // Metal has nowhere for a closure to live, so the identical body is
      // reached through a named entry point instead.
      tmol::score::common::mps_launch_if<Dev>(
          TMOL_MPS_SOURCE(kLKBallMetalShaderSource),
          "lk_ball_eval_rotamer_energies_by_block_f32",
          ctx,
          dispatch_indices.size(1),
          true);
    } else {
      lk_ball_rotamer_native<false, DeviceDispatch, Dev, Real, Int>(mgr, ctx);
    }

    return {output_t.template cast<Real>(), dispatch_indices_t};
  }

  static auto backward(
      ContextManager& mgr,
      TView<Vec<Real, 3>, 1, Dev> rot_coords,
      TView<Vec<Real, 3>, 2, Dev> water_coords,
      TView<Int, 1, Dev> rot_coord_offset,
      TView<Int, 1, Dev> pose_ind_for_atom,
      TView<Int, 2, Dev> first_rot_for_block,
      TView<Int, 2, Dev> first_rot_block_type,
      TView<Int, 1, Dev> block_ind_for_rot,
      TView<Int, 1, Dev> pose_ind_for_rot,
      TView<Int, 1, Dev> block_type_ind_for_rot,
      TView<Int, 1, Dev> n_rots_for_pose,
      TView<Int, 1, Dev> rot_offset_for_pose,
      TView<Int, 2, Dev> n_rots_for_block,
      TView<Int, 2, Dev> rot_offset_for_block,
      Int max_n_rots_per_pose,

      // For determining which atoms to retrieve from neighboring
      // residues we have to know how the blocks in the Pose
      // are connected
      TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_residue_connections,

      // InterBlockBondsep: [pose, block1, slot, (block2, min separation)]
      // and [pose, block1, slot, conn1, conn2]
      TView<Int, 4, Dev> pose_stack_near_blocks,
      TView<int8_t, 5, Dev> pose_stack_inter_block_bondsep,

      //////////////////////
      // Chemical properties
      // how many atoms for a given block
      // Dimsize n_block_types
      TView<Int, 1, Dev> block_type_n_atoms,

      // how many inter-block chemical bonds are there
      // Dimsize: n_block_types
      TView<Int, 1, Dev> block_type_n_interblock_bonds,

      // what atoms form the inter-block chemical bonds
      // Dimsize: n_block_types x max_n_interblock_bonds
      TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds,

      TView<Int, 2, Dev> block_type_tile_n_polar_atoms,
      TView<Int, 2, Dev> block_type_tile_n_occluder_atoms,
      TView<Int, 3, Dev> block_type_tile_pol_occ_inds,
      TView<LKBallTypeParams<Real>, 3, Dev> block_type_tile_lk_ball_params,

      // How many chemical bonds separate all pairs of atoms
      // within each block type?
      // Dimsize: n_block_types x max_n_atoms x max_n_atoms
      TView<Int, 3, Dev> block_type_path_distance,

      //////////////////////

      // LKBall potential parameters
      TView<LKBallGlobalParams<Real>, 1, Dev> global_params,
      TView<Int, 2, Dev> dispatch_indices,  // from forward pass
      TView<Real, 2, Dev> dTdV)
      -> std::tuple<TPack<Vec<Real, 3>, 1, Dev>, TPack<Vec<Real, 3>, 2, Dev>> {
    using tmol::score::common::accumulate;
    using Accum = std::conditional_t<
        Dev == Device::CUDA && std::is_same_v<Real, float>,
        double,
        Real>;
    using Real3 = Vec<Real, 3>;

    int const n_atoms = rot_coords.size(0);
    int const n_rots = rot_coord_offset.size(0);
    int const n_poses = n_rots_for_pose.size(0);
    int const max_n_blocks = first_rot_for_block.size(1);
    int const max_n_pose_atoms = water_coords.size(1);
    int const max_n_conn = pose_stack_inter_residue_connections.size(2);
    int const n_block_types = block_type_n_atoms.size(0);
    int const max_n_block_atoms = block_type_path_distance.size(1);
    int const max_n_interblock_bonds =
        block_type_atoms_forming_chemical_bonds.size(1);
    int const max_n_tiles = block_type_tile_pol_occ_inds.size(1);

    assert(max_n_interblock_bonds <= lk_max_n_conn);

    assert(first_rot_block_type.size(0) == n_poses);
    assert(first_rot_block_type.size(1) == max_n_blocks);

    assert(block_ind_for_rot.size(0) == n_rots);
    assert(pose_ind_for_rot.size(0) == n_rots);
    assert(block_type_ind_for_rot.size(0) == n_rots);

    assert(n_rots_for_pose.size(0) == n_poses);
    assert(rot_offset_for_pose.size(0) == n_poses);

    assert(n_rots_for_block.size(0) == n_poses);
    assert(n_rots_for_block.size(1) == max_n_blocks);

    assert(rot_offset_for_block.size(0) == n_poses);
    assert(rot_offset_for_block.size(1) == max_n_blocks);

    assert(pose_stack_inter_residue_connections.size(0) == n_poses);
    assert(pose_stack_inter_residue_connections.size(1) == max_n_blocks);

    assert(pose_stack_near_blocks.size(0) == n_poses);
    assert(pose_stack_near_blocks.size(1) == max_n_blocks);
    assert(pose_stack_near_blocks.size(3) == 2);

    assert(pose_stack_inter_block_bondsep.size(0) == n_poses);
    assert(pose_stack_inter_block_bondsep.size(1) == max_n_blocks);
    assert(
        pose_stack_inter_block_bondsep.size(2)
        == pose_stack_near_blocks.size(2));
    assert(pose_stack_inter_block_bondsep.size(3) == max_n_interblock_bonds);
    assert(pose_stack_inter_block_bondsep.size(4) == max_n_interblock_bonds);

    assert(water_coords.size(0) == n_atoms);

    assert(block_type_n_interblock_bonds.size(0) == n_block_types);

    assert(block_type_atoms_forming_chemical_bonds.size(0) == n_block_types);

    assert(block_type_tile_n_polar_atoms.size(0) == n_block_types);
    assert(block_type_tile_n_polar_atoms.size(1) == max_n_tiles);
    assert(block_type_tile_n_occluder_atoms.size(0) == n_block_types);
    assert(block_type_tile_n_occluder_atoms.size(1) == max_n_tiles);
    assert(block_type_tile_pol_occ_inds.size(0) == n_block_types);
    assert(block_type_tile_pol_occ_inds.size(1) == max_n_tiles);
    assert(block_type_tile_pol_occ_inds.size(2) == lk_tile_size);
    assert(block_type_tile_lk_ball_params.size(0) == n_block_types);
    assert(block_type_tile_lk_ball_params.size(1) == max_n_tiles);
    assert(block_type_tile_lk_ball_params.size(2) == lk_tile_size);

    assert(block_type_path_distance.size(0) == n_block_types);
    assert(block_type_path_distance.size(1) == max_n_block_atoms);
    assert(block_type_path_distance.size(2) == max_n_block_atoms);

    auto dV_d_pose_coords_t = TPack<Vec<Accum, 3>, 1, Dev>::zeros({n_atoms});
    auto dV_d_pose_coords = dV_d_pose_coords_t.view;

    auto dV_d_water_coords_t =
        TPack<Vec<Accum, 3>, 2, Dev>::zeros({n_atoms, lk_max_n_water});
    auto dV_d_water_coords = dV_d_water_coords_t.view;

    // Optimal launch box on v100 and a100 is nt=32, vt=1
    LAUNCH_BOX_32;
    // Define nt and reduce_t
    CTA_REAL_REDUCE_T_TYPEDEF;

    LKBallCtx<Real, Int, Dev, 2, Accum> ctx;
    ctx.rot_coords = rot_coords;
    ctx.water_coords = water_coords;
    ctx.rot_coord_offset = rot_coord_offset;
    ctx.rot_offset_for_block = rot_offset_for_block;
    ctx.block_type_ind_for_rot = block_type_ind_for_rot;
    ctx.block_ind_for_rot = block_ind_for_rot;
    ctx.pose_stack_inter_residue_connections =
        pose_stack_inter_residue_connections;
    ctx.pose_stack_near_blocks = pose_stack_near_blocks;
    ctx.pose_stack_inter_block_bondsep = pose_stack_inter_block_bondsep;
    ctx.dispatch_indices = dispatch_indices;
    ctx.block_type_n_atoms = block_type_n_atoms;
    ctx.block_type_n_interblock_bonds = block_type_n_interblock_bonds;
    ctx.block_type_atoms_forming_chemical_bonds =
        block_type_atoms_forming_chemical_bonds;
    ctx.block_type_path_distance = block_type_path_distance;
    ctx.block_type_tile_n_polar_atoms = block_type_tile_n_polar_atoms;
    ctx.block_type_tile_n_occluder_atoms = block_type_tile_n_occluder_atoms;
    ctx.block_type_tile_pol_occ_inds = block_type_tile_pol_occ_inds;
    ctx.block_type_tile_lk_ball_params = block_type_tile_lk_ball_params;
    ctx.global_params = global_params;
    int64_t weight_shape[] = {4, dTdV.size(1), 1, 1};
    int64_t weight_stride[] = {dTdV.stride(0), dTdV.stride(1), 1, 1};
    ctx.dTdV = TView<Real, 4, Dev>(dTdV.data(), weight_shape, weight_stride);
    ctx.dV_d_pose_coords = dV_d_pose_coords;
    ctx.dV_d_water_coords = dV_d_water_coords;
    lk_ball_rotamer_native<true, DeviceDispatch, Dev, Real, Int>(mgr, ctx);

    return {
        dV_d_pose_coords_t.template cast<Vec<Real, 3>>(),
        dV_d_water_coords_t.template cast<Vec<Real, 3>>()};
  }
};

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
