#pragma once

#include <tmol/utility/portable/array.hh>
#include <tmol/utility/portable/reduce_ops.hh>
#include <tmol/utility/portable/math.hh>
#include <tmol/utility/portable/matrix.hh>
#include <cmath>
#include <type_traits>
#include <vector>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/counting.hh>
#include <tmol/score/common/device_operations.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/sphere_overlap.impl.hh>
#include <tmol/score/common/tile_atom_pair_evaluation.hh>
#include <tmol/score/common/upper_triangle_indices.hh>
#include <tmol/score/elec/potentials/potentials.hh>
#include <tmol/score/ljlk/potentials/ljlk.hh>

#include <tmol/score/common/tview_binder.hh>
#include <tmol/score/ljlk/potentials/ljlk_elec_ctx.hh>
#include <tmol/backends/metal/score/common/mps_launch.hh>
#ifdef WITH_MPS
#include <tmol/backends/metal/score/ljlk/potentials/ljlk_elec_mps_kernels_source.h>
#endif

#include "ljlk_elec_pose_score.hh"
#include "ljlk_elec_pose_kernels.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <
    bool require_gradient,
    bool weighted,
    bool rotamer_pairs,
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
auto ljlk_elec_forward_impl(
    ContextManager& mgr,
    TView<LJLKExternalVec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 1, D> pose_ind_for_atom,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Int, 1, D> n_rots_for_pose,
    TView<Int, 1, D> rot_offset_for_pose,
    TView<Int, 2, D> n_rots_for_block,
    TView<Int, 2, D> rot_offset_for_block,
    Int max_n_rots_per_pose,
    TView<Int, 4, D> pose_stack_near_blocks,
    TView<int8_t, 5, D> pose_stack_inter_block_bondsep,
    TView<Int, 1, D> block_type_n_atoms,
    TView<Int, 2, D> block_type_atom_types,
    TView<Int, 1, D> block_type_n_interblock_bonds,
    TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,
    TView<Int, 3, D> block_type_ljlk_path_distance,
    TView<Int, 1, D> block_type_all_atoms_ligand_typed,
    TView<LJLKTypeParams<Real>, 1, D> ljlk_type_params,
    TView<LJGlobalParams<Real>, 1, D> ljlk_global_params,
    TView<Real, 2, D> block_type_partial_charge,
    TView<Int, 3, D> block_type_elec_inter_repr_path_distance,
    TView<Int, 3, D> block_type_elec_intra_repr_path_distance,
    TView<tmol::score::elec::potentials::ElecGlobalParams<Real>, 1, D>
        elec_global_params,
    TView<Int, 1, D> shared_compact_block_neighbors,
    TView<Int, 2, D> rotamer_dispatch_indices,
    TView<Real, 1, D> score_weights,
    TView<Real, 1, D> output_gradients,
    Int n_valid_rots = -1)
    -> std::tuple<TPack<Real, 4, D>, TPack<LJLKExternalVec<Real, 3>, 2, D>> {
  using Real3 = tmol::portable::Vec<Real, 3>;
  constexpr bool compute_scores = !require_gradient || !rotamer_pairs;
  int const n_atoms = rot_coords.size(0);
  int const n_poses = first_rot_for_block.size(0);

  int constexpr n_output_scores = weighted ? 1 : 4;
  int const n_outputs =
      rotamer_pairs ? rotamer_dispatch_indices.size(1) : n_poses;
  auto output_t =
      rotamer_pairs
          ? TPack<Real, 4, D>::empty(
                {n_output_scores, 1, 1, compute_scores ? n_outputs : 0})
          : TPack<Real, 4, D>::zeros({n_output_scores, n_outputs, 1, 1});
  auto output = output_t.view;
  std::vector<double> cpu_pose_totals;
  TPack<double, 1, D> cuda_pose_totals_t;
  double* pose_accum = nullptr;
  if constexpr (!rotamer_pairs) {
    if constexpr (D == tmol::Device::CPU) {
      cpu_pose_totals.resize(n_output_scores * n_outputs, 0.0);
      pose_accum = cpu_pose_totals.data();
    } else {
      cuda_pose_totals_t =
          TPack<double, 1, D>::zeros({n_output_scores * n_outputs});
      pose_accum = cuda_pose_totals_t.view.data();
    }
  }
  auto dV_dcoords_t =
      require_gradient ? TPack<Real3, 2, D>::zeros({n_output_scores, n_atoms})
                       : TPack<Real3, 2, D>::empty({n_output_scores, 0});
  auto dV_dcoords = dV_dcoords_t.view;

  ljlk_elec_native_forward<
      require_gradient,
      weighted,
      rotamer_pairs,
      DeviceOperations,
      D,
      Real,
      Int>(
      mgr,
      rot_coords,
      rot_coord_offset,
      first_rot_for_block,
      block_ind_for_rot,
      block_type_ind_for_rot,
      rot_offset_for_block,
      pose_stack_near_blocks,
      pose_stack_inter_block_bondsep,
      block_type_n_atoms,
      block_type_atom_types,
      block_type_n_interblock_bonds,
      block_type_atoms_forming_chemical_bonds,
      block_type_ljlk_path_distance,
      block_type_all_atoms_ligand_typed,
      ljlk_type_params,
      ljlk_global_params,
      block_type_partial_charge,
      block_type_elec_inter_repr_path_distance,
      block_type_elec_intra_repr_path_distance,
      elec_global_params,
      shared_compact_block_neighbors,
      rotamer_dispatch_indices,
      score_weights,
      output_gradients,
      n_valid_rots,
      output,
      dV_dcoords,
      pose_accum);
  return {output_t, dV_dcoords_t};
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKAndElecPoseScoreDispatch<DeviceOperations, D, Real, Int>::forward(
    ContextManager& mgr,
    TView<LJLKExternalVec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 1, D> pose_ind_for_atom,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Int, 1, D> n_rots_for_pose,
    TView<Int, 1, D> rot_offset_for_pose,
    TView<Int, 2, D> n_rots_for_block,
    TView<Int, 2, D> rot_offset_for_block,
    Int max_n_rots_per_pose,
    TView<Int, 4, D> pose_stack_near_blocks,
    TView<int8_t, 5, D> pose_stack_inter_block_bondsep,
    TView<Int, 1, D> block_type_n_atoms,
    TView<Int, 2, D> block_type_atom_types,
    TView<Int, 1, D> block_type_n_interblock_bonds,
    TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,
    TView<Int, 3, D> block_type_ljlk_path_distance,
    TView<Int, 1, D> block_type_all_atoms_ligand_typed,
    TView<LJLKTypeParams<Real>, 1, D> ljlk_type_params,
    TView<LJGlobalParams<Real>, 1, D> ljlk_global_params,
    TView<Real, 2, D> block_type_partial_charge,
    TView<Int, 3, D> block_type_elec_inter_repr_path_distance,
    TView<Int, 3, D> block_type_elec_intra_repr_path_distance,
    TView<tmol::score::elec::potentials::ElecGlobalParams<Real>, 1, D>
        elec_global_params,
    TView<Int, 1, D> shared_compact_block_neighbors,
    bool require_gradient,
    Int n_valid_rots)
    -> std::tuple<TPack<Real, 4, D>, TPack<LJLKExternalVec<Real, 3>, 2, D>> {
#define TMOL_LJLK_ELEC_FORWARD_ARGS                                           \
  mgr, rot_coords, rot_coord_offset, pose_ind_for_atom, first_rot_for_block,  \
      first_rot_block_type, block_ind_for_rot, pose_ind_for_rot,              \
      block_type_ind_for_rot, n_rots_for_pose, rot_offset_for_pose,           \
      n_rots_for_block, rot_offset_for_block, max_n_rots_per_pose,            \
      pose_stack_near_blocks, pose_stack_inter_block_bondsep,                 \
      block_type_n_atoms, block_type_atom_types,                              \
      block_type_n_interblock_bonds, block_type_atoms_forming_chemical_bonds, \
      block_type_ljlk_path_distance, block_type_all_atoms_ligand_typed,       \
      ljlk_type_params, ljlk_global_params, block_type_partial_charge,        \
      block_type_elec_inter_repr_path_distance,                               \
      block_type_elec_intra_repr_path_distance, elec_global_params,           \
      shared_compact_block_neighbors
  auto empty_weights = TPack<Real, 1, D>::empty({0});
  auto empty_rotamer_dispatch = TPack<Int, 2, D>::empty({0, 0});
  if (require_gradient) {
    return ljlk_elec_forward_impl<
        true,
        false,
        false,
        DeviceOperations,
        D,
        Real,
        Int>(
        TMOL_LJLK_ELEC_FORWARD_ARGS,
        empty_rotamer_dispatch.view,
        empty_weights.view,
        empty_weights.view,
        n_valid_rots);
  }
  return ljlk_elec_forward_impl<
      false,
      false,
      false,
      DeviceOperations,
      D,
      Real,
      Int>(
      TMOL_LJLK_ELEC_FORWARD_ARGS,
      empty_rotamer_dispatch.view,
      empty_weights.view,
      empty_weights.view,
      n_valid_rots);
#undef TMOL_LJLK_ELEC_FORWARD_ARGS
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKAndElecPoseScoreDispatch<DeviceOperations, D, Real, Int>::
    forward_weighted(
        ContextManager& mgr,
        TView<LJLKExternalVec<Real, 3>, 1, D> rot_coords,
        TView<Int, 1, D> rot_coord_offset,
        TView<Int, 1, D> pose_ind_for_atom,
        TView<Int, 2, D> first_rot_for_block,
        TView<Int, 2, D> first_rot_block_type,
        TView<Int, 1, D> block_ind_for_rot,
        TView<Int, 1, D> pose_ind_for_rot,
        TView<Int, 1, D> block_type_ind_for_rot,
        TView<Int, 1, D> n_rots_for_pose,
        TView<Int, 1, D> rot_offset_for_pose,
        TView<Int, 2, D> n_rots_for_block,
        TView<Int, 2, D> rot_offset_for_block,
        Int max_n_rots_per_pose,
        TView<Int, 4, D> pose_stack_near_blocks,
        TView<int8_t, 5, D> pose_stack_inter_block_bondsep,
        TView<Int, 1, D> block_type_n_atoms,
        TView<Int, 2, D> block_type_atom_types,
        TView<Int, 1, D> block_type_n_interblock_bonds,
        TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,
        TView<Int, 3, D> block_type_ljlk_path_distance,
        TView<Int, 1, D> block_type_all_atoms_ligand_typed,
        TView<LJLKTypeParams<Real>, 1, D> ljlk_type_params,
        TView<LJGlobalParams<Real>, 1, D> ljlk_global_params,
        TView<Real, 2, D> block_type_partial_charge,
        TView<Int, 3, D> block_type_elec_inter_repr_path_distance,
        TView<Int, 3, D> block_type_elec_intra_repr_path_distance,
        TView<tmol::score::elec::potentials::ElecGlobalParams<Real>, 1, D>
            elec_global_params,
        TView<Int, 1, D> shared_compact_block_neighbors,
        TView<Real, 1, D> score_weights,
        bool require_gradient,
        Int n_valid_rots) -> std::
        tuple<TPack<Real, 4, D>, TPack<LJLKExternalVec<Real, 3>, 2, D>> {
#define TMOL_LJLK_ELEC_WEIGHTED_FORWARD_ARGS                                  \
  mgr, rot_coords, rot_coord_offset, pose_ind_for_atom, first_rot_for_block,  \
      first_rot_block_type, block_ind_for_rot, pose_ind_for_rot,              \
      block_type_ind_for_rot, n_rots_for_pose, rot_offset_for_pose,           \
      n_rots_for_block, rot_offset_for_block, max_n_rots_per_pose,            \
      pose_stack_near_blocks, pose_stack_inter_block_bondsep,                 \
      block_type_n_atoms, block_type_atom_types,                              \
      block_type_n_interblock_bonds, block_type_atoms_forming_chemical_bonds, \
      block_type_ljlk_path_distance, block_type_all_atoms_ligand_typed,       \
      ljlk_type_params, ljlk_global_params, block_type_partial_charge,        \
      block_type_elec_inter_repr_path_distance,                               \
      block_type_elec_intra_repr_path_distance, elec_global_params,           \
      shared_compact_block_neighbors
  auto empty_rotamer_dispatch = TPack<Int, 2, D>::empty({0, 0});
  auto empty_output_gradients = TPack<Real, 1, D>::empty({0});
  if (require_gradient) {
    return ljlk_elec_forward_impl<
        true,
        true,
        false,
        DeviceOperations,
        D,
        Real,
        Int>(
        TMOL_LJLK_ELEC_WEIGHTED_FORWARD_ARGS,
        empty_rotamer_dispatch.view,
        score_weights,
        empty_output_gradients.view,
        n_valid_rots);
  }
  return ljlk_elec_forward_impl<
      false,
      true,
      false,
      DeviceOperations,
      D,
      Real,
      Int>(
      TMOL_LJLK_ELEC_WEIGHTED_FORWARD_ARGS,
      empty_rotamer_dispatch.view,
      score_weights,
      empty_output_gradients.view,
      n_valid_rots);
#undef TMOL_LJLK_ELEC_WEIGHTED_FORWARD_ARGS
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKAndElecPoseScoreDispatch<DeviceOperations, D, Real, Int>::
    build_rotamer_dispatch(
        ContextManager& mgr,
        TView<LJLKExternalVec<Real, 3>, 1, D> rot_coords,
        TView<Int, 1, D> rot_coord_offset,
        TView<Int, 2, D> first_rot_block_type,
        TView<Int, 1, D> block_type_ind_for_rot,
        TView<Int, 2, D> n_rots_for_block,
        TView<Int, 2, D> rot_offset_for_block,
        TView<Int, 2, D> lockstep_group_for_block,
        TView<Int, 1, D> block_type_n_atoms,
        Real max_dis,
        int64_t candidate_begin,
        int64_t candidate_count) -> TPack<Int, 2, D> {
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_block_type.size(0);
  int const max_n_blocks = first_rot_block_type.size(1);
  auto scratch_rot_spheres_t = D == Device::CPU
                                   ? TPack<Real, 2, D>::zeros({n_rots, 4})
                                   : TPack<Real, 2, D>::empty({n_rots, 4});
  auto scratch_block_spheres_t =
      D == Device::CPU ? TPack<Real, 3, D>::zeros({n_poses, max_n_blocks, 4})
                       : TPack<Real, 3, D>::empty({n_poses, max_n_blocks, 4});
  score::common::sphere_overlap::
      compute_rot_spheres<DeviceOperations, D, Real, Int>::f(
          mgr,
          rot_coords,
          rot_coord_offset,
          block_type_ind_for_rot,
          block_type_n_atoms,
          scratch_rot_spheres_t.view);
  score::common::sphere_overlap::
      compute_block_spheres_from_rot_spheres<DeviceOperations, D, Real, Int>::f(
          mgr,
          scratch_rot_spheres_t.view,
          n_rots_for_block,
          rot_offset_for_block,
          scratch_block_spheres_t.view);
  return score::common::sphere_overlap::
      rot_neighbor_indices_from_block_neighbors<
          DeviceOperations,
          D,
          Real,
          Int>::
          f(mgr,
            first_rot_block_type,
            scratch_block_spheres_t.view,
            n_rots_for_block,
            rot_offset_for_block,
            scratch_rot_spheres_t.view,
            lockstep_group_for_block,
            max_dis,
            candidate_begin,
            candidate_count);
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKAndElecPoseScoreDispatch<DeviceOperations, D, Real, Int>::
    forward_weighted_rotamers(
        ContextManager& mgr,
        TView<LJLKExternalVec<Real, 3>, 1, D> rot_coords,
        TView<Int, 1, D> rot_coord_offset,
        TView<Int, 1, D> pose_ind_for_atom,
        TView<Int, 2, D> first_rot_for_block,
        TView<Int, 2, D> first_rot_block_type,
        TView<Int, 1, D> block_ind_for_rot,
        TView<Int, 1, D> pose_ind_for_rot,
        TView<Int, 1, D> block_type_ind_for_rot,
        TView<Int, 1, D> n_rots_for_pose,
        TView<Int, 1, D> rot_offset_for_pose,
        TView<Int, 2, D> n_rots_for_block,
        TView<Int, 2, D> rot_offset_for_block,
        TView<Int, 2, D> lockstep_group_for_block,
        Int max_n_rots_per_pose,
        TView<Int, 4, D> pose_stack_near_blocks,
        TView<int8_t, 5, D> pose_stack_inter_block_bondsep,
        TView<Int, 1, D> block_type_n_atoms,
        TView<Int, 2, D> block_type_atom_types,
        TView<Int, 1, D> block_type_n_interblock_bonds,
        TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,
        TView<Int, 3, D> block_type_ljlk_path_distance,
        TView<Int, 1, D> block_type_all_atoms_ligand_typed,
        TView<LJLKTypeParams<Real>, 1, D> ljlk_type_params,
        TView<LJGlobalParams<Real>, 1, D> ljlk_global_params,
        TView<Real, 2, D> block_type_partial_charge,
        TView<Int, 3, D> block_type_elec_inter_repr_path_distance,
        TView<Int, 3, D> block_type_elec_intra_repr_path_distance,
        TView<tmol::score::elec::potentials::ElecGlobalParams<Real>, 1, D>
            elec_global_params,
        Real max_dis,
        TView<Real, 1, D> score_weights,
        TView<Real, 1, D> output_gradients,
        TPack<Int, 2, D> shared_dispatch_indices)
        -> std::tuple<
            TPack<Real, 4, D>,
            TPack<LJLKExternalVec<Real, 3>, 2, D>,
            TPack<Int, 2, D>> {
  TPack<Int, 2, D> rotamer_dispatch_indices;
  if (shared_dispatch_indices.size(0) == 3) {
    rotamer_dispatch_indices = shared_dispatch_indices;
  } else {
    rotamer_dispatch_indices = build_rotamer_dispatch(
        mgr,
        rot_coords,
        rot_coord_offset,
        first_rot_block_type,
        block_type_ind_for_rot,
        n_rots_for_block,
        rot_offset_for_block,
        lockstep_group_for_block,
        block_type_n_atoms,
        max_dis);
  }
  assert(
      output_gradients.size(0) == 0
      || output_gradients.size(0) == rotamer_dispatch_indices.view.size(1));
  if constexpr (D == tmol::Device::MPS) {
    // Metal runs the packing variant only: weighted energies, no gradient.
    // The kernel body is ljlk_elec_kernels.hh, reached through a named entry
    // point because Metal has nowhere for the lambda below to live.
    if (output_gradients.size(0) != 0) {
      throw std::runtime_error(
          "tmol: the fused LJ/LK-elec rotamer backward has no Metal kernel");
    }
    int64_t const n_pairs = rotamer_dispatch_indices.view.size(1);
    auto output_t = TPack<Real, 4, D>::empty({1, 1, 1, n_pairs});
    LJLKElecRotCtxT<
        tmol::score::common::TViewBinder<D>::template View,
        Real,
        Int>
        ctx;
    ctx.rot_coords = rot_coords;
    ctx.rot_coord_offset = rot_coord_offset;
    ctx.block_ind_for_rot = block_ind_for_rot;
    ctx.block_type_ind_for_rot = block_type_ind_for_rot;
    ctx.pose_stack_near_blocks = pose_stack_near_blocks;
    ctx.pose_stack_inter_block_bondsep = pose_stack_inter_block_bondsep;
    ctx.dispatch_indices = rotamer_dispatch_indices.view;
    ctx.block_type_n_atoms = block_type_n_atoms;
    ctx.block_type_atom_types = block_type_atom_types;
    ctx.block_type_n_interblock_bonds = block_type_n_interblock_bonds;
    ctx.block_type_atoms_forming_chemical_bonds =
        block_type_atoms_forming_chemical_bonds;
    ctx.block_type_ljlk_path_distance = block_type_ljlk_path_distance;
    ctx.block_type_all_atoms_ligand_typed = block_type_all_atoms_ligand_typed;
    ctx.ljlk_type_params = ljlk_type_params;
    ctx.ljlk_global_params = ljlk_global_params;
    ctx.block_type_partial_charge = block_type_partial_charge;
    ctx.block_type_elec_inter_repr_path_distance =
        block_type_elec_inter_repr_path_distance;
    ctx.block_type_elec_intra_repr_path_distance =
        block_type_elec_intra_repr_path_distance;
    ctx.elec_global_params = elec_global_params;
    ctx.score_weights = score_weights;
    ctx.output = output_t.view;
    if (n_pairs > 0) {
      tmol::score::common::mps_launch_if<D>(
          TMOL_MPS_SOURCE(kLJLKElecMetalShaderSource),
          "ljlk_elec_eval_weighted_rotamer_energies_f32",
          ctx,
          n_pairs,
          true);
    }
    auto empty_gradients_t =
        TPack<LJLKExternalVec<Real, 3>, 2, D>::empty({1, 0});
    return {output_t, empty_gradients_t, rotamer_dispatch_indices};
  } else {
    auto empty_compact_block_neighbors = TPack<Int, 1, D>::empty({0});
#define TMOL_LJLK_ELEC_WEIGHTED_ROTAMER_ARGS                                  \
  mgr, rot_coords, rot_coord_offset, pose_ind_for_atom, first_rot_for_block,  \
      first_rot_block_type, block_ind_for_rot, pose_ind_for_rot,              \
      block_type_ind_for_rot, n_rots_for_pose, rot_offset_for_pose,           \
      n_rots_for_block, rot_offset_for_block, max_n_rots_per_pose,            \
      pose_stack_near_blocks, pose_stack_inter_block_bondsep,                 \
      block_type_n_atoms, block_type_atom_types,                              \
      block_type_n_interblock_bonds, block_type_atoms_forming_chemical_bonds, \
      block_type_ljlk_path_distance, block_type_all_atoms_ligand_typed,       \
      ljlk_type_params, ljlk_global_params, block_type_partial_charge,        \
      block_type_elec_inter_repr_path_distance,                               \
      block_type_elec_intra_repr_path_distance, elec_global_params,           \
      empty_compact_block_neighbors.view, rotamer_dispatch_indices.view,      \
      score_weights, output_gradients
    if (shared_dispatch_indices.size(0) == 3) {
      if (output_gradients.size(0) == 0) {
        auto result = ljlk_elec_forward_impl<
            false,
            true,
            true,
            DeviceOperations,
            D,
            Real,
            Int>(TMOL_LJLK_ELEC_WEIGHTED_ROTAMER_ARGS);
        return {
            std::get<0>(result), std::get<1>(result), rotamer_dispatch_indices};
      }
      auto result = ljlk_elec_forward_impl<
          true,
          true,
          true,
          DeviceOperations,
          D,
          Real,
          Int>(TMOL_LJLK_ELEC_WEIGHTED_ROTAMER_ARGS);
      return {
          std::get<0>(result), std::get<1>(result), rotamer_dispatch_indices};
    }
    auto result = ljlk_elec_forward_impl<
        false,
        true,
        true,
        DeviceOperations,
        D,
        Real,
        Int>(TMOL_LJLK_ELEC_WEIGHTED_ROTAMER_ARGS);
#undef TMOL_LJLK_ELEC_WEIGHTED_ROTAMER_ARGS
    return {std::get<0>(result), std::get<1>(result), rotamer_dispatch_indices};
  }
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKAndElecPoseScoreDispatch<DeviceOperations, D, Real, Int>::
    reduce_weighted_scores(
        ContextManager& mgr,
        TView<Real, 2, D> score_lanes,
        TView<Real, 2, D> score_weights,
        Int fused_weight_begin,
        Int fused_weight_width) -> TPack<Real, 1, D> {
  int const n_score_lanes = score_lanes.size(0);
  int const n_poses = score_lanes.size(1);
  auto output_t = TPack<Real, 1, D>::empty({n_poses});
  auto output = output_t.view;
  LAUNCH_BOX_32_OCC_AS(launch_t, 32);
  DeviceOperations<D>::template forall<launch_t>(
      mgr, n_poses, [=] TMOL_DEVICE_FUNC(int pose) {
        Real total = 0;
        for (int lane = 0; lane < n_score_lanes; ++lane) {
          if (lane == fused_weight_begin) {
            total += score_lanes[lane][pose];
            continue;
          }
          int weight_index = lane;
          if (lane > fused_weight_begin) {
            weight_index += fused_weight_width - 1;
          }
          total += score_weights[weight_index][0] * score_lanes[lane][pose];
        }
        output[pose] = total;
      });
  return output_t;
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKAndElecPoseScoreDispatch<DeviceOperations, D, Real, Int>::
    reduce_weighted_score_gradients(
        ContextManager& mgr,
        TView<Real, 1, D> output_gradient,
        TView<Real, 2, D> score_weights,
        Int n_score_lanes,
        Int fused_weight_begin,
        Int fused_weight_width) -> TPack<Real, 2, D> {
  int const n_poses = output_gradient.size(0);
  auto lane_gradients_t = TPack<Real, 2, D>::empty({n_score_lanes, n_poses});
  auto lane_gradients = lane_gradients_t.view;
  LAUNCH_BOX_32_OCC_AS(launch_t, 32);
  DeviceOperations<D>::template forall<launch_t>(
      mgr, n_score_lanes * n_poses, [=] TMOL_DEVICE_FUNC(int index) {
        int const lane = index / n_poses;
        int const pose = index - lane * n_poses;
        Real multiplier = 1;
        if (lane != fused_weight_begin) {
          int weight_index = lane;
          if (lane > fused_weight_begin) {
            weight_index += fused_weight_width - 1;
          }
          multiplier = score_weights[weight_index][0];
        }
        lane_gradients[lane][pose] = multiplier * output_gradient[pose];
      });
  return lane_gradients_t;
}

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
