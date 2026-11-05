#pragma once

#include <tmol/utility/portable/array.hh>
#include <tmol/utility/portable/reduce_ops.hh>
#include <tmol/utility/portable/math.hh>
#include <tmol/utility/portable/matrix.hh>
#include <vector>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/counting.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/pose_total_reduce.hh>
#include <tmol/score/common/sphere_overlap.impl.hh>
#include <tmol/score/common/tile_atom_pair_evaluation.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/upper_triangle_indices.hh>
#include <tmol/score/common/warp_segreduce.hh>
#include <tmol/score/common/warp_stride_reduce.hh>

#include <tmol/score/ljlk/potentials/lj.hh>
#include <tmol/score/ljlk/potentials/ljlk.hh>
#include <tmol/score/ljlk/potentials/ljlk_ctx.hh>
#include <tmol/score/common/tview_binder.hh>
#include <tmol/score/ljlk/potentials/ljlk_pose_kernels.hh>
#include <tmol/backends/metal/score/common/mps_launch.hh>
#ifdef WITH_MPS
#include <tmol/backends/metal/score/ljlk/potentials/ljlk_mps_kernels_source.h>
#endif
#include <tmol/score/ljlk/potentials/ljlk_pose_score.hh>
#include <tmol/score/ljlk/potentials/lk_isotropic.hh>

// Operator definitions; safe for CPU compilation

#include <chrono>

// ljlk_max_n_conn (the maximum number of inter-residue chemical bonds) and
// ljlk_tile_size come from ljlk_kernels.hh, which the kernel bodies and this
// dispatch must agree on.

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

using tmol::portable::Vec;

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKPoseScoreDispatch<DeviceOperations, D, Real, Int>::
    build_compact_block_neighbors(
        ContextManager& mgr,
        TView<Vec<Real, 3>, 1, D> rot_coords,
        TView<Int, 1, D> rot_coord_offset,
        TView<Int, 2, D> first_rot_for_block,
        TView<Int, 1, D> block_ind_for_rot,
        TView<Int, 1, D> pose_ind_for_rot,
        TView<Int, 1, D> block_type_ind_for_rot,
        TView<Int, 1, D> block_type_n_atoms,
        Real reach) -> TPack<Int, 1, D> {
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);
  int const n_pairs = score::common::checked_triangular_size(
      max_n_blocks, true, "LJ/LK compact block-neighbor dispatch");
  int const n_candidates = score::common::checked_dispatch_product(
      n_poses, n_pairs, "LJ/LK compact block-neighbor dispatch");
  auto block_spheres_t =
      D == Device::CPU ? TPack<Real, 3, D>::zeros({n_poses, max_n_blocks, 4})
                       : TPack<Real, 3, D>::empty({n_poses, max_n_blocks, 4});
  if constexpr (D == Device::CUDA) {
    auto neighbor_indices_t =
        rot_coord_offset.size(0) == 0
            ? TPack<Int, 1, D>::zeros({int64_t(n_candidates) + 1})
            : TPack<Int, 1, D>::empty({int64_t(n_candidates) + 1});
    score::common::sphere_overlap::
        compute_block_spheres<DeviceOperations, D, Real, Int, true>::f(
            mgr,
            rot_coords,
            rot_coord_offset,
            block_ind_for_rot,
            pose_ind_for_rot,
            block_type_ind_for_rot,
            block_type_n_atoms,
            block_spheres_t.view,
            neighbor_indices_t.view.data());
    score::common::sphere_overlap::
        detect_compact_block_neighbors<DeviceOperations, D, Real, Int>::f(
            mgr,
            first_rot_for_block,
            block_spheres_t.view,
            neighbor_indices_t.view,
            reach);
    return neighbor_indices_t;
  } else if constexpr (D == Device::MPS) {
    // The compact block-neighbor list is a CPU/CUDA optimization; the Metal
    // forward tests for MPS before reaching the paths that consume it, so
    // this is unreachable rather than merely unported. The CPU branch below
    // cannot be instantiated here in any case -- its spatial-sweep helper
    // reads host memory, and an MPS tensor's data pointer is an MTLBuffer.
    throw std::runtime_error(
        "tmol: LJ/LK compact block neighbors are not available on the Metal "
        "backend");
  } else {
    score::common::sphere_overlap::
        compute_block_spheres<DeviceOperations, D, Real, Int>::f(
            mgr,
            rot_coords,
            rot_coord_offset,
            block_ind_for_rot,
            pose_ind_for_rot,
            block_type_ind_for_rot,
            block_type_n_atoms,
            block_spheres_t.view);
    std::vector<Int> retained;
    if (score::common::sphere_overlap::try_cpu_spatial_compact_block_neighbors(
            first_rot_for_block, block_spheres_t.view, reach, retained)) {
      auto neighbor_indices_t =
          TPack<Int, 1, D>::empty({int64_t(retained.size()) + 1});
      neighbor_indices_t.view[0] = static_cast<Int>(retained.size());
      if (!retained.empty()) {
        std::memcpy(
            neighbor_indices_t.view.data() + 1,
            retained.data(),
            retained.size() * sizeof(Int));
      }
      return neighbor_indices_t;
    }
    auto neighbor_indices_t =
        TPack<Int, 1, D>::zeros({int64_t(n_candidates) + 1});
    score::common::sphere_overlap::
        detect_compact_block_neighbors<DeviceOperations, D, Real, Int>::f(
            mgr,
            first_rot_for_block,
            block_spheres_t.view,
            neighbor_indices_t.view,
            reach);
    return neighbor_indices_t;
  }
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKPoseScoreDispatch<DeviceOperations, D, Real, Int>::forward(
    ContextManager& mgr,
    // common params
    TView<Vec<Real, 3>, 1, D> rot_coords,
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

    // InterBlockBondsep: [pose, block1, slot, (block2, min separation)]
    // and [pose, block1, slot, conn1, conn2]
    TView<Int, 4, D> pose_stack_near_blocks,
    TView<int8_t, 5, D> pose_stack_inter_block_bondsep,

    //////////////////////
    // Chemical properties
    // how many atoms for a given block
    // Dimsize n_block_types
    TView<Int, 1, D> block_type_n_atoms,

    TView<Int, 2, D> block_type_n_heavy_atoms_in_tile,
    TView<Int, 2, D> block_type_heavy_atoms_in_tile,

    // what are the atom types for these atoms
    // Dimsize: n_block_types x max_n_atoms
    TView<Int, 2, D> block_type_atom_types,

    // how many inter-block chemical bonds are there
    // Dimsize: n_block_types
    TView<Int, 1, D> block_type_n_interblock_bonds,

    // what atoms form the inter-block chemical bonds
    // Dimsize: n_block_types x max_n_interblock_bonds
    TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,

    // what is the path distance between pairs of atoms in the block
    // Dimsize: n_block_types x max_n_atoms x max_n_atoms
    TView<Int, 3, D> block_type_path_distance,
    TView<Int, 1, D> block_type_all_atoms_ligand_typed,
    //////////////////////

    // LJ parameters
    TView<LJLKTypeParams<Real>, 1, D> type_params,
    TView<LJGlobalParams<Real>, 1, D> global_params,

    Real max_dis,

    TView<Int, 1, D> shared_compact_block_neighbors,

    // should the output be per-pose (npose x nterms x 1 x 1)
    //   or per block-pair (npose x nterms x len x len)
    bool output_block_pair_energies,
    bool require_gradient) -> std::
    tuple<TPack<Real, 4, D>, TPack<Vec<Real, 3>, 2, D>, TPack<Int, 3, D> > {
  using Real3 = Vec<Real, 3>;

  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const n_block_types = block_type_n_atoms.size(0);
  int const max_n_block_atoms = block_type_atom_types.size(1);
  int const max_n_interblock_bonds =
      block_type_atoms_forming_chemical_bonds.size(1);

  int const max_n_tiles = block_type_n_heavy_atoms_in_tile.size(2);
  int64_t const n_atom_types = type_params.size(0);

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

  assert(max_n_interblock_bonds <= ljlk_max_n_conn);

  assert(pose_stack_near_blocks.size(0) == n_poses);
  assert(pose_stack_near_blocks.size(1) == max_n_blocks);
  assert(pose_stack_near_blocks.size(3) == 2);

  assert(pose_stack_inter_block_bondsep.size(0) == n_poses);
  assert(pose_stack_inter_block_bondsep.size(1) == max_n_blocks);
  assert(
      pose_stack_inter_block_bondsep.size(2) == pose_stack_near_blocks.size(2));
  assert(pose_stack_inter_block_bondsep.size(3) == max_n_interblock_bonds);
  assert(pose_stack_inter_block_bondsep.size(4) == max_n_interblock_bonds);

  assert(block_type_n_heavy_atoms_in_tile.size(0) == n_block_types);

  assert(block_type_heavy_atoms_in_tile.size(0) == n_block_types);
  assert(
      block_type_heavy_atoms_in_tile.size(1) == ljlk_tile_size * max_n_tiles);

  assert(block_type_atom_types.size(0) == n_block_types);
  assert(block_type_atom_types.size(1) == max_n_block_atoms);

  assert(block_type_n_interblock_bonds.size(0) == n_block_types);

  assert(block_type_atoms_forming_chemical_bonds.size(0) == n_block_types);

  assert(block_type_path_distance.size(0) == n_block_types);
  assert(block_type_path_distance.size(1) == max_n_block_atoms);
  assert(block_type_path_distance.size(2) == max_n_block_atoms);

  // Block-pair autograd recomputes coordinate derivatives in backward.
  bool const accumulate_derivs =
      require_gradient && !output_block_pair_energies;
  auto dV_dcoords_t = accumulate_derivs
                          ? TPack<Vec<Real, 3>, 2, D>::zeros({3, n_atoms})
                          : TPack<Vec<Real, 3>, 2, D>::empty({3, 0});
  auto dV_dcoords = dV_dcoords_t.view;

  bool const use_shared_compact_block_neighbors =
      shared_compact_block_neighbors.size(0) != 0;
  TPack<Real, 3, D> scratch_rot_spheres_t;
  TPack<Int, 3, D> scratch_rot_neighbors_t;
  TView<Real, 3, D> scratch_rot_spheres;
  TView<Int, 3, D> scratch_rot_neighbors;
  if (!use_shared_compact_block_neighbors) {
    scratch_rot_spheres_t =
        D == Device::CPU ? TPack<Real, 3, D>::zeros({n_poses, max_n_blocks, 4})
                         : TPack<Real, 3, D>::empty({n_poses, max_n_blocks, 4});
    scratch_rot_neighbors_t =
        D == Device::CPU
            ? TPack<Int, 3, D>::zeros({n_poses, max_n_blocks, max_n_blocks})
            : TPack<Int, 3, D>::empty({n_poses, max_n_blocks, max_n_blocks});
    scratch_rot_spheres = scratch_rot_spheres_t.view;
    scratch_rot_neighbors = scratch_rot_neighbors_t.view;
  }

  // Metal accumulates whole-pose totals into per-block-pair cells and reduces
  // them afterwards; see the same comment in elec_pose_score.impl.hh for why.
  // In short: one atomic per threadgroup is already the CUDA shape, but
  // thousands of them per pose still land in a varying order, float addition
  // is not associative, and a line search that finite-differences the energy
  // cannot tolerate a non-reproducible total. Writing a cell per threadgroup
  // removes the atomic; summing the cells is fixed-order. The CPU path below
  // already avoids this with a serial double accumulator.
  bool const mps_cell_reduction =
      D == tmol::Device::MPS && !output_block_pair_energies;
  TPack<Real, 4, D> output_t;
  TPack<Real, 4, D> cells_t;
  if (output_block_pair_energies) {
    output_t =
        TPack<Real, 4, D>::zeros({3, n_poses, max_n_blocks, max_n_blocks});
  } else {
    output_t = TPack<Real, 4, D>::zeros({3, n_poses, 1, 1});
    if (mps_cell_reduction) {
      cells_t =
          TPack<Real, 4, D>::zeros({3, n_poses, max_n_blocks, max_n_blocks});
    }
  }
  auto output = mps_cell_reduction ? cells_t.view : output_t.view;

  // CPU workgroups execute serially. Accumulate their totals in double so
  // thousands of small block-pair terms are not rounded away in a large
  // float pose total. Atom-pair math, derivatives and returned dtype stay Real.
  // The CUDA path does not allocate or access this host scratch storage.
  std::vector<double> cpu_pose_totals;
  if constexpr (D == tmol::Device::CPU) {
    if (!output_block_pair_energies) cpu_pose_totals.resize(3 * n_poses, 0.0);
  }
  double* cpu_pose_accum = cpu_pose_totals.data();

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  // Large workloads benefit from a second kernel variant whose register budget
  // exposes a full 32 resident warps per SM. Keep the unconstrained variant for
  // small workloads, where register pressure is less important than avoiding
  // spills.
  LAUNCH_BOX_32_OCC_AS(launch_t_high_occupancy, 32);
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;
  // The total number of unique block pairs (including self-pairs)
  int const max_n_upper_triangle_inds = score::common::checked_triangular_size(
      max_n_blocks, true, "LJ/LK block-pair dispatch");

  // There are two versions of scoring:
  // Block-pair scoring, where the output is written to an n-pose x n-blocks x
  // n-blocks tensor, and whole-pose scoring, where the output is
  // atomic-incremented into an n-pose x 1 x 1 tensor. In the whole-pose scoring
  // case, we compute the atomic derivatives in the forward pass and then weight
  // them by whatever weights are applied to the terms; in a very minimal
  // backward pass (see compiled_ops.cpp). Thread 0 in each CTA atomic- adds the
  // block-pair energy for the block pair it is assigned to. In the block-pair
  // scoring case, we are unable to compute the atomic derivatives in the
  // forward pass, because how the block-pair outputs will be weighted is not
  // known until the backward pass and the memory to store atom-pair derivatives
  // for each interacting block pair would be prohibitive. So we do not compute
  // derivatives at all in the forward pass. The same energy-only kernel also
  // handles whole-pose calls that do not require gradients, avoiding the
  // register footprint and runtime branches of the derivative-capable kernel.
  // State the kernel bodies read, named explicitly rather than captured
  // implicitly by [=]. The two lambdas below close over this one object, and
  // the Metal launch binds the same fields positionally; see ljlk_ctx.hh.
  LJLKCtx<Real, Int, D> ctx;
  ctx.rot_coords = rot_coords;
  ctx.rot_coord_offset = rot_coord_offset;
  ctx.block_ind_for_rot = block_ind_for_rot;
  ctx.pose_ind_for_rot = pose_ind_for_rot;
  ctx.block_type_ind_for_rot = block_type_ind_for_rot;
  ctx.first_rot_block_type = first_rot_block_type;
  ctx.rot_offset_for_block = rot_offset_for_block;
  ctx.pose_stack_near_blocks = pose_stack_near_blocks;
  ctx.pose_stack_inter_block_bondsep = pose_stack_inter_block_bondsep;
  ctx.scratch_rot_neighbors = scratch_rot_neighbors;
  ctx.shared_compact_block_neighbors = shared_compact_block_neighbors;
  ctx.block_type_n_atoms = block_type_n_atoms;
  ctx.block_type_atom_types = block_type_atom_types;
  ctx.block_type_n_interblock_bonds = block_type_n_interblock_bonds;
  ctx.block_type_atoms_forming_chemical_bonds =
      block_type_atoms_forming_chemical_bonds;
  ctx.block_type_path_distance = block_type_path_distance;
  ctx.block_type_all_atoms_ligand_typed = block_type_all_atoms_ligand_typed;
  ctx.type_params = type_params;
  ctx.global_params = global_params;
  ctx.output = output;
  ctx.dV_dcoords = dV_dcoords;
  ctx.max_n_blocks = max_n_blocks;
  ctx.max_n_upper_triangle_inds = max_n_upper_triangle_inds;
  // The store writes cells whenever the output view is cell-shaped.
  ctx.output_block_pair_energies =
      output_block_pair_energies || mps_cell_reduction;
  ctx.use_shared_compact_block_neighbors = use_shared_compact_block_neighbors;
  ctx.require_gradient = require_gradient;
  ctx.max_dis = max_dis;

  LJLKPoseEvaluator<false, DeviceOperations, D, Real, Int, launch_t>
      eval_energies_by_block{ctx, cpu_pose_accum};
  LJLKPoseEvaluator<true, DeviceOperations, D, Real, Int, launch_t>
      eval_energies{ctx, cpu_pose_accum};

  ///////////////////////////////////////////////////////////////////////

  // Three steps
  // 0: setup
  // 1: launch a kernel to find a small bounding sphere surrounding the blocks
  // 2: launch a kernel to look for spheres that are within striking distance of
  // each other
  // 3: launch a kernel to evaluate lj/lk between pairs of blocks
  // within striking distance
  if (!use_shared_compact_block_neighbors) {
    score::common::sphere_overlap::
        compute_block_spheres<DeviceOperations, D, Real, Int>::f(
            mgr,
            rot_coords,
            rot_coord_offset,
            block_ind_for_rot,
            pose_ind_for_rot,
            block_type_ind_for_rot,
            block_type_n_atoms,
            scratch_rot_spheres);

    score::common::sphere_overlap::
        detect_block_neighbors<DeviceOperations, D, Real, Int>::f(
            mgr,
            first_rot_block_type,
            scratch_rot_spheres,
            scratch_rot_neighbors,
            max_dis);
  }

  if (D == tmol::Device::MPS) {
    // Metal cannot launch a device lambda -- there is nowhere for a closure
    // to live, and entry points are compiled by a separate toolchain -- so
    // the same kernel body is reached through a named entry rather than
    // through eval_energies_by_block / eval_energies. The context those
    // lambdas close over is the same object either way.
    //
    // This is tested ahead of the compact-block-neighbor paths because those
    // launch through DeviceOperations, which on this backend would run the
    // kernel on the CPU.
    // Entry selection depends on whether gradients are wanted, not on the
    // output shape: the cell store is independent of the derivative
    // accumulation, so a whole-pose call with gradients gets both.
    tmol::score::common::mps_launch_if<D>(
        TMOL_MPS_SOURCE(kLJLKMetalShaderSource),
        require_gradient && !output_block_pair_energies
            ? "ljlk_eval_energies_f32"
            : "ljlk_eval_energies_by_block_f32",
        ctx,
        int64_t(n_poses) * int64_t(max_n_upper_triangle_inds),
        true);
    if (mps_cell_reduction) {
      // Compensated, fixed-order reduction rather than torch's tree sum.
      // A tree sum is reproducible but still lets the total move by several
      // ULP when a coordinate moves by a fraction of one, because its error
      // term moves with its inputs; see pose_total_reduce_kernels.hh.
      score::common::pose_total_reduce<DeviceOperations, D, Real>::f(
          mgr, cells_t.view, output_t.view);
    }
  } else {
#ifdef __NVCC__
    if (!output_block_pair_energies && !use_shared_compact_block_neighbors
        && score::common::sphere_overlap::should_compact_block_neighbors(
            n_poses, max_n_blocks, require_gradient)) {
      if (require_gradient) {
        score::common::sphere_overlap::launch_compact_block_neighbors<
            DeviceOperations,
            D,
            launch_t_high_occupancy,
            Int>(mgr, scratch_rot_neighbors, eval_energies);
      } else {
        score::common::sphere_overlap::launch_compact_block_neighbors<
            DeviceOperations,
            D,
            launch_t_high_occupancy,
            Int>(mgr, scratch_rot_neighbors, eval_energies_by_block);
      }
    } else
#endif
    {
      ljlk_pose_forward_native<DeviceOperations, D, Real, Int>(
          mgr, ctx, shared_compact_block_neighbors, n_poses, cpu_pose_accum);
    }
  }

  return {output_t, dV_dcoords_t, scratch_rot_neighbors_t};
}  // LJLKPoseScoreDispatch::forward

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKPoseScoreDispatch<DeviceOperations, D, Real, Int>::backward(
    ContextManager& mgr,
    // common params
    TView<Vec<Real, 3>, 1, D> rot_coords,
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

    // InterBlockBondsep: [pose, block1, slot, (block2, min separation)]
    // and [pose, block1, slot, conn1, conn2]
    TView<Int, 4, D> pose_stack_near_blocks,
    TView<int8_t, 5, D> pose_stack_inter_block_bondsep,

    //////////////////////
    // Chemical properties
    // how many atoms for a given block
    // Dimsize n_block_types
    TView<Int, 1, D> block_type_n_atoms,

    TView<Int, 2, D> block_type_n_heavy_atoms_in_tile,
    TView<Int, 2, D> block_type_heavy_atoms_in_tile,

    // what are the atom types for these atoms
    // Dimsize: n_block_types x max_n_atoms
    TView<Int, 2, D> block_type_atom_types,

    // how many inter-block chemical bonds are there
    // Dimsize: n_block_types
    TView<Int, 1, D> block_type_n_interblock_bonds,

    // what atoms form the inter-block chemical bonds
    // Dimsize: n_block_types x max_n_interblock_bonds
    TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,

    // what is the path distance between pairs of atoms in the block
    // Dimsize: n_block_types x max_n_atoms x max_n_atoms
    TView<Int, 3, D> block_type_path_distance,
    TView<Int, 1, D> block_type_all_atoms_ligand_typed,
    //////////////////////

    // LJ parameters
    TView<LJLKTypeParams<Real>, 1, D> type_params,
    TView<LJGlobalParams<Real>, 1, D> global_params,

    TView<Int, 3, D> scratch_rot_neighbors,  // from forward pass

    TView<Real, 4, D> dTdV  // nterms x nposes x len x len
    ) -> TPack<Vec<Real, 3>, 1, D> {
  using tmol::score::common::accumulate;
  using Real3 = Vec<Real, 3>;

  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const n_block_types = block_type_n_atoms.size(0);
  int const max_n_block_atoms = block_type_atom_types.size(1);
  int const max_n_interblock_bonds =
      block_type_atoms_forming_chemical_bonds.size(1);

  int const max_n_tiles = block_type_n_heavy_atoms_in_tile.size(2);
  int64_t const n_atom_types = type_params.size(0);

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

  assert(max_n_interblock_bonds <= ljlk_max_n_conn);

  assert(pose_stack_near_blocks.size(0) == n_poses);
  assert(pose_stack_near_blocks.size(1) == max_n_blocks);
  assert(pose_stack_near_blocks.size(3) == 2);

  assert(pose_stack_inter_block_bondsep.size(0) == n_poses);
  assert(pose_stack_inter_block_bondsep.size(1) == max_n_blocks);
  assert(
      pose_stack_inter_block_bondsep.size(2) == pose_stack_near_blocks.size(2));
  assert(pose_stack_inter_block_bondsep.size(3) == max_n_interblock_bonds);
  assert(pose_stack_inter_block_bondsep.size(4) == max_n_interblock_bonds);

  assert(block_type_n_heavy_atoms_in_tile.size(0) == n_block_types);

  assert(block_type_heavy_atoms_in_tile.size(0) == n_block_types);
  assert(
      block_type_heavy_atoms_in_tile.size(1) == ljlk_tile_size * max_n_tiles);

  assert(block_type_atom_types.size(0) == n_block_types);
  assert(block_type_atom_types.size(1) == max_n_block_atoms);

  assert(block_type_n_interblock_bonds.size(0) == n_block_types);

  assert(block_type_atoms_forming_chemical_bonds.size(0) == n_block_types);

  assert(block_type_path_distance.size(0) == n_block_types);
  assert(block_type_path_distance.size(1) == max_n_block_atoms);
  assert(block_type_path_distance.size(2) == max_n_block_atoms);

  auto dV_dcoords_t = TPack<Vec<Real, 3>, 1, D>::zeros({n_atoms});
  auto dV_dcoords = dV_dcoords_t.view;

  LJLKCtx<Real, Int, D, 4> ctx;
  ctx.rot_coords = rot_coords;
  ctx.rot_coord_offset = rot_coord_offset;
  ctx.block_ind_for_rot = block_ind_for_rot;
  ctx.pose_ind_for_rot = pose_ind_for_rot;
  ctx.block_type_ind_for_rot = block_type_ind_for_rot;
  ctx.first_rot_block_type = first_rot_block_type;
  ctx.rot_offset_for_block = rot_offset_for_block;
  ctx.pose_stack_near_blocks = pose_stack_near_blocks;
  ctx.pose_stack_inter_block_bondsep = pose_stack_inter_block_bondsep;
  ctx.block_type_n_atoms = block_type_n_atoms;
  ctx.block_type_atom_types = block_type_atom_types;
  ctx.block_type_n_interblock_bonds = block_type_n_interblock_bonds;
  ctx.block_type_atoms_forming_chemical_bonds =
      block_type_atoms_forming_chemical_bonds;
  ctx.block_type_path_distance = block_type_path_distance;
  ctx.block_type_all_atoms_ligand_typed = block_type_all_atoms_ligand_typed;
  ctx.type_params = type_params;
  ctx.global_params = global_params;
  ctx.dTdV = dTdV;
  int64_t grad_shape[] = {1, n_atoms};
  int64_t grad_stride[] = {n_atoms, 1};
  ctx.dV_dcoords =
      TView<Vec<Real, 3>, 2, D>(dV_dcoords.data(), grad_shape, grad_stride);
  ctx.max_n_blocks = max_n_blocks;
  ctx.scratch_rot_neighbors = scratch_rot_neighbors;
  ctx.max_n_upper_triangle_inds = score::common::checked_triangular_size(
      max_n_blocks, true, "LJ/LK derivative block-pair dispatch");
  ljlk_backward_native<false, DeviceOperations, D, Real, Int>(
      mgr, ctx, n_poses);

  return dV_dcoords_t;
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKRotamerScoreDispatch<DeviceOperations, D, Real, Int>::forward(
    ContextManager& mgr,
    // common params
    TView<Vec<Real, 3>, 1, D> rot_coords,
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
    // [n_poses, max_n_blocks]; blocks sharing an id >= 0 move in
    // lockstep, so only matching rotamer indices ever coexist
    TView<Int, 2, D> lockstep_group_for_block,
    Int max_n_rots_per_pose,

    // InterBlockBondsep: [pose, block1, slot, (block2, min separation)]
    // and [pose, block1, slot, conn1, conn2]
    TView<Int, 4, D> pose_stack_near_blocks,
    TView<int8_t, 5, D> pose_stack_inter_block_bondsep,

    //////////////////////
    // Chemical properties
    // how many atoms for a given block
    // Dimsize n_block_types
    TView<Int, 1, D> block_type_n_atoms,

    TView<Int, 2, D> block_type_n_heavy_atoms_in_tile,
    TView<Int, 2, D> block_type_heavy_atoms_in_tile,

    // what are the atom types for these atoms
    // Dimsize: n_block_types x max_n_atoms
    TView<Int, 2, D> block_type_atom_types,

    // how many inter-block chemical bonds are there
    // Dimsize: n_block_types
    TView<Int, 1, D> block_type_n_interblock_bonds,

    // what atoms form the inter-block chemical bonds
    // Dimsize: n_block_types x max_n_interblock_bonds
    TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,

    // what is the path distance between pairs of atoms in the block
    // Dimsize: n_block_types x max_n_atoms x max_n_atoms
    TView<Int, 3, D> block_type_path_distance,
    TView<Int, 1, D> block_type_all_atoms_ligand_typed,
    //////////////////////

    // LJ parameters
    TView<LJLKTypeParams<Real>, 1, D> type_params,
    TView<LJGlobalParams<Real>, 1, D> global_params,

    Real max_dis,

    // should the output be per-pose (npose x nterms x 1 x 1)
    //   or per block-pair (npose x nterms x len x len)
    bool output_block_pair_energies,
    bool require_gradient) -> std::
    tuple<TPack<Real, 2, D>, TPack<Vec<Real, 3>, 2, D>, TPack<Int, 2, D> > {
  using Real3 = Vec<Real, 3>;

  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const n_block_types = block_type_n_atoms.size(0);
  int const max_n_block_atoms = block_type_atom_types.size(1);
  int const max_n_interblock_bonds =
      block_type_atoms_forming_chemical_bonds.size(1);

  int const max_n_tiles = block_type_n_heavy_atoms_in_tile.size(2);
  int64_t const n_atom_types = type_params.size(0);

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

  assert(max_n_interblock_bonds <= ljlk_max_n_conn);

  assert(pose_stack_near_blocks.size(0) == n_poses);
  assert(pose_stack_near_blocks.size(1) == max_n_blocks);
  assert(pose_stack_near_blocks.size(3) == 2);

  assert(pose_stack_inter_block_bondsep.size(0) == n_poses);
  assert(pose_stack_inter_block_bondsep.size(1) == max_n_blocks);
  assert(
      pose_stack_inter_block_bondsep.size(2) == pose_stack_near_blocks.size(2));
  assert(pose_stack_inter_block_bondsep.size(3) == max_n_interblock_bonds);
  assert(pose_stack_inter_block_bondsep.size(4) == max_n_interblock_bonds);

  assert(block_type_n_heavy_atoms_in_tile.size(0) == n_block_types);

  assert(block_type_heavy_atoms_in_tile.size(0) == n_block_types);
  assert(
      block_type_heavy_atoms_in_tile.size(1) == ljlk_tile_size * max_n_tiles);

  assert(block_type_atom_types.size(0) == n_block_types);
  assert(block_type_atom_types.size(1) == max_n_block_atoms);

  assert(block_type_n_interblock_bonds.size(0) == n_block_types);

  assert(block_type_atoms_forming_chemical_bonds.size(0) == n_block_types);

  assert(block_type_path_distance.size(0) == n_block_types);
  assert(block_type_path_distance.size(1) == max_n_block_atoms);
  assert(block_type_path_distance.size(2) == max_n_block_atoms);

  auto dV_dcoords_t = TPack<Vec<Real, 3>, 2, D>::empty({3, 0});

  auto scratch_rot_spheres_t = D == Device::CPU
                                   ? TPack<Real, 2, D>::zeros({n_rots, 4})
                                   : TPack<Real, 2, D>::empty({n_rots, 4});
  auto scratch_rot_spheres = scratch_rot_spheres_t.view;

  auto scratch_block_spheres_t =
      D == Device::CPU ? TPack<Real, 3, D>::zeros({n_poses, max_n_blocks, 4})
                       : TPack<Real, 3, D>::empty({n_poses, max_n_blocks, 4});
  auto scratch_block_spheres = scratch_block_spheres_t.view;

  score::common::sphere_overlap::
      compute_rot_spheres<DeviceOperations, D, Real, Int>::f(
          mgr,
          rot_coords,
          rot_coord_offset,
          block_type_ind_for_rot,
          block_type_n_atoms,
          scratch_rot_spheres);

  score::common::sphere_overlap::
      compute_block_spheres_from_rot_spheres<DeviceOperations, D, Real, Int>::f(
          mgr,
          scratch_rot_spheres,
          n_rots_for_block,
          rot_offset_for_block,
          scratch_block_spheres);

  auto dispatch_indices_t =
      score::common::sphere_overlap::rot_neighbor_indices_from_block_neighbors<
          DeviceOperations,
          D,
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

  auto dispatch_indices = dispatch_indices_t.view;

  TPack<Real, 2, D> output_t;
  if (output_block_pair_energies) {
    output_t = TPack<Real, 2, D>::zeros({3, dispatch_indices.size(1)});
  } else {
    output_t = TPack<Real, 2, D>::zeros({3, n_poses});
  }
  auto output = output_t.view;

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;

  // State the kernel body reads, named explicitly rather than captured
  // implicitly by [=], so the Metal entry point can be handed the same
  // fields; see ljlk_ctx.hh.
  LJLKCtx<Real, Int, D, 2> ctx;
  ctx.rot_coords = rot_coords;
  ctx.rot_coord_offset = rot_coord_offset;
  ctx.block_ind_for_rot = block_ind_for_rot;
  ctx.pose_ind_for_rot = pose_ind_for_rot;
  ctx.block_type_ind_for_rot = block_type_ind_for_rot;
  ctx.first_rot_block_type = first_rot_block_type;
  ctx.rot_offset_for_block = rot_offset_for_block;
  ctx.pose_stack_near_blocks = pose_stack_near_blocks;
  ctx.pose_stack_inter_block_bondsep = pose_stack_inter_block_bondsep;
  ctx.dispatch_indices = dispatch_indices;
  ctx.block_type_n_atoms = block_type_n_atoms;
  ctx.block_type_atom_types = block_type_atom_types;
  ctx.block_type_n_interblock_bonds = block_type_n_interblock_bonds;
  ctx.block_type_atoms_forming_chemical_bonds =
      block_type_atoms_forming_chemical_bonds;
  ctx.block_type_path_distance = block_type_path_distance;
  ctx.block_type_all_atoms_ligand_typed = block_type_all_atoms_ligand_typed;
  ctx.type_params = type_params;
  ctx.global_params = global_params;
  ctx.output = output;
  ctx.max_n_blocks = max_n_blocks;
  ctx.output_block_pair_energies = true;
  ctx.max_dis = max_dis;
  // Unused by the rotamer-pair body -- it walks the dispatch list rather than
  // the block-pair upper triangle, and computes no derivatives -- but the
  // Metal launcher copies every scalar slot, so none may be left
  // uninitialized.
  ctx.max_n_upper_triangle_inds = 0;
  ctx.use_shared_compact_block_neighbors = false;
  ctx.require_gradient = false;

  // Only the whole-pose CPU path accumulates in double; the rotamer path
  // writes one cell per workgroup.
  double* cpu_pose_accum = nullptr;

  auto eval_energies_by_block = ([=] TMOL_DEVICE_FUNC(int cta) {
    typedef LJLKSharedMem<Real, ljlk_tile_size, ljlk_max_n_conn, launch_t>
        shared_mem_union;
    SHARED_MEMORY shared_mem_union shared;
    ljlk_eval_rotamer_energies_by_block<
        DeviceOperations,
        D,
        Real,
        Int,
        launch_t>(ctx, shared, cta, cpu_pose_accum);
  });

  ///////////////////////////////////////////////////////////////////////

  // Three steps
  // 0: setup
  // 1: launch a kernel to find a small bounding sphere surrounding the blocks
  // 2: launch a kernel to look for spheres that are within striking distance of
  // each other
  // 3: launch a kernel to evaluate lj/lk between pairs of blocks
  // within striking distance

  assert(output_block_pair_energies);
  // Each pair writes its own score; backward recomputes its derivatives.
  if (D == tmol::Device::MPS) {
    // Metal has nowhere for a closure to live, so the identical body is
    // reached through a named entry point instead.
    tmol::score::common::mps_launch_if<D>(
        TMOL_MPS_SOURCE(kLJLKMetalShaderSource),
        "ljlk_eval_rotamer_energies_by_block_f32",
        ctx,
        dispatch_indices.size(1),
        true);
  } else {
    DeviceOperations<D>::template foreach_independent_workgroup<launch_t>(
        mgr, dispatch_indices.size(1), eval_energies_by_block);
  }

  return {output_t, dV_dcoords_t, dispatch_indices_t};
}  // LJLKRotamerScoreDispatch::forward

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKRotamerScoreDispatch<DeviceOperations, D, Real, Int>::backward(
    ContextManager& mgr,
    // common params
    TView<Vec<Real, 3>, 1, D> rot_coords,
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

    // InterBlockBondsep: [pose, block1, slot, (block2, min separation)]
    // and [pose, block1, slot, conn1, conn2]
    TView<Int, 4, D> pose_stack_near_blocks,
    TView<int8_t, 5, D> pose_stack_inter_block_bondsep,

    //////////////////////
    // Chemical properties
    // how many atoms for a given block
    // Dimsize n_block_types
    TView<Int, 1, D> block_type_n_atoms,

    TView<Int, 2, D> block_type_n_heavy_atoms_in_tile,
    TView<Int, 2, D> block_type_heavy_atoms_in_tile,

    // what are the atom types for these atoms
    // Dimsize: n_block_types x max_n_atoms
    TView<Int, 2, D> block_type_atom_types,

    // how many inter-block chemical bonds are there
    // Dimsize: n_block_types
    TView<Int, 1, D> block_type_n_interblock_bonds,

    // what atoms form the inter-block chemical bonds
    // Dimsize: n_block_types x max_n_interblock_bonds
    TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,

    // what is the path distance between pairs of atoms in the block
    // Dimsize: n_block_types x max_n_atoms x max_n_atoms
    TView<Int, 3, D> block_type_path_distance,
    TView<Int, 1, D> block_type_all_atoms_ligand_typed,
    //////////////////////

    // LJ parameters
    TView<LJLKTypeParams<Real>, 1, D> type_params,
    TView<LJGlobalParams<Real>, 1, D> global_params,

    TView<Int, 2, D> dispatch_indices,  // from forward pass

    TView<Real, 2, D> dTdV  // nterms x n-dispatch
    ) -> TPack<Vec<Real, 3>, 1, D> {
  using tmol::score::common::accumulate;
  using Real3 = Vec<Real, 3>;

  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const n_block_types = block_type_n_atoms.size(0);
  int const max_n_block_atoms = block_type_atom_types.size(1);
  int const max_n_interblock_bonds =
      block_type_atoms_forming_chemical_bonds.size(1);

  int const max_n_tiles = block_type_n_heavy_atoms_in_tile.size(2);
  int64_t const n_atom_types = type_params.size(0);

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

  assert(max_n_interblock_bonds <= ljlk_max_n_conn);

  assert(pose_stack_near_blocks.size(0) == n_poses);
  assert(pose_stack_near_blocks.size(1) == max_n_blocks);
  assert(pose_stack_near_blocks.size(3) == 2);

  assert(pose_stack_inter_block_bondsep.size(0) == n_poses);
  assert(pose_stack_inter_block_bondsep.size(1) == max_n_blocks);
  assert(
      pose_stack_inter_block_bondsep.size(2) == pose_stack_near_blocks.size(2));
  assert(pose_stack_inter_block_bondsep.size(3) == max_n_interblock_bonds);
  assert(pose_stack_inter_block_bondsep.size(4) == max_n_interblock_bonds);

  assert(block_type_n_heavy_atoms_in_tile.size(0) == n_block_types);

  assert(block_type_heavy_atoms_in_tile.size(0) == n_block_types);
  assert(
      block_type_heavy_atoms_in_tile.size(1) == ljlk_tile_size * max_n_tiles);

  assert(block_type_atom_types.size(0) == n_block_types);
  assert(block_type_atom_types.size(1) == max_n_block_atoms);

  assert(block_type_n_interblock_bonds.size(0) == n_block_types);

  assert(block_type_atoms_forming_chemical_bonds.size(0) == n_block_types);

  assert(block_type_path_distance.size(0) == n_block_types);
  assert(block_type_path_distance.size(1) == max_n_block_atoms);
  assert(block_type_path_distance.size(2) == max_n_block_atoms);

  auto dV_dcoords_t = TPack<Vec<Real, 3>, 1, D>::zeros({n_atoms});
  auto dV_dcoords = dV_dcoords_t.view;

  LJLKCtx<Real, Int, D, 2> ctx;
  ctx.rot_coords = rot_coords;
  ctx.rot_coord_offset = rot_coord_offset;
  ctx.block_ind_for_rot = block_ind_for_rot;
  ctx.pose_ind_for_rot = pose_ind_for_rot;
  ctx.block_type_ind_for_rot = block_type_ind_for_rot;
  ctx.first_rot_block_type = first_rot_block_type;
  ctx.rot_offset_for_block = rot_offset_for_block;
  ctx.pose_stack_near_blocks = pose_stack_near_blocks;
  ctx.pose_stack_inter_block_bondsep = pose_stack_inter_block_bondsep;
  ctx.block_type_n_atoms = block_type_n_atoms;
  ctx.block_type_atom_types = block_type_atom_types;
  ctx.block_type_n_interblock_bonds = block_type_n_interblock_bonds;
  ctx.block_type_atoms_forming_chemical_bonds =
      block_type_atoms_forming_chemical_bonds;
  ctx.block_type_path_distance = block_type_path_distance;
  ctx.block_type_all_atoms_ligand_typed = block_type_all_atoms_ligand_typed;
  ctx.type_params = type_params;
  ctx.global_params = global_params;
  ctx.dTdV = dTdV;
  int64_t grad_shape[] = {1, n_atoms};
  int64_t grad_stride[] = {n_atoms, 1};
  ctx.dV_dcoords =
      TView<Vec<Real, 3>, 2, D>(dV_dcoords.data(), grad_shape, grad_stride);
  ctx.max_n_blocks = max_n_blocks;
  ctx.dispatch_indices = dispatch_indices;
  ljlk_backward_native<true, DeviceOperations, D, Real, Int>(mgr, ctx, n_poses);

  return dV_dcoords_t;
}

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
