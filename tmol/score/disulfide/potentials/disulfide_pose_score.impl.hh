#pragma once
#include <tmol/score/disulfide/potentials/disulfide_rotamer_kernels.hh>

#include <tmol/utility/portable/reduce_ops.hh>
#include <tmol/utility/portable/matrix.hh>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/counting.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/uaid_util.hh>
#include <tmol/score/common/warp_segreduce.hh>
#include <tmol/score/common/warp_stride_reduce.hh>

#include <tmol/score/disulfide/potentials/disulfide_pose_score.hh>

// Operator definitions; safe for CPU comM_PIlation

#include <chrono>
#include <cmath>

#include "params.hh"
#include "potentials.hh"
#include "disulfide_pose_kernels.hh"

namespace tmol {
namespace score {
namespace disulfide {
namespace potentials {

using tmol::portable::Vec;

// The disulfide potential is intended to score cyteine-cysteine chemical
// bonds and the parameters are taken from distributions of bond geometries
// in the PDB. However, the logic is general enough that it can support
// non-cysteine residues that form S-S chemical bonds, and even so general
// that it can support cases where a residue forms more than one disulfide
// bond to its neighbors. Have we ever seen such a beast? No. Should we spend
// a few extra minutes to make sure that we can handle it properly _when_
// (not "if"; it's never "if") that crazy beast shows up.
//
// Just like the other potentials, this function will return a list of
// values that it has computed along with a list of which rotamer pairs
// those values correspond to. The trick is to keep track of which
// connection pair is being evaluated in a second tensor, thus allowing
// us to score the beast's double disulfide bond to another beast
// (that is, two disulfide bonds between residues i and j) without
// double counting.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto DisulfidePoseScoreDispatch<DeviceDispatch, D, Real, Int>::forward(
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

    TView<Int, 2, D> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<bool, 2, D> disulfide_conns,
    TView<Int, 3, D> block_type_atom_downstream_of_conn,

    TView<DisulfideGlobalParams<Real>, 1, D> global_params,
    bool output_block_pair_energies,
    bool compute_derivs

    ) -> std::tuple<TPack<Real, 4, D>, TPack<Vec<Real, 3>, 2, D>> {
  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);
  int const max_n_conns = disulfide_conns.size(1);

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

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_connections.size(2) == max_n_conns);

  int n_V;
  if (output_block_pair_energies) {
    n_V = max_n_blocks;
  } else {
    n_V = 1;
  }
  TPack<Real, 4, D> V_t = TPack<Real, 4, D>::zeros({1, n_poses, n_V, n_V});
  auto dV_dx_t = compute_derivs
                     ? TPack<Vec<Real, 3>, 2, D>::zeros({1, n_atoms})
                     : TPack<Vec<Real, 3>, 2, D>::empty({1, n_atoms});

  disulfide_pose_forward<DeviceDispatch, D, Real, Int>(
      mgr,
      rot_coords,
      rot_coord_offset,
      first_rot_for_block,
      first_rot_block_type,
      pose_stack_inter_block_connections,
      disulfide_conns,
      block_type_atom_downstream_of_conn,
      global_params,
      output_block_pair_energies,
      V_t.view,
      dV_dx_t.view);
  return {V_t, dV_dx_t};
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto DisulfidePoseScoreDispatch<DeviceDispatch, D, Real, Int>::backward(
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

    TView<Int, 2, D> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<bool, 2, D> disulfide_conns,
    TView<Int, 3, D> block_type_atom_downstream_of_conn,

    TView<DisulfideGlobalParams<Real>, 1, D> global_params,

    TView<Real, 4, D> dTdV  // n_terms x n_dispatch_total
    ) -> TPack<Vec<Real, 3>, 2, D> {
  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);
  int const max_n_conns = disulfide_conns.size(1);

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

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_connections.size(2) == max_n_conns);

  assert(dTdV.size(0) == 1);
  assert(dTdV.size(1) == n_poses);
  // backward pass only when block_pair_scoring
  assert(dTdV.size(2) == max_n_blocks);
  assert(dTdV.size(3) == max_n_blocks);

  auto dV_dcoords_t = TPack<Vec<Real, 3>, 2, D>::zeros({1, n_atoms});

  disulfide_pose_backward<DeviceDispatch, D, Real, Int>(
      mgr,
      rot_coords,
      rot_coord_offset,
      first_rot_for_block,
      first_rot_block_type,
      pose_stack_inter_block_connections,
      disulfide_conns,
      block_type_atom_downstream_of_conn,
      global_params,
      dTdV,
      dV_dcoords_t.view);
  return dV_dcoords_t;
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto DisulfideRotamerScoreDispatch<DeviceDispatch, D, Real, Int>::forward(
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

    TView<Int, 2, D> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<bool, 2, D> disulfide_conns,
    TView<Int, 3, D> block_type_atom_downstream_of_conn,

    TView<DisulfideGlobalParams<Real>, 1, D> global_params,
    bool output_block_pair_energies,
    bool compute_derivs

    )
    -> std::tuple<
        TPack<Real, 2, D>,
        TPack<Vec<Real, 3>, 2, D>,
        TPack<Int, 2, D>,
        TPack<Int, 2, D>> {
  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);
  int const max_n_conns = disulfide_conns.size(1);

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

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_connections.size(2) == max_n_conns);

  auto n_energies_for_rot_t = TPack<int64_t, 1, D>::zeros({n_rots});
  auto n_energies_for_rot = n_energies_for_rot_t.view;

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  disulfide_rotamer_count_native<DeviceDispatch,D,Real,Int>(mgr,
      block_ind_for_rot,
      pose_ind_for_rot,
      block_type_ind_for_rot,
      n_rots_for_block,
      pose_stack_inter_block_connections,
      disulfide_conns,
      n_energies_for_rot);


  int64_t const max_n_energies_for_rot_64 = DeviceDispatch<D>::reduce(
      mgr,
      n_energies_for_rot.data(),
      n_rots,
      tmol::portable::maximum_t<int64_t>());
  int const max_n_energies_for_rot = score::common::checked_dispatch_size(
      max_n_energies_for_rot_64, "disulfide per-rotamer dispatch");
  int const candidate_dispatch = score::common::checked_dispatch_product(
      n_rots, max_n_energies_for_rot, "disulfide candidate dispatch");

  auto n_energies_for_rot_offset_t = TPack<int64_t, 1, D>::zeros({n_rots});
  auto n_energies_for_rot_offset = n_energies_for_rot_offset_t.view;
  int64_t const n_dispatch_total_64 =
      DeviceDispatch<D>::template scan_and_return_total<
          tmol::portable::scan_type::exclusive>(
          mgr,
          n_energies_for_rot.data(),
          n_energies_for_rot_offset.data(),
          n_rots,
          tmol::portable::plus_t<int64_t>());
  int const n_dispatch_total = score::common::checked_dispatch_size(
      n_dispatch_total_64, "disulfide output dispatch");

  TPack<Real, 2, D> V_t;
  auto dispatch_indices_t = TPack<Int, 2, D>::zeros({3, n_dispatch_total});
  auto conns_for_dispatch_indices_t =
      TPack<Int, 2, D>::zeros({2, n_dispatch_total});
  if (output_block_pair_energies) {
    V_t = TPack<Real, 2, D>::zeros({1, n_dispatch_total});
  } else {
    V_t = TPack<Real, 2, D>::zeros({1, n_poses});
  }
  auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({1, n_atoms});

  auto V = V_t.view;
  auto dV_dx = dV_dx_t.view;
  auto dispatch_indices = dispatch_indices_t.view;
  // keep track of which connection between residues we are
  // scoring as it is possible (though I've never seen it!)
  // to imagine a disulfide-like chemical bond forming twice
  // between two residues
  auto conns_for_dispatch_indices = conns_for_dispatch_indices_t.view;

  disulfide_rotamer_fill_native<DeviceDispatch,D,Real,Int>(mgr,
      first_rot_for_block,
      block_ind_for_rot,
      pose_ind_for_rot,
      block_type_ind_for_rot,
      n_rots_for_block,
      pose_stack_inter_block_connections,
      disulfide_conns,
      n_energies_for_rot,
      n_energies_for_rot_offset,
      dispatch_indices,
      conns_for_dispatch_indices,
      max_n_energies_for_rot,
      candidate_dispatch);


  disulfide_rotamer_score_native<false, DeviceDispatch,D,Real,Int>(mgr,
      rot_coords,
      rot_coord_offset,
      block_ind_for_rot,
      block_type_ind_for_rot,
      disulfide_conns,
      block_type_atom_downstream_of_conn,
      global_params,
      dispatch_indices,
      conns_for_dispatch_indices,
      V,
      dV_dx,
      {});


  return {V_t, dV_dx_t, dispatch_indices_t, conns_for_dispatch_indices_t};
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto DisulfideRotamerScoreDispatch<DeviceDispatch, D, Real, Int>::backward(
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

    TView<Int, 2, D> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<bool, 2, D> disulfide_conns,
    TView<Int, 3, D> block_type_atom_downstream_of_conn,

    TView<DisulfideGlobalParams<Real>, 1, D> global_params,
    TView<Int, 2, D> dispatch_indices,            // from forward pass
    TView<Int, 2, D> conns_for_dispatch_indices,  // from forward pass

    TView<Real, 2, D> dTdV  // n_terms x n_dispatch_total
    ) -> TPack<Vec<Real, 3>, 2, D> {
  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);
  int const max_n_conns = disulfide_conns.size(1);
  int n_dispatch_total = dispatch_indices.size(1);

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

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_connections.size(2) == max_n_conns);

  assert(dispatch_indices.size(0) == 3);
  assert(conns_for_dispatch_indices.size(0) == 2);
  assert(conns_for_dispatch_indices.size(1) == n_dispatch_total);
  assert(dTdV.size(0) == 1);
  assert(
      dTdV.size(1)
      == n_dispatch_total);  // backward pass only when block_pair_scoring

  auto dV_dcoords_t = TPack<Vec<Real, 3>, 2, D>::zeros({1, n_atoms});
  auto dV_dx = dV_dcoords_t.view;

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;

  disulfide_rotamer_score_native<true,DeviceDispatch,D,Real,Int>(mgr,
      rot_coords,
      rot_coord_offset,
      block_ind_for_rot,
      block_type_ind_for_rot,
      disulfide_conns,
      block_type_atom_downstream_of_conn,
      global_params,
      dispatch_indices,
      conns_for_dispatch_indices,
      {},
      dV_dx,
      dTdV);

  return dV_dcoords_t;
}

}  // namespace potentials
}  // namespace disulfide
}  // namespace score
}  // namespace tmol
