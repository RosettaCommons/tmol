#pragma once
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/utility/portable/matrix.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/rotamer_correlation.hh>
namespace tmol::score::common {
using tmol::portable::Vec;
// One diagonal entry per valid rotamer plus connected upper-block pairs.
// Zeroed int64 counts, exclusive offsets and output owners are caller-owned.
// Matching nonnegative lockstep groups select matching local rotamer indices.
template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
void connected_rotamer_count_native(
    ContextManager& mgr,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Int, 2, D> n_rots_for_block,
    TView<Int, 2, D> lockstep_group_for_block,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<int64_t, 1, D> n_output_intxns_for_rot_conn) {
  LAUNCH_BOX_32;
  int const max_n_conns = pose_stack_inter_block_connections.size(2);
  auto count_intxns_for_rot_conn = ([=] TMOL_DEVICE_FUNC(int index) {
    int const rot_ind = index / (max_n_conns + 1);
    int const conn_ind = index % (max_n_conns + 1);
    int const pose_ind = pose_ind_for_rot[rot_ind];
    int const block_ind = block_ind_for_rot[rot_ind];
    int const block_type_ind = block_type_ind_for_rot[rot_ind];
    if (block_type_ind == -1) return;

    if (conn_ind == max_n_conns) {
      n_output_intxns_for_rot_conn[index] = 1;
    } else {
      int const other_block_ind =
          pose_stack_inter_block_connections[pose_ind][block_ind][conn_ind][0];
      if (other_block_ind == -1) return;
      int const other_block_n_rots =
          n_rots_for_block[pose_ind][other_block_ind];
      if (block_ind < other_block_ind) {
        // blocks that sample in lockstep only ever coexist at matching
        // rotamer indices, so this rotamer pairs with exactly one of theirs
        n_output_intxns_for_rot_conn[index] =
            same_rotamer_group(
                lockstep_group_for_block, pose_ind, block_ind, other_block_ind)
                ? 1
                : other_block_n_rots;
      }
    }
  });
  DeviceOps<D>::template forall<launch_t>(
      mgr, n_output_intxns_for_rot_conn.size(0), count_intxns_for_rot_conn);
}

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
void connected_rotamer_fill_native(
    ContextManager& mgr,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 2, D> lockstep_group_for_block,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<int64_t, 1, D> n_output_intxns_for_rot_conn_offset,
    TView<Int, 1, D> rotconn_for_output_intxn,
    TView<Int, 2, D> dispatch_indices) {
  LAUNCH_BOX_32;
  int const max_n_conns = pose_stack_inter_block_connections.size(2);
  auto record_dispatch_indices = ([=] TMOL_DEVICE_FUNC(int index) {
    int const rotconn_ind = rotconn_for_output_intxn[index];
    int const rot_ind1 = rotconn_ind / (max_n_conns + 1);
    int const conn_ind1 = rotconn_ind % (max_n_conns + 1);
    int const pose_ind = pose_ind_for_rot[rot_ind1];
    dispatch_indices[0][index] = pose_ind;
    dispatch_indices[1][index] = rot_ind1;
    int rot_ind2;
    if (conn_ind1 == max_n_conns) {
      rot_ind2 = rot_ind1;
    } else {
      int const block_ind1 = block_ind_for_rot[rot_ind1];
      int const rotconn_offset =
          n_output_intxns_for_rot_conn_offset[rotconn_ind];
      int const local_rot_ind2 = index - rotconn_offset;
      int const block_ind2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                            [0];
      int const offset2 =
          same_rotamer_group(
              lockstep_group_for_block, pose_ind, block_ind1, block_ind2)
              ? rot_ind1 - first_rot_for_block[pose_ind][block_ind1]
              : local_rot_ind2;
      rot_ind2 = first_rot_for_block[pose_ind][block_ind2] + offset2;
    }
    dispatch_indices[2][index] = rot_ind2;
  });
  DeviceOps<D>::template forall<launch_t>(
      mgr, dispatch_indices.size(1), record_dispatch_indices);
}

}  // namespace tmol::score::common
