#pragma once
#include <tmol/utility/portable/matrix.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/score/common/counting.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/io/details/compiled/params_types.hh>

namespace tmol::io::details::compiled {
using tmol::portable::Vec;
template <typename Real, typename Int, Device D>
struct HisTautContext {
  TView<Vec<Real, 3>, 3, D> coords;
  TView<Int, 2, D> res_types, res_type_variants;
  TView<int64_t, 1, D> his_pose_ind, his_res_ind;
  TView<bool, 3, D> atom_is_present;
  TView<HisAtomIndsInCanonicalOrdering<Int>, 1, D> his_atom_inds;
  TView<int64_t, 3, D> his_remapping_dst_index;
  TView<int64_t, 2, D> his_taut;
};
// Caller initializes the remapping to identity and state output to zero. Only
// selected histidines change variants/remapping; coordinates remain borrowed.
template <
    template <Device> class DeviceOps,
    Device Dev,
    typename Real,
    typename Int,
    class Context>
void launch_resolve_his_taut(Context& mgr, HisTautContext<Real, Int, Dev> c) {
  int const n_poses = c.coords.size(0);
  int const max_n_blocks = c.coords.size(1);
  int const max_n_block_atoms = c.coords.size(2);
  int const n_his = score::common::checked_dispatch_size(
      c.his_pose_ind.size(0), "histidine resolution");

  assert(c.res_types.size(0) == n_poses);
  assert(c.res_types.size(1) == max_n_blocks);
  assert(c.res_type_variants.size(0) == n_poses);
  assert(c.res_type_variants.size(1) == max_n_blocks);
  assert(c.his_res_ind.size(0) == n_his);
  assert(c.his_atom_inds.size(0) >= 1);
  assert(c.his_remapping_dst_index.size(0) == n_poses);
  assert(c.his_remapping_dst_index.size(1) == max_n_blocks);
  assert(c.his_remapping_dst_index.size(2) == max_n_block_atoms);

  for (int d = 0; d < 3; ++d)
    assert(c.atom_is_present.size(d) == c.coords.size(d));
  assert(c.his_taut.size(0) == n_poses && c.his_taut.size(1) == max_n_blocks);

  LAUNCH_BOX_32;

  auto f_his_resolver = ([=] TMOL_DEVICE_FUNC(int ind) {
    int const ip = c.his_pose_ind[ind];
    int const ir = c.his_res_ind[ind];
    if (ip < 0 || ip >= n_poses || ir < 0 || ir >= max_n_blocks) return;
    auto row = c.his_atom_inds.size(0) == 1 ? 0 : c.res_types[ip][ir];
    if (row < 0 || row >= c.his_atom_inds.size(0)) return;
    HisAtomIndsInCanonicalOrdering<Int> atom_inds = c.his_atom_inds[row];
    Int slots[] = {
        atom_inds.his_ND1_in_co,
        atom_inds.his_NE2_in_co,
        atom_inds.his_HD1_in_co,
        atom_inds.his_HE2_in_co,
        atom_inds.his_HN_in_co,
        atom_inds.his_NH_in_co,
        atom_inds.his_NN_in_co,
        atom_inds.his_CG_in_co};
    for (auto slot : slots)
      if (slot < 0 || slot >= max_n_block_atoms) return;
    bool const ND1_present = c.atom_is_present[ip][ir][atom_inds.his_ND1_in_co];
    bool const NE2_present = c.atom_is_present[ip][ir][atom_inds.his_NE2_in_co];
    bool const HD1_present = c.atom_is_present[ip][ir][atom_inds.his_HD1_in_co];
    bool const HE2_present = c.atom_is_present[ip][ir][atom_inds.his_HE2_in_co];
    bool const HN_present = c.atom_is_present[ip][ir][atom_inds.his_HN_in_co];
    bool const NH_present = c.atom_is_present[ip][ir][atom_inds.his_NH_in_co];
    bool const NN_present = c.atom_is_present[ip][ir][atom_inds.his_NN_in_co];
    bool const CG_present = c.atom_is_present[ip][ir][atom_inds.his_CG_in_co];

    int state = his_taut_unresolved;
    int variant = his_taut_variant_unresolved;

    if (HD1_present && !HE2_present) {
      state = his_taut_HD1;
      variant = his_taut_variant_ND1_protonated;
    } else if (HE2_present && !HD1_present) {
      state = his_taut_HE2;
      variant = his_taut_variant_NE2_protonated;
    } else if (
        HN_present && !HD1_present && !HE2_present && ND1_present
        && NE2_present) {
      Vec<Real, 3> his_ND1_coord = c.coords[ip][ir][atom_inds.his_ND1_in_co];
      Vec<Real, 3> his_NE2_coord = c.coords[ip][ir][atom_inds.his_NE2_in_co];
      Vec<Real, 3> his_HN_coord = c.coords[ip][ir][atom_inds.his_HN_in_co];
      Real dis2_ND1 = (his_ND1_coord - his_HN_coord).squaredNorm();
      Real dis2_NE2 = (his_NE2_coord - his_HN_coord).squaredNorm();

      if (dis2_ND1 < dis2_NE2) {
        state = his_taut_HD1;
        variant = his_taut_variant_ND1_protonated;
        c.his_remapping_dst_index[ip][ir][atom_inds.his_HD1_in_co] =
            atom_inds.his_HN_in_co;
      } else {
        state = his_taut_HE2;
        variant = his_taut_variant_NE2_protonated;
        c.his_remapping_dst_index[ip][ir][atom_inds.his_HE2_in_co] =
            atom_inds.his_HN_in_co;
      }
    } else if (NH_present && NN_present && HN_present && CG_present) {
      Vec<Real, 3> his_NH_coord = c.coords[ip][ir][atom_inds.his_NH_in_co];
      Vec<Real, 3> his_NN_coord = c.coords[ip][ir][atom_inds.his_NN_in_co];
      Vec<Real, 3> his_CG_coord = c.coords[ip][ir][atom_inds.his_CG_in_co];
      Real dis2_NH = (his_NH_coord - his_CG_coord).squaredNorm();
      Real dis2_NN = (his_NN_coord - his_CG_coord).squaredNorm();
      if (dis2_NH < dis2_NN) {
        state = his_taut_NH_is_ND1;
        c.his_remapping_dst_index[ip][ir][atom_inds.his_ND1_in_co] =
            atom_inds.his_NH_in_co;
        c.his_remapping_dst_index[ip][ir][atom_inds.his_HD1_in_co] =
            atom_inds.his_HN_in_co;
        c.his_remapping_dst_index[ip][ir][atom_inds.his_NE2_in_co] =
            atom_inds.his_NN_in_co;
        variant = his_taut_variant_ND1_protonated;
      } else {
        state = his_taut_NN_is_ND1;
        c.his_remapping_dst_index[ip][ir][atom_inds.his_ND1_in_co] =
            atom_inds.his_NN_in_co;
        c.his_remapping_dst_index[ip][ir][atom_inds.his_HE2_in_co] =
            atom_inds.his_HN_in_co;
        c.his_remapping_dst_index[ip][ir][atom_inds.his_NE2_in_co] =
            atom_inds.his_NH_in_co;
        variant = his_taut_variant_NE2_protonated;
      }

    } else if (!HD1_present && !HE2_present && !HN_present) {
      // arbitrary choice: go with his_taut_HE2
      state = his_taut_HE2;
      variant = his_taut_variant_NE2_protonated;
    } else if (HD1_present && HE2_present) {
      state = his_taut_HD1_HE2;
      variant = his_taut_variant_both_ND1_and_NE2_protonated;
    }
    c.his_taut[ip][ir] = state;
    c.res_type_variants[ip][ir] = variant;
  });

  DeviceOps<Dev>::template forall<launch_t>(mgr, n_his, f_his_resolver);
}
}  // namespace tmol::io::details::compiled
