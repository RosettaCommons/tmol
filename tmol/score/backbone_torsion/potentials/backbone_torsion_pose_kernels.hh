#pragma once

#include <tmol/utility/portable/reduce_ops.hh>
#include <tmol/utility/portable/matrix.hh>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/context_manager.hh>

#include <tmol/score/unresolved_atom_types.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/counting.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/uaid_util.hh>

#include <tmol/score/backbone_torsion/potentials/params_types.hh>
#include <tmol/score/backbone_torsion/potentials/potentials.hh>

// Complete whole-pose science; callers supply zeroed output buffers.
// Framework bindings own allocation and registration, DeviceDispatch execution.

namespace tmol {
namespace score {
namespace backbone_torsion {
namespace potentials {

using tmol::portable::Vec;
template <typename Real>
using CoordQuad = tmol::portable::Mat<Real, 4, 3>;

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
void backbone_torsion_pose_forward(
    ContextManager& mgr,

    TView<Vec<Real, 3>, 1, Dev> rot_coords,
    TView<Int, 1, Dev> rot_coord_offset,
    TView<Int, 2, Dev> first_rot_for_block,
    TView<Int, 2, Dev> first_rot_block_type,

    TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_block_connections,

    TView<Int, 3, Dev> block_type_atom_downstream_of_conn,

    TView<Int, 2, Dev> block_type_rama_table,
    TView<Int, 2, Dev> block_type_omega_table,
    TView<Int, 1, Dev> block_type_lower_conn_ind,
    TView<Int, 1, Dev> block_type_upper_conn_ind,
    TView<Int, 1, Dev> block_type_is_pro,

    TView<UnresolvedAtomID<Int>, 2, Dev> block_type_torsion_atoms,

    TView<Real, 3, Dev> rama_tables,
    TView<RamaTableParams<Real>, 1, Dev> rama_table_params,
    TView<Real, 4, Dev> omega_tables,
    TView<RamaTableParams<Real>, 1, Dev> omega_table_params,
    bool output_block_pair_energies,
    bool compute_derivs,
    TView<Real, 4, Dev> V,
    TView<Vec<Real, 3>, 2, Dev> dV_dxyz) {
  using tmol::score::common::accumulate;
  using Real3 = Vec<Real, 3>;

  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const n_block_types = block_type_atom_downstream_of_conn.size(0);
  int const n_rama_tables = rama_tables.size(0);
  int const n_omega_tables = omega_tables.size(0);

  assert(first_rot_block_type.size(0) == n_poses);
  assert(first_rot_block_type.size(1) == max_n_blocks);

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);

  assert(block_type_rama_table.size(0) == n_block_types);
  assert(block_type_omega_table.size(0) == n_block_types);
  assert(block_type_lower_conn_ind.size(0) == n_block_types);
  assert(block_type_upper_conn_ind.size(0) == n_block_types);

  assert(block_type_torsion_atoms.size(0) == n_block_types);
  assert(block_type_torsion_atoms.size(1) == 12);

  assert(rama_table_params.size(0) == n_rama_tables);
  assert(omega_table_params.size(0) == n_omega_tables);

  LAUNCH_BOX_32;
  // Define nt
  CTA_LAUNCH_T_PARAMS;

  // RamaPrePro is a pseudo-1body term that's really a 2-body term:
  // the score for residue i changes if residue i+1 becomes proline.

  bool const accumulate_derivs = compute_derivs && !output_block_pair_energies;

  auto rama_omega_func = ([=] TMOL_DEVICE_FUNC(int ind) {
    int const pose_ind = ind / max_n_blocks;
    int const block_ind1 = ind % max_n_blocks;
    int const block_type1 = first_rot_block_type[pose_ind][block_ind1];
    int const rot_ind1 = first_rot_for_block[pose_ind][block_ind1];

    if (block_type1 == -1) {
      // Filter out blocks that are not real
      return;
    }

    bool block1_has_upper_neighbor = true;
    int const upper_conn = block_type_upper_conn_ind[block_type1];
    if (upper_conn < 0) {
      return;
    }
    int const block_ind2 =
        pose_stack_inter_block_connections[pose_ind][block_ind1][upper_conn][0];
    if (block_ind2 == -1) {
      // the upper neighbor is not a real residue: skip
      return;
    }
    int const block_type2 = first_rot_block_type[pose_ind][block_ind2];
    int const rot_ind2 = first_rot_for_block[pose_ind][block_ind2];

    int const upper_nbr_is_pro = block_type_is_pro[block_type2];
    int const rama_table_ind =
        block_type_rama_table[block_type1][upper_nbr_is_pro];
    int const omega_table_ind =
        block_type_omega_table[block_type1][upper_nbr_is_pro];

    int const rot_offset1 = rot_coord_offset[rot_ind1];

    // correction for cyclic peptides
    int const V_ind1 =
        output_block_pair_energies ? min(block_ind1, block_ind2) : 0;
    int const V_ind2 =
        output_block_pair_energies ? max(block_ind1, block_ind2) : 0;

    bool valid_phipsi = true;
    Vec<Int, 4> phi_ats;
    for (int i = 0; i < 4; ++i) {
      UnresolvedAtomID<Int> i_at = block_type_torsion_atoms[block_type1][i];
      phi_ats[i] = resolve_rotamer_pair_atom_from_uaid(
          i_at,
          rot_ind1,
          rot_ind2,
          block_ind1,
          block_ind2,
          block_type2,
          pose_ind,
          rot_coord_offset,
          first_rot_for_block,
          first_rot_block_type,
          pose_stack_inter_block_connections,
          block_type_atom_downstream_of_conn);
      valid_phipsi &= phi_ats[i] != -1;
    }
    CoordQuad<Real> phi_coords;
    if (valid_phipsi) {
      for (int i = 0; i < 4; ++i) {
        tmol::portable::row_ref(phi_coords, i) = rot_coords[phi_ats[i]];
      }
    }

    Vec<Int, 4> psi_ats;
    for (int i = 0; i < 4; ++i) {
      UnresolvedAtomID<Int> i_at = block_type_torsion_atoms[block_type1][i + 4];
      psi_ats[i] = resolve_rotamer_pair_atom_from_uaid(
          i_at,
          rot_ind1,
          rot_ind2,
          block_ind1,
          block_ind2,
          block_type2,
          pose_ind,
          rot_coord_offset,
          first_rot_for_block,
          first_rot_block_type,
          pose_stack_inter_block_connections,
          block_type_atom_downstream_of_conn);
      valid_phipsi &= psi_ats[i] != -1;
    }
    CoordQuad<Real> psi_coords;
    if (valid_phipsi) {
      for (int i = 0; i < 4; ++i) {
        tmol::portable::row_ref(psi_coords, i) = rot_coords[psi_ats[i]];
      }
    }

    // accumulate rama
    if (valid_phipsi && rama_table_ind >= 0) {
      if (accumulate_derivs) {
        auto rama = rama_V_dV<Dev, Real, Int>(
            phi_coords,
            psi_coords,
            rama_tables[rama_table_ind],
            tmol::portable::vec_from<Real, 2>(
                rama_table_params[rama_table_ind].bbstarts),
            tmol::portable::vec_from<Real, 2>(
                rama_table_params[rama_table_ind].bbsteps));
        accumulate<Dev, Real>::add(
            V[0][pose_ind][V_ind1][V_ind2], common::get<0>(rama));
        for (int j = 0; j < 4; ++j) {
          accumulate<Dev, Vec<Real, 3>>::add(
              dV_dxyz[0][phi_ats[j]], common::get<1>(rama).row(j));
          accumulate<Dev, Vec<Real, 3>>::add(
              dV_dxyz[0][psi_ats[j]], common::get<2>(rama).row(j));
        }
      } else {
        accumulate<Dev, Real>::add(
            V[0][pose_ind][V_ind1][V_ind2],
            rama_V<Dev, Real, Int>(
                phi_coords,
                psi_coords,
                rama_tables[rama_table_ind],
                tmol::portable::vec_from<Real, 2>(
                    rama_table_params[rama_table_ind].bbstarts),
                tmol::portable::vec_from<Real, 2>(
                    rama_table_params[rama_table_ind].bbsteps)));
      }
    }

    if (omega_table_ind < 0) {
      return;
    }

    CoordQuad<Real> omega_coords;
    Vec<Int, 4> omega_ats;
    for (int i = 0; i < 4; ++i) {
      // TO DO: replace "8" in the lookup below
      UnresolvedAtomID<Int> i_at = block_type_torsion_atoms[block_type1][i + 8];

      // Check to see if the omega uaids are actually defined for this block
      // type. If the atom offset [0] or the connection index [1] are both -1,
      // this is a sentinel for an undefined omega.
      if (i_at.atom_id == -1 && i_at.conn_id == -1)
        return;  // no omega and we already calculated rama

      omega_ats[i] = resolve_rotamer_pair_atom_from_uaid(
          i_at,
          rot_ind1,
          rot_ind2,
          block_ind1,
          block_ind2,
          block_type2,
          pose_ind,
          rot_coord_offset,
          first_rot_for_block,
          first_rot_block_type,
          pose_stack_inter_block_connections,
          block_type_atom_downstream_of_conn);

      // omega_atom_ind == -1 -> UAID resolution failed
      if (omega_ats[i] == -1)
        return;  // no omega and we already calculated rama
    }

    for (int i = 0; i < 4; ++i) {
      tmol::portable::row_ref(omega_coords, i) = rot_coords[omega_ats[i]];
    }

    if (valid_phipsi && omega_table_ind >= 0) {
      if (accumulate_derivs) {
        auto omega = omega_bbdep_V_dV<Dev, Real, Int>(
            phi_coords,
            psi_coords,
            omega_coords,
            omega_tables[omega_table_ind][0],
            omega_tables[omega_table_ind][1],
            tmol::portable::vec_from<Real, 2>(
                omega_table_params[omega_table_ind].bbstarts),
            tmol::portable::vec_from<Real, 2>(
                omega_table_params[omega_table_ind].bbsteps),
            32.8);
        accumulate<Dev, Real>::add(
            V[1][pose_ind][V_ind1][V_ind2], common::get<0>(omega));
        for (int j = 0; j < 4; ++j) {
          // omega : [V, dVdphi, dVdpsi, dVdomega]
          accumulate<Dev, Vec<Real, 3>>::add(
              dV_dxyz[1][phi_ats[j]], common::get<1>(omega).row(j));
          accumulate<Dev, Vec<Real, 3>>::add(
              dV_dxyz[1][psi_ats[j]], common::get<2>(omega).row(j));
          accumulate<Dev, Vec<Real, 3>>::add(
              dV_dxyz[1][omega_ats[j]], common::get<3>(omega).row(j));
        }
      } else {
        accumulate<Dev, Real>::add(
            V[1][pose_ind][V_ind1][V_ind2],
            omega_bbdep_V<Dev, Real, Int>(
                phi_coords,
                psi_coords,
                omega_coords,
                omega_tables[omega_table_ind][0],
                omega_tables[omega_table_ind][1],
                tmol::portable::vec_from<Real, 2>(
                    omega_table_params[omega_table_ind].bbstarts),
                tmol::portable::vec_from<Real, 2>(
                    omega_table_params[omega_table_ind].bbsteps),
                32.8));
      }
    } else {
      // if rama is undefined, fall back to old version
      if (accumulate_derivs) {
        auto omega = omega_V_dV<Dev, Real, Int>(omega_coords, 32.8);
        accumulate<Dev, Real>::add(
            V[1][pose_ind][V_ind1][V_ind2], common::get<0>(omega));
        for (int j = 0; j < 4; ++j) {
          // omega : [V, dVdomega]
          accumulate<Dev, Vec<Real, 3>>::add(
              dV_dxyz[1][omega_ats[j]], common::get<1>(omega).row(j));
        }
      } else {
        accumulate<Dev, Real>::add(
            V[1][pose_ind][V_ind1][V_ind2],
            omega_V<Real>(omega_coords, Real(32.8)));
      }
    }
  });

  DeviceDispatch<Dev>::template forall_grouped<launch_t>(
      mgr, n_poses, max_n_blocks, rama_omega_func);
};

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
void backbone_torsion_pose_backward(
    ContextManager& mgr,

    TView<Vec<Real, 3>, 1, Dev> rot_coords,
    TView<Int, 1, Dev> rot_coord_offset,
    TView<Int, 2, Dev> first_rot_for_block,
    TView<Int, 2, Dev> first_rot_block_type,

    TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_block_connections,

    TView<Int, 3, Dev> block_type_atom_downstream_of_conn,

    TView<Int, 2, Dev> block_type_rama_table,
    TView<Int, 2, Dev> block_type_omega_table,
    TView<Int, 1, Dev> block_type_lower_conn_ind,
    TView<Int, 1, Dev> block_type_upper_conn_ind,
    TView<Int, 1, Dev> block_type_is_pro,

    TView<UnresolvedAtomID<Int>, 2, Dev> block_type_torsion_atoms,

    TView<Real, 3, Dev> rama_tables,
    TView<RamaTableParams<Real>, 1, Dev> rama_table_params,
    TView<Real, 4, Dev> omega_tables,
    TView<RamaTableParams<Real>, 1, Dev> omega_table_params,

    TView<Real, 4, Dev> dTdV,
    TView<Vec<Real, 3>, 2, Dev> dV_dxyz) {
  using tmol::score::common::accumulate;
  using Real3 = Vec<Real, 3>;

  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const n_block_types = block_type_atom_downstream_of_conn.size(0);
  int const n_rama_tables = rama_tables.size(0);
  int const n_omega_tables = omega_tables.size(0);

  assert(first_rot_block_type.size(0) == n_poses);
  assert(first_rot_block_type.size(1) == max_n_blocks);

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);

  assert(block_type_rama_table.size(0) == n_block_types);
  assert(block_type_omega_table.size(0) == n_block_types);
  assert(block_type_lower_conn_ind.size(0) == n_block_types);
  assert(block_type_upper_conn_ind.size(0) == n_block_types);

  assert(block_type_torsion_atoms.size(0) == n_block_types);
  assert(block_type_torsion_atoms.size(1) == 12);

  assert(rama_table_params.size(0) == n_rama_tables);
  assert(omega_table_params.size(0) == n_omega_tables);

  LAUNCH_BOX_32;
  // Define nt
  CTA_LAUNCH_T_PARAMS;

  auto rama_omega_func = ([=] TMOL_DEVICE_FUNC(int ind) {
    int const pose_ind = ind / max_n_blocks;
    int const block_ind1 = ind % max_n_blocks;
    int const block_type1 = first_rot_block_type[pose_ind][block_ind1];
    int const rot_ind1 = first_rot_for_block[pose_ind][block_ind1];

    if (block_type1 == -1) {
      // Filter out blocks that are not real
      return;
    }

    bool block1_has_upper_neighbor = true;
    int const upper_conn = block_type_upper_conn_ind[block_type1];
    if (upper_conn < 0) {
      return;
    }
    int const block_ind2 =
        pose_stack_inter_block_connections[pose_ind][block_ind1][upper_conn][0];
    if (block_ind2 == -1) {
      // the upper neighbor is not a real residue: skip
      return;
    }

    // The forward pass writes a block pair at its canonical (min, max) slot so
    // a cycle's closing bond does not collide with the bond that opens it;
    // read the same slot here or that bond's gradient is silently zero.
    int const dTdV_ind1 = min(block_ind1, block_ind2);
    int const dTdV_ind2 = max(block_ind1, block_ind2);

    if (dTdV[0][pose_ind][dTdV_ind1][dTdV_ind2] == 0
        && dTdV[1][pose_ind][dTdV_ind1][dTdV_ind2] == 0) {
      return;
    }

    int const block_type2 = first_rot_block_type[pose_ind][block_ind2];
    int const rot_ind2 = first_rot_for_block[pose_ind][block_ind2];

    int const upper_nbr_is_pro = block_type_is_pro[block_type2];
    int const rama_table_ind =
        block_type_rama_table[block_type1][upper_nbr_is_pro];
    int const omega_table_ind =
        block_type_omega_table[block_type1][upper_nbr_is_pro];

    int const rot_offset1 = rot_coord_offset[rot_ind1];

    bool valid_phipsi = true;
    Vec<Int, 4> phi_ats;
    for (int i = 0; i < 4; ++i) {
      UnresolvedAtomID<Int> i_at = block_type_torsion_atoms[block_type1][i];
      phi_ats[i] = resolve_rotamer_pair_atom_from_uaid(
          i_at,
          rot_ind1,
          rot_ind2,
          block_ind1,
          block_ind2,
          block_type2,
          pose_ind,
          rot_coord_offset,
          first_rot_for_block,
          first_rot_block_type,
          pose_stack_inter_block_connections,
          block_type_atom_downstream_of_conn);
      valid_phipsi &= phi_ats[i] != -1;
    }
    CoordQuad<Real> phi_coords;
    if (valid_phipsi) {
      for (int i = 0; i < 4; ++i) {
        tmol::portable::row_ref(phi_coords, i) = rot_coords[phi_ats[i]];
      }
    }

    Vec<Int, 4> psi_ats;
    for (int i = 0; i < 4; ++i) {
      UnresolvedAtomID<Int> i_at = block_type_torsion_atoms[block_type1][i + 4];
      psi_ats[i] = resolve_rotamer_pair_atom_from_uaid(
          i_at,
          rot_ind1,
          rot_ind2,
          block_ind1,
          block_ind2,
          block_type2,
          pose_ind,
          rot_coord_offset,
          first_rot_for_block,
          first_rot_block_type,
          pose_stack_inter_block_connections,
          block_type_atom_downstream_of_conn);
      valid_phipsi &= psi_ats[i] != -1;
    }
    CoordQuad<Real> psi_coords;
    if (valid_phipsi) {
      for (int i = 0; i < 4; ++i) {
        tmol::portable::row_ref(psi_coords, i) = rot_coords[psi_ats[i]];
      }
    }

    // accumulate rama
    Real rama_block_weight = dTdV[0][pose_ind][dTdV_ind1][dTdV_ind2];
    if (valid_phipsi && rama_table_ind >= 0) {
      auto rama = rama_V_dV<Dev, Real, Int>(
          phi_coords,
          psi_coords,
          rama_tables[rama_table_ind],
          tmol::portable::vec_from<Real, 2>(
              rama_table_params[rama_table_ind].bbstarts),
          tmol::portable::vec_from<Real, 2>(
              rama_table_params[rama_table_ind].bbsteps));
      for (int j = 0; j < 4; ++j) {
        accumulate<Dev, Vec<Real, 3>>::add(
            dV_dxyz[0][phi_ats[j]],
            common::get<1>(rama).row(j) * rama_block_weight);
        accumulate<Dev, Vec<Real, 3>>::add(
            dV_dxyz[0][psi_ats[j]],
            common::get<2>(rama).row(j) * rama_block_weight);
      }
    }

    if (omega_table_ind < 0) {
      return;
    }

    CoordQuad<Real> omega_coords;
    Vec<Int, 4> omega_ats;
    for (int i = 0; i < 4; ++i) {
      UnresolvedAtomID<Int> i_at = block_type_torsion_atoms[block_type1][i + 8];

      // Check to see if the omega uaids are actually defined for this block
      // type. If the atom offset [0] or the connection index [1] are both -1,
      // this is a sentinel for an undefined omega.
      if (i_at.atom_id == -1 && i_at.conn_id == -1)
        return;  // no omega and we already calculated rama

      omega_ats[i] = resolve_rotamer_pair_atom_from_uaid(
          i_at,
          rot_ind1,
          rot_ind2,
          block_ind1,
          block_ind2,
          block_type2,
          pose_ind,
          rot_coord_offset,
          first_rot_for_block,
          first_rot_block_type,
          pose_stack_inter_block_connections,
          block_type_atom_downstream_of_conn);

      // omega_atom_ind == -1 -> UAID resolution failed
      if (omega_ats[i] == -1)
        return;  // no omega and we already calculated rama
    }

    for (int i = 0; i < 4; ++i) {
      tmol::portable::row_ref(omega_coords, i) = rot_coords[omega_ats[i]];
    }

    Real omega_block_weight = dTdV[1][pose_ind][dTdV_ind1][dTdV_ind2];
    if (valid_phipsi && omega_table_ind >= 0) {
      auto omega = omega_bbdep_V_dV<Dev, Real, Int>(
          phi_coords,
          psi_coords,
          omega_coords,
          omega_tables[omega_table_ind][0],
          omega_tables[omega_table_ind][1],
          tmol::portable::vec_from<Real, 2>(
              omega_table_params[omega_table_ind].bbstarts),
          tmol::portable::vec_from<Real, 2>(
              omega_table_params[omega_table_ind].bbsteps),
          32.8);
      for (int j = 0; j < 4; ++j) {
        // omega : [V, dVdphi, dVdpsi, dVdomega]
        accumulate<Dev, Vec<Real, 3>>::add(
            dV_dxyz[1][phi_ats[j]],
            common::get<1>(omega).row(j) * omega_block_weight);
        accumulate<Dev, Vec<Real, 3>>::add(
            dV_dxyz[1][psi_ats[j]],
            common::get<2>(omega).row(j) * omega_block_weight);
        accumulate<Dev, Vec<Real, 3>>::add(
            dV_dxyz[1][omega_ats[j]],
            common::get<3>(omega).row(j) * omega_block_weight);
      }
    } else {
      // if rama is undefined, fall back to old version
      auto omega = omega_V_dV<Dev, Real, Int>(omega_coords, 32.8);
      for (int j = 0; j < 4; ++j) {
        // omega : [V, dVdomega]
        accumulate<Dev, Vec<Real, 3>>::add(
            dV_dxyz[1][omega_ats[j]],
            common::get<1>(omega).row(j) * omega_block_weight);
      }
    }
  });

  DeviceDispatch<Dev>::template forall_grouped<launch_t>(
      mgr, n_poses, max_n_blocks, rama_omega_func);
};

}  // namespace potentials
}  // namespace backbone_torsion
}  // namespace score
}  // namespace tmol
