#pragma once
#include <tmol/score/metal/potentials/metal_coordination_pose_kernels.hh>
namespace tmol::score::metal::potentials {
// Caller-owned sparse fan/site/bridge entries with explicit chemical identity.
// Counts and score/gradient storage start at zero; offsets are exclusive.
// The adapter checks dispatch extents and preserves each connection-specific
// code alongside canonical pair indices. Backward uses arbitrary entry weights.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
void metal_rotamer_count_native(
    ContextManager& mgr,
    TView<Int, 2, D> first_rot_block_type,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Int, 2, D> n_rots_for_block,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 2, D> conn_atom,
    TView<Int, 2, D> conn_metal,
    TView<Int, 2, D> conn_virt,
    TView<Int, 2, D> conn_key,
    TView<MetalSiteParams<Real>, 3, D> site_params,
    TView<Vec<Int, 2>, 2, D> fan_atoms,
    TView<Int, 3, D> bridge_internal_metal,
    TView<Real, 3, D> bridge_internal_d0,
    TView<int64_t, 1, D> n_energies_for_rot) {
  LAUNCH_BOX_32;
  int const max_n_conns = conn_virt.size(1);
  int const max_n_internal = bridge_internal_metal.size(2);
  auto count_dispatch_indices = ([=] TMOL_DEVICE_FUNC(int rot_ind) {
    int const pose_ind = pose_ind_for_rot[rot_ind];
    int const block_ind = block_ind_for_rot[rot_ind];
    int const block_type = block_type_ind_for_rot[rot_ind];
    if (pose_ind < 0 || block_ind < 0 || block_type < 0) {
      return;
    }
    int64_t n_energies = fan_atoms[block_type][0][0] >= 0 ? 1 : 0;
    for (int conn = 0; conn < max_n_conns; conn++) {
      if (conn_virt[block_type][conn] == NOT_A_SITE) {
        continue;
      }
      int const other_block =
          pose_stack_inter_block_connections[pose_ind][block_ind][conn][0];
      if (other_block < 0) {
        continue;
      }
      n_energies += n_rots_for_block[pose_ind][other_block];
    }
    for_each_bridge<Int, Real, D>(
        pose_ind,
        block_ind,
        block_type,
        max_n_conns,
        max_n_internal,
        first_rot_block_type,
        pose_stack_inter_block_connections,
        conn_atom,
        conn_metal,
        conn_virt,
        conn_key,
        site_params,
        bridge_internal_metal,
        bridge_internal_d0,
        [&](int, int other, int, int, int, Real, Real, int) {
          n_energies += n_rots_for_block[pose_ind][other];
        });
    n_energies_for_rot[rot_ind] = n_energies;
  });
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, n_energies_for_rot.size(0), count_dispatch_indices);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
void metal_rotamer_fill_native(
    ContextManager& mgr,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Int, 2, D> n_rots_for_block,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 2, D> conn_atom,
    TView<Int, 2, D> conn_metal,
    TView<Int, 2, D> conn_virt,
    TView<Int, 2, D> conn_key,
    TView<MetalSiteParams<Real>, 3, D> site_params,
    TView<Vec<Int, 2>, 2, D> fan_atoms,
    TView<Int, 3, D> bridge_internal_metal,
    TView<Real, 3, D> bridge_internal_d0,
    TView<int64_t, 1, D> n_energies_for_rot,
    TView<int64_t, 1, D> n_energies_for_rot_offset,
    TView<Int, 2, D> dispatch_indices,
    TView<Int, 2, D> terms_for_dispatch,
    int max_n_energies_for_rot,
    int candidate_dispatch) {
  LAUNCH_BOX_32;
  int const max_n_conns = conn_virt.size(1);
  int const max_n_internal = bridge_internal_metal.size(2);
  auto mark_dispatch_indices = ([=] TMOL_DEVICE_FUNC(int ind) {
    int const rot_ind1 = ind / max_n_energies_for_rot;
    int const local_ind = ind % max_n_energies_for_rot;
    if (local_ind >= n_energies_for_rot[rot_ind1]) {
      return;
    }
    int const pose_ind = pose_ind_for_rot[rot_ind1];
    int const block_ind1 = block_ind_for_rot[rot_ind1];
    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const sparse_ind = n_energies_for_rot_offset[rot_ind1] + local_ind;
    dispatch_indices[0][sparse_ind] = pose_ind;

    int remaining = local_ind;
    if (fan_atoms[block_type1][0][0] >= 0) {
      if (remaining == 0) {
        dispatch_indices[1][sparse_ind] = rot_ind1;
        dispatch_indices[2][sparse_ind] = rot_ind1;
        terms_for_dispatch[0][sparse_ind] = rot_ind1;
        terms_for_dispatch[1][sparse_ind] = rot_ind1;
        terms_for_dispatch[2][sparse_ind] = -1;
        return;
      }
      remaining -= 1;
    }
    for (int conn1 = 0; conn1 < max_n_conns; conn1++) {
      if (conn_virt[block_type1][conn1] == NOT_A_SITE) {
        continue;
      }
      int const block_ind2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn1][0];
      if (block_ind2 < 0) {
        continue;
      }
      int const n_rots2 = n_rots_for_block[pose_ind][block_ind2];
      if (remaining < n_rots2) {
        int const rot_ind2 =
            first_rot_for_block[pose_ind][block_ind2] + remaining;
        // lower block first, as the other pair terms report it
        bool const metal_first = block_ind1 < block_ind2;
        dispatch_indices[1][sparse_ind] = metal_first ? rot_ind1 : rot_ind2;
        dispatch_indices[2][sparse_ind] = metal_first ? rot_ind2 : rot_ind1;
        terms_for_dispatch[0][sparse_ind] = rot_ind1;
        terms_for_dispatch[1][sparse_ind] = rot_ind2;
        terms_for_dispatch[2][sparse_ind] = conn1;
        return;
      }
      remaining -= n_rots2;
    }
    // bridges follow, coded -2 - code: -1 is the fan
    bool placed = false;
    for_each_bridge<Int, Real, D>(
        pose_ind,
        block_ind1,
        block_type1,
        max_n_conns,
        max_n_internal,
        first_rot_block_type,
        pose_stack_inter_block_connections,
        conn_atom,
        conn_metal,
        conn_virt,
        conn_key,
        site_params,
        bridge_internal_metal,
        bridge_internal_d0,
        [&](int code, int other, int, int, int, Real, Real, int) {
          if (placed) {
            return;
          }
          int const n_rots2 = n_rots_for_block[pose_ind][other];
          if (remaining < n_rots2) {
            int const rot_ind2 =
                first_rot_for_block[pose_ind][other] + remaining;
            bool const first = block_ind1 < other;
            dispatch_indices[1][sparse_ind] = first ? rot_ind1 : rot_ind2;
            dispatch_indices[2][sparse_ind] = first ? rot_ind2 : rot_ind1;
            terms_for_dispatch[0][sparse_ind] = rot_ind1;
            terms_for_dispatch[1][sparse_ind] = rot_ind2;
            terms_for_dispatch[2][sparse_ind] = -2 - code;
            placed = true;
            return;
          }
          remaining -= n_rots2;
        });
  });
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, candidate_dispatch, mark_dispatch_indices);
}

template <
    bool Backward,
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
void metal_rotamer_score_native(
    ContextManager& mgr,
    TView<Vec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 2, D> first_rot_block_type,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 2, D> conn_atom,
    TView<Int, 2, D> conn_metal,
    TView<Int, 2, D> conn_virt,
    TView<Int, 2, D> conn_key,
    TView<MetalSiteParams<Real>, 3, D> site_params,
    TView<Vec<Int, 2>, 2, D> fan_atoms,
    TView<MetalFanParams<Real>, 2, D> fan_params,
    TView<Int, 3, D> bridge_internal_metal,
    TView<Real, 3, D> bridge_internal_d0,
    TView<MetalBridgeParams<Real>, 1, D> bridge_params,
    TView<Int, 2, D> dispatch_indices,
    TView<Int, 2, D> terms_for_dispatch,
    bool compute_derivs,
    bool output_block_pair_energies,
    TView<Real, 2, D> V,
    TView<Vec<Real, 3>, 2, D> dV_dx,
    TView<Real, 2, D> dTdV) {
  LAUNCH_BOX_32;
  int const max_n_conns = conn_virt.size(1);
  int const max_n_internal = bridge_internal_metal.size(2);
  int const max_n_fan = fan_atoms.size(1);
  auto eval_energies = ([=] TMOL_DEVICE_FUNC(int dispatch_ind) {
    Real const weight = Backward ? dTdV[0][dispatch_ind] : Real(1);
    if (Backward && weight == 0) return;
    int const pose_ind =
        Backward ? pose_ind_for_rot[terms_for_dispatch[0][dispatch_ind]]
                 : dispatch_indices[0][dispatch_ind];
    int const rot_ind1 = terms_for_dispatch[0][dispatch_ind];
    int const rot_ind2 = terms_for_dispatch[1][dispatch_ind];
    int const conn1 = terms_for_dispatch[2][dispatch_ind];
    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const offset1 = rot_coord_offset[rot_ind1];
    Real* V_out = Backward ? nullptr
                           : (output_block_pair_energies ? &V[0][dispatch_ind]
                                                         : &V[0][pose_ind]);

    if (conn1 <= -2) {
      int const bridge_pose = pose_ind_for_rot[rot_ind1];
      auto const bridge = bridge_from_code<Int, Real, D>(
          -2 - conn1,
          bridge_pose,
          block_ind_for_rot[rot_ind1],
          block_type1,
          block_type_ind_for_rot[rot_ind2],
          max_n_conns,
          max_n_internal,
          first_rot_block_type,
          pose_stack_inter_block_connections,
          conn_metal,
          conn_key,
          site_params,
          bridge_internal_metal,
          bridge_internal_d0);
      if (bridge.valid) {
        accumulate_metal_bridge<Real, D>(
            rot_coords,
            offset1 + bridge.metal1,
            rot_coord_offset[rot_ind2] + bridge.metal2,
            bridge.d1,
            bridge.d2,
            bridge_params[bridge.key],
            dV_dx,
            Backward || compute_derivs,
            weight,
            V_out);
      }
      return;
    }

    if (conn1 < 0) {
      for (int i = 0; i < max_n_fan; i++) {
        Vec<Int, 2> const pair = fan_atoms[block_type1][i];
        if (pair[0] < 0) {
          break;
        }
        accumulate_metal_fan_pair<Real, D>(
            rot_coords,
            offset1 + pair[0],
            offset1 + pair[1],
            fan_params[block_type1][i],
            dV_dx,
            Backward || compute_derivs,
            weight,
            V_out);
      }
      return;
    }

    int const block_ind1 = block_ind_for_rot[rot_ind1];
    int const conn2 =
        pose_stack_inter_block_connections[pose_ind][block_ind1][conn1][1];
    int const block_type2 = block_type_ind_for_rot[rot_ind2];
    int const key = conn_key[block_type2][conn2];
    if (key < 0) {
      return;
    }
    int const virt = conn_virt[block_type1][conn1];
    accumulate_metal_site<Real, D>(
        rot_coords,
        offset1 + conn_metal[block_type1][conn1],
        virt >= 0 ? offset1 + virt : -1,
        rot_coord_offset[rot_ind2] + conn_atom[block_type2][conn2],
        site_params[block_type1][conn1][key],
        dV_dx,
        Backward || compute_derivs,
        weight,
        V_out);
  });
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, terms_for_dispatch.size(1), eval_energies);
}

}  // namespace tmol::score::metal::potentials
