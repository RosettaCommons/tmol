#pragma once

#include <tmol/utility/portable/reduce_ops.hh>
#include <tmol/utility/portable/matrix.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>

#include "params_types.hh"
#include "potentials.hh"

// Shared whole-pose metal coordination and coordinate derivatives. Callers
// supply zeroed output views; DeviceDispatch controls execution only.
namespace tmol {
namespace score {
namespace metal {
namespace potentials {
using tmol::portable::Vec;

// conn_virt values that are not a virtual atom index
constexpr int NOT_A_SITE = -2;

// Each metal block owns its site connections: the fan is a one-body energy on
// the block, and each filled site is a two-body energy between the site's
// metal atom (a block may hold several) and the block across the connection.
// Donor blocks never initiate, so every site is scored once.

// Bridges: two metals in different blocks bonded to one donor atom. Each is
// visited once, from block1 as the metal whose own atom it charges:
//  - through a donor in another block, from the lower of the two metal
//    blocks (metal site conn1 on block1 -> donor -> another site on "other");
//  - through a donor in block1 itself, an internal satisfier of block1's
//    metals that also fills a site on "other".
// f(code, other_block, other_type, metal1_atom, metal2_atom, d1, d2, key);
// code names the bridge for bridge_from_code
template <typename Int, typename Real, tmol::Device D, typename Func>
TMOL_DEVICE_FUNC void for_each_bridge(
    int pose_ind,
    int block_ind1,
    int block_type1,
    int max_n_conns,
    int max_n_internal,
    TView<Int, 2, D> first_rot_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 2, D> conn_atom,
    TView<Int, 2, D> conn_metal,
    TView<Int, 2, D> conn_virt,
    TView<Int, 2, D> conn_key,
    TView<MetalSiteParams<Real>, 3, D> site_params,
    TView<Int, 3, D> bridge_internal_metal,
    TView<Real, 3, D> bridge_internal_d0,
    Func f) {
  for (int conn1 = 0; conn1 < max_n_conns; conn1++) {
    int const partner =
        pose_stack_inter_block_connections[pose_ind][block_ind1][conn1][0];
    if (partner < 0) {
      continue;
    }
    int const pconn =
        pose_stack_inter_block_connections[pose_ind][block_ind1][conn1][1];
    int const ptype = first_rot_block_type[pose_ind][partner];
    if (ptype < 0) {
      continue;
    }
    if (conn_virt[block_type1][conn1] != NOT_A_SITE) {
      int const key = conn_key[ptype][pconn];
      if (key < 0) {
        continue;
      }
      Int const donor_atom = conn_atom[ptype][pconn];
      for (int c = 0; c < max_n_conns; c++) {
        if (c == pconn || conn_atom[ptype][c] != donor_atom) {
          continue;
        }
        int const other =
            pose_stack_inter_block_connections[pose_ind][partner][c][0];
        if (other <= block_ind1) {
          continue;
        }
        int const oconn =
            pose_stack_inter_block_connections[pose_ind][partner][c][1];
        int const otype = first_rot_block_type[pose_ind][other];
        if (otype < 0 || conn_virt[otype][oconn] == NOT_A_SITE) {
          continue;
        }
        f(conn1 * max_n_conns + c,
          other,
          otype,
          conn_metal[block_type1][conn1],
          conn_metal[otype][oconn],
          site_params[block_type1][conn1][key].d0,
          site_params[otype][oconn][key].d0,
          key);
      }
    } else if (conn_virt[ptype][pconn] != NOT_A_SITE) {
      int const key = conn_key[block_type1][conn1];
      if (key < 0) {
        continue;
      }
      for (int k = 0; k < max_n_internal; k++) {
        Int const metal = bridge_internal_metal[block_type1][conn1][k];
        if (metal < 0) {
          break;
        }
        f(max_n_conns * max_n_conns + conn1 * max_n_internal + k,
          partner,
          ptype,
          metal,
          conn_metal[ptype][pconn],
          bridge_internal_d0[block_type1][conn1][k],
          site_params[ptype][pconn][key].d0,
          key);
      }
    }
  }
}

// The bridge for_each_bridge named by code, between rotamers of block_ind1
// (block type type1) and of the other metal's block (block type type2).
template <typename Int, typename Real, tmol::Device D>
struct Bridge {
  bool valid;
  Int metal1;
  Int metal2;
  Real d1;
  Real d2;
  int key;
};

template <typename Int, typename Real, tmol::Device D>
TMOL_DEVICE_FUNC Bridge<Int, Real, D> bridge_from_code(
    int code,
    int pose_ind,
    int block_ind1,
    int type1,
    int type2,
    int max_n_conns,
    int max_n_internal,
    TView<Int, 2, D> first_rot_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 2, D> conn_metal,
    TView<Int, 2, D> conn_key,
    TView<MetalSiteParams<Real>, 3, D> site_params,
    TView<Int, 3, D> bridge_internal_metal,
    TView<Real, 3, D> bridge_internal_d0) {
  Bridge<Int, Real, D> out{false, -1, -1, 0, 0, -1};
  int const n_external = max_n_conns * max_n_conns;
  if (code < n_external) {
    int const conn1 = code / max_n_conns;
    int const c = code % max_n_conns;
    int const partner =
        pose_stack_inter_block_connections[pose_ind][block_ind1][conn1][0];
    int const pconn =
        pose_stack_inter_block_connections[pose_ind][block_ind1][conn1][1];
    int const oconn =
        pose_stack_inter_block_connections[pose_ind][partner][c][1];
    int const key = conn_key[first_rot_block_type[pose_ind][partner]][pconn];
    if (key < 0) {
      return out;
    }
    out = {
        true,
        conn_metal[type1][conn1],
        conn_metal[type2][oconn],
        site_params[type1][conn1][key].d0,
        site_params[type2][oconn][key].d0,
        key};
  } else {
    int const conn1 = (code - n_external) / max_n_internal;
    int const k = (code - n_external) % max_n_internal;
    int const pconn =
        pose_stack_inter_block_connections[pose_ind][block_ind1][conn1][1];
    int const key = conn_key[type1][conn1];
    Int const metal = bridge_internal_metal[type1][conn1][k];
    if (key < 0 || metal < 0) {
      return out;
    }
    out = {
        true,
        metal,
        conn_metal[type2][pconn],
        bridge_internal_d0[type1][conn1][k],
        site_params[type2][pconn][key].d0,
        key};
  }
  return out;
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
void metal_coordination_pose_forward(
    ContextManager& mgr,
    TView<Vec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,

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
    bool output_block_pair_energies,
    bool compute_derivs,
    TView<Real, 4, D> V,
    TView<Vec<Real, 3>, 2, D> dV_dx) {
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);
  int const max_n_conns = conn_virt.size(1);
  int const max_n_fan = fan_atoms.size(1);
  int const max_n_internal = bridge_internal_metal.size(2);

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_connections.size(2) == max_n_conns);

  LAUNCH_BOX_32;

  auto eval_energies = ([=] TMOL_DEVICE_FUNC(int ind) {
    int const pose_ind = ind / max_n_blocks;
    int const block_ind1 = ind % max_n_blocks;
    int const rot_ind1 = first_rot_for_block[pose_ind][block_ind1];
    if (rot_ind1 < 0) {
      return;
    }
    int const block_type1 = first_rot_block_type[pose_ind][block_ind1];
    if (block_type1 < 0) {
      return;
    }
    Int const offset1 = rot_coord_offset[rot_ind1];
    int const Vind1 = output_block_pair_energies ? block_ind1 : 0;

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
          compute_derivs,
          Real(1),
          &V[0][pose_ind][Vind1][Vind1]);
    }

    for (int conn1 = 0; conn1 < max_n_conns; conn1++) {
      int const virt = conn_virt[block_type1][conn1];
      if (virt == NOT_A_SITE) {
        continue;
      }
      int const block_ind2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn1][0];
      if (block_ind2 < 0) {
        continue;
      }
      int const conn2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn1][1];
      int const rot_ind2 = first_rot_for_block[pose_ind][block_ind2];
      int const block_type2 = first_rot_block_type[pose_ind][block_ind2];
      if (rot_ind2 < 0 || block_type2 < 0) {
        continue;
      }
      int const key = conn_key[block_type2][conn2];
      if (key < 0) {
        continue;
      }
      // upper triangle, as the other pair terms report it
      int const lo = block_ind1 < block_ind2 ? block_ind1 : block_ind2;
      int const hi = block_ind1 < block_ind2 ? block_ind2 : block_ind1;
      int const Vlo = output_block_pair_energies ? lo : 0;
      int const Vhi = output_block_pair_energies ? hi : 0;
      accumulate_metal_site<Real, D>(
          rot_coords,
          offset1 + conn_metal[block_type1][conn1],
          virt >= 0 ? offset1 + virt : -1,
          rot_coord_offset[rot_ind2] + conn_atom[block_type2][conn2],
          site_params[block_type1][conn1][key],
          dV_dx,
          compute_derivs,
          Real(1),
          &V[0][pose_ind][Vlo][Vhi]);
    }

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
        [&](int,
            int other,
            int,
            Int metal1,
            Int metal2,
            Real d1,
            Real d2,
            int key) {
          int const lo = block_ind1 < other ? block_ind1 : other;
          int const hi = block_ind1 < other ? other : block_ind1;
          int const Vlo = output_block_pair_energies ? lo : 0;
          int const Vhi = output_block_pair_energies ? hi : 0;
          accumulate_metal_bridge<Real, D>(
              rot_coords,
              offset1 + metal1,
              rot_coord_offset[first_rot_for_block[pose_ind][other]] + metal2,
              d1,
              d2,
              bridge_params[key],
              dV_dx,
              compute_derivs,
              Real(1),
              &V[0][pose_ind][Vlo][Vhi]);
        });
  });

  DeviceDispatch<D>::template forall_grouped<launch_t>(
      mgr, n_poses, max_n_blocks, eval_energies);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
void metal_coordination_pose_backward(
    ContextManager& mgr,
    TView<Vec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,

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
    TView<Real, 4, D> dTdV,
    TView<Vec<Real, 3>, 2, D> dV_dx) {
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);
  int const max_n_conns = conn_virt.size(1);
  int const max_n_fan = fan_atoms.size(1);
  int const max_n_internal = bridge_internal_metal.size(2);

  assert(dTdV.size(0) == 1);
  assert(dTdV.size(1) == n_poses);
  assert(dTdV.size(2) == max_n_blocks);
  assert(dTdV.size(3) == max_n_blocks);

  LAUNCH_BOX_32;

  auto eval_derivs = ([=] TMOL_DEVICE_FUNC(int ind) {
    int const pose_ind = ind / max_n_blocks;
    int const block_ind1 = ind % max_n_blocks;
    int const rot_ind1 = first_rot_for_block[pose_ind][block_ind1];
    if (rot_ind1 < 0) {
      return;
    }
    int const block_type1 = first_rot_block_type[pose_ind][block_ind1];
    if (block_type1 < 0) {
      return;
    }
    Int const offset1 = rot_coord_offset[rot_ind1];

    Real const dTdV_fan = dTdV[0][pose_ind][block_ind1][block_ind1];
    if (dTdV_fan != 0) {
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
            true,
            dTdV_fan,
            nullptr);
      }
    }

    for (int conn1 = 0; conn1 < max_n_conns; conn1++) {
      int const virt = conn_virt[block_type1][conn1];
      if (virt == NOT_A_SITE) {
        continue;
      }
      int const block_ind2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn1][0];
      if (block_ind2 < 0) {
        continue;
      }
      int const lo = block_ind1 < block_ind2 ? block_ind1 : block_ind2;
      int const hi = block_ind1 < block_ind2 ? block_ind2 : block_ind1;
      Real const dTdV_site = dTdV[0][pose_ind][lo][hi];
      if (dTdV_site == 0) {
        continue;
      }
      int const conn2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn1][1];
      int const rot_ind2 = first_rot_for_block[pose_ind][block_ind2];
      int const block_type2 = first_rot_block_type[pose_ind][block_ind2];
      if (rot_ind2 < 0 || block_type2 < 0) {
        continue;
      }
      int const key = conn_key[block_type2][conn2];
      if (key < 0) {
        continue;
      }
      accumulate_metal_site<Real, D>(
          rot_coords,
          offset1 + conn_metal[block_type1][conn1],
          virt >= 0 ? offset1 + virt : -1,
          rot_coord_offset[rot_ind2] + conn_atom[block_type2][conn2],
          site_params[block_type1][conn1][key],
          dV_dx,
          true,
          dTdV_site,
          nullptr);
    }

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
        [&](int,
            int other,
            int,
            Int metal1,
            Int metal2,
            Real d1,
            Real d2,
            int key) {
          int const lo = block_ind1 < other ? block_ind1 : other;
          int const hi = block_ind1 < other ? other : block_ind1;
          Real const dTdV_bridge = dTdV[0][pose_ind][lo][hi];
          if (dTdV_bridge == 0) {
            return;
          }
          accumulate_metal_bridge<Real, D>(
              rot_coords,
              offset1 + metal1,
              rot_coord_offset[first_rot_for_block[pose_ind][other]] + metal2,
              d1,
              d2,
              bridge_params[key],
              dV_dx,
              true,
              dTdV_bridge,
              nullptr);
        });
  });

  DeviceDispatch<D>::template forall_grouped<launch_t>(
      mgr, n_poses, max_n_blocks, eval_derivs);
}

}  // namespace potentials
}  // namespace metal
}  // namespace score
}  // namespace tmol
