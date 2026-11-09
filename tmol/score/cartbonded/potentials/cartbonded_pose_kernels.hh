#pragma once

#include <tmol/utility/portable/reduce_ops.hh>
#include <tmol/utility/portable/matrix.hh>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/context_manager.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/connection.hh>
#include <tmol/score/common/counting.hh>
#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/hash_util.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/warp_segreduce.hh>
#include <tmol/score/common/warp_stride_reduce.hh>

// Whole-pose physics and coordinate derivatives. The caller supplies zeroed
// outputs; DeviceDispatch owns only the execution policy. No Torch dependency.

#include "potentials.hh"
#include "params_types.hh"

namespace tmol {
namespace score {
namespace cartbonded {
namespace potentials {

using tmol::portable::Vec;

// Replaces Eigen's `(v.array() != -1).count()`.
template <typename Int, int N>
TMOL_DEVICE_FUNC Int count_real_atoms(Vec<Int, N> const& v) {
  Int n = 0;
  for (int i = 0; i < N; ++i) {
    if (v[i] != -1) {
      ++n;
    }
  }
  return n;
}

// Replaces Eigen's runtime-sized comma-initializer
// `dst.head(na + nb) << a.tail(na), b.head(nb);`.
template <typename Int, int ND, int NA, int NB>
TMOL_DEVICE_FUNC void join_tail_head(
    Vec<Int, ND>& dst,
    Vec<Int, NA> const& a,
    Int na,
    Vec<Int, NB> const& b,
    Int nb) {
  for (Int i = 0; i < na; ++i) {
    dst[i] = a[NA - na + i];
  }
  for (Int i = 0; i < nb; ++i) {
    dst[na + i] = b[i];
  }
}
template <typename Real>
using CoordQuad = tmol::portable::Mat<Real, 4, 3>;

enum class subgraph_type { length, angle, torsion };

template <typename Int>
TMOL_DEVICE_FUNC subgraph_type get_subgraph_type(Vec<Int, 4> subgraph) {
  if (subgraph[2] == -1 && subgraph[3] == -1) return subgraph_type::length;
  if (subgraph[3] == -1) return subgraph_type::angle;
  return subgraph_type::torsion;
}

template <typename Int, Int size>
TMOL_DEVICE_FUNC Vec<Int, size> atom_local_to_global_indices(
    Vec<Int, size> local_indices, Int offset) {
  Vec<Int, size> global_indices;
  for (int i = 0; i < size; i++) {
    if (local_indices[i] != -1)
      global_indices[i] = local_indices[i] + offset;
    else
      global_indices[i] = -1;
  }
  return global_indices;
}

// Get the atom ID for an atom index from a lookup table, preserving (-1)s
template <typename Int, tmol::Device D>
TMOL_DEVICE_FUNC Int
get_atom_id(TensorAccessor<Int, 1, D> atom_id_table, Int atom_index) {
  return (atom_index == -1) ? -1 : atom_id_table[atom_index];
}

// From a Vec of atom indices and a lookup table, return a new Vec with their
// IDs preserving (-1)s
template <typename Int, Int size, tmol::Device D>
TMOL_DEVICE_FUNC Vec<Int, size> get_atom_ids(
    TensorAccessor<Int, 1, D> atom_id_table, Vec<Int, size> atoms) {
  Vec<Int, size> atom_ids;
  for (int i = 0; i < size; i++) {
    atom_ids[i] = get_atom_id(atom_id_table, atoms[i]);
  }
  return atom_ids;
}

// Reverse the non-(-1) elements of a subgraph
template <typename Int>
TMOL_DEVICE_FUNC void reverse_subgraph(Vec<Int, 4>& subgraph) {
  subgraph_type type = get_subgraph_type(subgraph);
  Int temp;
  switch (type) {
    case subgraph_type::length:
      temp = subgraph[0];
      subgraph[0] = subgraph[1];
      subgraph[1] = temp;
      break;
    case subgraph_type::angle:
      temp = subgraph[0];
      subgraph[0] = subgraph[2];
      subgraph[2] = temp;
      break;
    case subgraph_type::torsion:
      temp = subgraph[0];
      subgraph[0] = subgraph[3];
      subgraph[3] = temp;
      temp = subgraph[1];
      subgraph[1] = subgraph[2];
      subgraph[2] = temp;
      break;
  }
}

template <
    typename Real,
    typename Int,
    int N,
    tmol::Device D,
    typename GradReal = Real>
TMOL_DEVICE_FUNC void accumulate_result(
    Real& val,
    common::tuple<Real, Vec<Vec<Real, 3>, N>> to_add,
    Vec<Int, N> atoms,
    bool accumulate_derivs,
    TensorAccessor<Vec<GradReal, 3>, 1, D> dV,
    const Real& weight = 1.0) {
  val += common::get<0>(to_add);
  if (accumulate_derivs) {
    for (int i = 0; i < N; i++) {
      for (int axis = 0; axis < 3; ++axis)
        accumulate<D, GradReal>::add(
            dV[atoms[i]][axis],
            GradReal(common::get<1>(to_add)[i][axis] * weight));
    }
  }
}

// A connection-centred improper is not a bonded path. Share the ordered
// enumeration and parameter lookup across pose/rotamer forward/backward paths;
// only score accumulation differs between the callers.
template <typename Int, tmol::Device D, int nt, typename ScoreSubgraph>
TMOL_DEVICE_FUNC void evaluate_connection_impropers(
    int tid,
    int block_type1,
    int block_type2,
    int rot_coord_offset1,
    int rot_coord_offset2,
    int conn_ind1,
    int conn_ind2,
    TView<Vec<Int, 3>, 3, D> atom_paths_from_conn,
    TView<Int, 2, D> atom_is_rosetta,
    TView<Int, 2, D> atom_unique_ids,
    TView<Int, 2, D> atom_wildcard_ids,
    TView<Int, 2, D> atom_cross_ids,
    TView<Vec<Int, 5>, 1, D> hash_keys,
    ScoreSubgraph const& score_subgraph) {
  // the three substituents in every order, centre fixed at position 2,
  // matching the atm3-is-the-centre convention the database rows use
  int const ORDER[6][3] = {
      {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  for (int i = tid; i < 2 * 6; i += nt) {
    bool const reverse = i % 2 == 1;
    int const ordering = i / 2;
    int const block_typeA = reverse ? block_type2 : block_type1;
    int const block_typeB = reverse ? block_type1 : block_type2;
    int const rot_coord_offsetA =
        reverse ? rot_coord_offset2 : rot_coord_offset1;
    int const rot_coord_offsetB =
        reverse ? rot_coord_offset1 : rot_coord_offset2;
    int const conn_indA = reverse ? conn_ind2 : conn_ind1;
    int const conn_indB = reverse ? conn_ind1 : conn_ind2;

    Int const centre = atom_paths_from_conn[block_typeA][conn_indA][0][0];
    Int const partner = atom_paths_from_conn[block_typeB][conn_indB][0][0];
    if (centre == -1 || partner == -1) continue;

    // paths 1..3 are the connection atom plus one neighbour each
    Int nbr[2] = {-1, -1};
    int n_nbr = 0;
    for (int p = 1; p <= 3; ++p) {
      Int const cand = atom_paths_from_conn[block_typeA][conn_indA][p][1];
      if (cand == -1) continue;
      if (n_nbr < 2) nbr[n_nbr] = cand;
      ++n_nbr;
    }
    // a planarity centre has exactly three substituents, one across
    if (n_nbr != 2) continue;
    // a centre the Rosetta terms do not type belongs to the generic
    // term, which carries its own improper for it
    if (!atom_is_rosetta[block_typeA][centre]) continue;

    Int const item[3] = {nbr[0], nbr[1], partner};
    bool const across[3] = {false, false, true};
    int const* order = ORDER[ordering];

    Vec<Int, 4> local, atom_indices;
    bool is_b[4] = {false, false, false, false};
    for (int slot = 0; slot < 4; ++slot) {
      int const which = (slot == 2) ? -1 : order[slot < 2 ? slot : 2];
      local[slot] = (which == -1) ? centre : item[which];
      is_b[slot] = (which == -1) ? false : across[which];
      atom_indices[slot] =
          local[slot] + (is_b[slot] ? rot_coord_offsetB : rot_coord_offsetA);
    }

    int param_index = -1;
    for (int lookup_mode = 1; lookup_mode < 3; ++lookup_mode) {
      Vec<Int, 4> ids;
      for (int slot = 0; slot < 4; ++slot) {
        ids[slot] = is_b[slot]
                        ? atom_cross_ids[block_typeB][local[slot]]
                        : (lookup_mode == 2
                               ? atom_wildcard_ids[block_typeA][local[slot]]
                               : atom_unique_ids[block_typeA][local[slot]]);
      }
      param_index = hash_lookup<Int, 4, D>(ids, hash_keys);
      if (param_index != -1) break;
    }
    if (param_index != -1) {
      score_subgraph(atom_indices, param_index);
    }
  }
}

// Complete explicit connection records override only lengths and angles.
// Share dispatch/ownership and the legacy fallback across all four kernels.
template <typename Int, tmol::Device D, int nt, typename ScoreSubgraph>
TMOL_DEVICE_FUNC void evaluate_connection_paths(
    int tid,
    int block_type1,
    int block_type2,
    int rot_coord_offset1,
    int rot_coord_offset2,
    int conn_ind1,
    int conn_ind2,
    TView<Vec<Int, 3>, 3, D> atom_paths_from_conn,
    TView<Int, 1, D> block_type_is_fragment,
    TView<Int, 2, D> atom_is_rosetta,
    TView<Int, 2, D> atom_unique_ids,
    TView<Int, 2, D> atom_wildcard_ids,
    TView<Int, 2, D> atom_cross_ids,
    TView<Vec<Int, 5>, 1, D> hash_keys,
    TView<Vec<Int, 5>, 1, D> connection_hash_keys,
    TView<Vec<Int, 2>, 1, D> connection_spans,
    TView<Vec<Int, 5>, 1, D> connection_paths,
    ScoreSubgraph const& score_subgraph) {
  if (connection_spans.size(0) != 0) {
    Vec<Int, 4> key = {block_type1, conn_ind1, block_type2, conn_ind2};
    int const found = hash_lookup<Int, 4, D>(key, connection_hash_keys);
    if (found != -1) {
      int const start = connection_spans[found][0];
      int const signed_count = connection_spans[found][1];
      int const sign = signed_count < 0 ? -1 : 1;
      int const count = sign * signed_count;
      for (int i = tid; i < count; i += nt) {
        auto const path = connection_paths[start + i];
        Vec<Int, 4> atoms = {-1, -1, -1, -1};
        for (int j = 0; j < 4; ++j) {
          int const encoded = sign * path[j];
          if (encoded != 0) {
            atoms[j] = encoded > 0 ? rot_coord_offset1 + encoded - 1
                                   : rot_coord_offset2 - encoded - 1;
          }
        }
        score_subgraph(atoms, path[4]);
      }
      return;
    }
  }
  int n_connection_spanning_subgraphs = common::NUM_INTER_RES_PATHS;
  for (int i = tid; i < 2 * n_connection_spanning_subgraphs; i += nt) {
    bool reverse = i % 2 == 1;
    // interleave the subgraphs from the two directions so we can have
    // better warp coherence.
    int const subgraph_index = i / 2;
    int const block_typeA = reverse ? block_type2 : block_type1;
    int const block_typeB = reverse ? block_type1 : block_type2;
    int const rot_coord_offsetA =
        reverse ? rot_coord_offset2 : rot_coord_offset1;
    int const rot_coord_offsetB =
        reverse ? rot_coord_offset1 : rot_coord_offset2;
    int const conn_indA = reverse ? conn_ind2 : conn_ind1;
    int const conn_indB = reverse ? conn_ind1 : conn_ind2;

    int param_index = -1;
    Vec<Int, 4> subgraph_atom_indices = {-1, -1, -1, -1};

    tuple<int, int, int> spanning_subgraphs =
        get_connection_spanning_subgraph_indices(subgraph_index);
    int resA_path_ind = common::get<1>(spanning_subgraphs);
    int resB_path_ind = common::get<2>(spanning_subgraphs);
    // Grab the paths from each block
    Vec<Int, 3> resA_path =
        atom_paths_from_conn[block_typeA][conn_indA][resA_path_ind];
    Vec<Int, 3> resB_path =
        atom_paths_from_conn[block_typeB][conn_indB][resB_path_ind];

    // Make sure these are valid paths
    if (resA_path[0] == -1 || resB_path[0] == -1) continue;
    // Reverse the first path so that we can join them head-to-head
    resA_path.reverseInPlace();
    // Get a new Vec containing the global indices of the atoms
    Vec<Int, 3> resA_atom_indices =
        atom_local_to_global_indices(resA_path, rot_coord_offsetA);
    Vec<Int, 3> resB_atom_indices =
        atom_local_to_global_indices(resB_path, rot_coord_offsetB);
    // Calculate the size of each path
    Int resA_size = count_real_atoms(resA_atom_indices);
    Int resB_size = count_real_atoms(resB_atom_indices);

    if (resA_size + resB_size == 4) {
      // genbonded scores a torsion unless both central atoms are Rosetta-typed
      bool rosetta_owned = true;
      for (int pos = 1; pos <= 2; ++pos) {
        bool const in_a = pos < resA_size;
        Int const local =
            in_a ? resA_path[3 - resA_size + pos] : resB_path[pos - resA_size];
        rosetta_owned &=
            atom_is_rosetta[in_a ? block_typeA : block_typeB][local] != 0;
      }
      if (!rosetta_owned) continue;
    }

    // Prefer exact parameters when both blocks share a residue base name,
    // then fall back to the historical unique/wildcard and
    // wildcard/wildcard lookups used by polymer connections.
    int const first_lookup_mode = block_type_is_fragment[block_typeA]
                                          && block_type_is_fragment[block_typeB]
                                      ? 0
                                      : 1;
    for (int lookup_mode = first_lookup_mode; lookup_mode < 3; ++lookup_mode) {
      const auto& resA_atom_id_table = (lookup_mode == 2)
                                           ? atom_wildcard_ids[block_typeA]
                                           : atom_unique_ids[block_typeA];
      const auto& resB_atom_id_table = (lookup_mode == 0)
                                           ? atom_unique_ids[block_typeB]
                                           : atom_cross_ids[block_typeB];

      // Get the atom IDs
      Vec<Int, 3> resA_subgraph_atom_ids =
          get_atom_ids(resA_atom_id_table, resA_path);
      Vec<Int, 3> resB_subgraph_atom_ids =
          get_atom_ids(resB_atom_id_table, resB_path);

      // Make the joined data structures
      Vec<Int, 4> path;
      Vec<Int, 4> atom_indices;
      Vec<Int, 4> subgraph_atom_ids;

      // Init with -1s
      path = Vec<Int, 4>::Constant(-1);
      atom_indices = Vec<Int, 4>::Constant(-1);
      subgraph_atom_ids = Vec<Int, 4>::Constant(-1);

      // Join the paths into 1
      join_tail_head(path, resA_path, resA_size, resB_path, resB_size);
      join_tail_head(
          atom_indices,
          resA_atom_indices,
          resA_size,
          resB_atom_indices,
          resB_size);
      join_tail_head(
          subgraph_atom_ids,
          resA_subgraph_atom_ids,
          resA_size,
          resB_subgraph_atom_ids,
          resB_size);

      // Do the lookup
      param_index = hash_lookup<Int, 4, D>(subgraph_atom_ids, hash_keys);
      if (param_index != -1) {
        // we found it!
        subgraph_atom_indices = atom_indices;
        break;
      }
    }
    if (param_index != -1) {
      score_subgraph(subgraph_atom_indices, param_index);
    }
  }
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    typename GradReal = Real,
    typename EnergyReal = Real>
void cartbonded_pose_forward(
    ContextManager& mgr,

    TView<Vec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,

    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Vec<Int, 3>, 3, D> atom_paths_from_conn,
    TView<Int, 2, D> atom_unique_ids,
    TView<Int, 2, D> atom_wildcard_ids,
    TView<Int, 2, D> atom_is_rosetta,
    TView<Int, 1, D> block_type_is_fragment,
    TView<Int, 2, D> atom_cross_ids,
    TView<Vec<Int, 5>, 1, D> connection_hash_keys,
    TView<Vec<Int, 2>, 1, D> connection_spans,
    TView<Vec<Int, 5>, 1, D> connection_paths,
    TView<Vec<Int, 5>, 1, D> hash_keys,
    TView<Vec<Real, 7>, 1, D> hash_values,
    TView<Vec<Int, 4>, 1, D> cart_subgraphs,

    TView<Int, 1, D> cart_subgraph_offsets,

    TView<Vec<Int, 3>, 1, D> cart_subgraph_type_counts,

    TView<Vec<Int, 3>, 1, D> cart_subgraph_type_offsets,
    TView<Int, 1, D> cart_subgraph_param_indices,

    bool output_block_pair_energies,

    bool compute_derivs,
    TView<EnergyReal, 4, D> V,
    TView<Vec<GradReal, 3>, 2, D> dV_dx) {
  using tmol::score::common::get_connection_spanning_subgraphs_offset;
  using tmol::score::common::get_n_connection_spanning_subgraphs;

  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const max_n_conns = pose_stack_inter_block_connections.size(2);
  int const workgroups_per_pose = score::common::checked_dispatch_product(
      max_n_blocks,
      int64_t(max_n_conns) + 1,
      "Cartesian bonded per-pose dispatch");
  int const n_block_types = cart_subgraph_offsets.size(0);
  int const n_subgraphs = cart_subgraphs.size(0);
  int const n_max_atoms_per_block = atom_unique_ids.size(1);

  assert(first_rot_block_type.size(0) == n_poses);
  assert(first_rot_block_type.size(1) == max_n_blocks);

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_connections.size(2) == max_n_conns);

  assert(atom_paths_from_conn.size(0) == n_block_types);
  assert(atom_paths_from_conn.size(1) == max_n_conns);
  assert(atom_paths_from_conn.size(2) == MAX_PATHS_FROM_CONN);

  assert(atom_unique_ids.size(0) == n_block_types);
  assert(atom_unique_ids.size(1) == n_max_atoms_per_block);

  assert(atom_wildcard_ids.size(0) == n_block_types);
  assert(atom_cross_ids.size(0) == n_block_types);
  assert(atom_wildcard_ids.size(1) == n_max_atoms_per_block);
  assert(atom_cross_ids.size(1) == n_max_atoms_per_block);

  assert(cart_subgraph_offsets.size(0) == n_block_types);
  assert(cart_subgraph_type_counts.size(0) == n_block_types);
  assert(cart_subgraph_type_offsets.size(0) == n_block_types);
  assert(cart_subgraph_param_indices.size(0) == n_subgraphs);

  // Algorithm: launch n_rots * (max_n_conns + 1) CTAs, each looking
  // at a single residue's (rotamer's) connection (+ one "connection" to self),
  // killing CTAs that connect to lower residues to avoid double counting.
  // Each CTA iterates across all the subgraphs for that connection,
  // then performs a reduction so that thread-0 can write the results to global
  // memory.

  // This interaction kernel benefits from a 64-register budget on modern GPUs,
  // which leaves enough resident warps to hide its dependency latency.
  LAUNCH_BOX_32_OCC(32);
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;

  // Convention: conn == max_n_conns ==> the intra-rotamer set of cart energies

  bool const accumulate_derivs = compute_derivs && !output_block_pair_energies;

  auto eval_subgraphs_for_interaction = ([=] TMOL_DEVICE_FUNC(int cta) {
    // Only one element of this union: the shared memory array for
    // the reduction, which itself only will take any space if NT > 32
    // as the reduction otherwise uses only warp shuffle operations!
    SHARED_MEMORY union shared_mem_union {
      shared_mem_union() {}
      int stub;  // can't have an empty union, so, we'll have an integer
                 // placeholder.
      CTA_REAL_REDUCE_T_VARIABLE;
    } shared;

    int const pose_ind = cta / workgroups_per_pose;
    int const block_conn = cta % workgroups_per_pose;
    int const block_ind1 = block_conn / (max_n_conns + 1);
    int const conn_ind1 = block_conn % (max_n_conns + 1);
    int const block_type1 = first_rot_block_type[pose_ind][block_ind1];

    if (block_type1 == -1) {
      return;
    }

    int const rot_ind1 = first_rot_for_block[pose_ind][block_ind1];
    int const rot_coord_offset1 = rot_coord_offset[rot_ind1];

    Real length_score = 0.0;
    Real angle_score = 0.0;
    Real torsion_score = 0.0;
    Real improper_torsion_score = 0.0;
    Real hxyl_torsion_score = 0.0;

    auto score_subgraph = ([&] TMOL_DEVICE_FUNC(
                               Vec<Int, 4> atoms, Int param_index) {
      Vec<Real, 7> params = hash_values[param_index];

      Vec<Real, 3> atom1 = rot_coords[atoms[0]];
      Vec<Real, 3> atom2 = rot_coords[atoms[1]];

      subgraph_type type = get_subgraph_type(atoms);

      int score_type = params[0];

      // Real score;
      switch (type) {
        case subgraph_type::length: {
          if (accumulate_derivs) {
            auto eval = cblength_V_dV(atom1, atom2, params[2], params[1]);
            accumulate_result<Real, Int, 2, D>(
                length_score,
                eval,
                atoms.template head<2>(),
                true,
                dV_dx[score_type],
                1.0);
          } else {
            length_score += cblength_V(atom1, atom2, params[2], params[1]);
          }

          break;
        }
        case subgraph_type::angle: {
          Vec<Real, 3> atom3 = rot_coords[atoms[2]];
          if (accumulate_derivs) {
            auto eval = cbangle_V_dV(atom1, atom2, atom3, params[2], params[1]);
            accumulate_result<Real, Int, 3, D>(
                angle_score,
                eval,
                atoms.template head<3>(),
                true,
                dV_dx[score_type],
                1.0);
          } else {
            angle_score += cbangle_V(atom1, atom2, atom3, params[2], params[1]);
          }

          break;
        }
        case subgraph_type::torsion: {
          Vec<Real, 3> atom3 = rot_coords[atoms[2]];
          Vec<Real, 3> atom4 = rot_coords[atoms[3]];
          Real& tor_score = (score_type == 3)   ? improper_torsion_score
                            : (score_type == 4) ? hxyl_torsion_score
                                                : torsion_score;
          if (accumulate_derivs) {
            auto eval = cbtorsion_V_dV(
                atom1,
                atom2,
                atom3,
                atom4,
                params[1],
                params[2],
                params[3],
                params[4],
                params[5],
                params[6]);
            accumulate_result<Real, Int, 4, D>(
                tor_score,
                eval,
                atoms.template head<4>(),
                true,
                dV_dx[score_type],
                1.0);
          } else {
            tor_score += cbtorsion_V(
                atom1,
                atom2,
                atom3,
                atom4,
                params[1],
                params[2],
                params[3],
                params[4],
                params[5],
                params[6]);
          }

          break;
        }
      }
    });

    int block_ind2 = -1;
    if (conn_ind1 == max_n_conns) {
      // intra-residue!
      block_ind2 = block_ind1;
      // Threads in the CTA iterate across the subgraphs within the block
      int subgraph_offset = cart_subgraph_offsets[block_type1];
      int subgraph_offset_next = block_type1 + 1 == n_block_types
                                     ? n_subgraphs
                                     : cart_subgraph_offsets[block_type1 + 1];
      int n_subgraphs = subgraph_offset_next - subgraph_offset;
      auto eval_intra_res_subgraphs = ([&] TMOL_DEVICE_FUNC(int tid) {
        // printf(" Intra res interaction for rot %d (bt %d); cta %d tid %d \n",
        // rot_ind1, block_type1, cta, tid);

        for (int i = tid; i < n_subgraphs; i += nt) {
          int const subgraph_index = subgraph_offset + i;
          int const param_index = cart_subgraph_param_indices[subgraph_index];
          if (param_index != -1) {
            Vec<Int, 4> subgraph_atom_indices = atom_local_to_global_indices(
                cart_subgraphs[subgraph_index], rot_coord_offset1);
            score_subgraph(subgraph_atom_indices, param_index);
          }
        }
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          eval_intra_res_subgraphs);
    } else {
      // Inter residue!
      block_ind2 = pose_stack_inter_block_connections[pose_ind][block_ind1]
                                                     [conn_ind1][0];
      int const conn_ind2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                            [1];
      if (block_ind2 == -1) {
        return;
      }
      if (block_ind1 > block_ind2) {
        // to avoid double counting, only have the lower-indexed block
        // handle the interaction
        return;
      }
      int const rot_ind2 = first_rot_for_block[pose_ind][block_ind2];
      int const block_type2 = first_rot_block_type[pose_ind][block_ind2];
      int const rot_coord_offset2 = rot_coord_offset[rot_ind2];

      // Use capture-by-reference here so that we can write to the length_score,
      // angle_score, and torsion_score variables
      auto eval_inter_res_subgraphs = ([&] TMOL_DEVICE_FUNC(int tid) {
        evaluate_connection_paths<Int, D, nt>(
            tid,
            block_type1,
            block_type2,
            rot_coord_offset1,
            rot_coord_offset2,
            conn_ind1,
            conn_ind2,
            atom_paths_from_conn,
            block_type_is_fragment,
            atom_is_rosetta,
            atom_unique_ids,
            atom_wildcard_ids,
            atom_cross_ids,
            hash_keys,
            connection_hash_keys,
            connection_spans,
            connection_paths,
            score_subgraph);
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          eval_inter_res_subgraphs);

      auto eval_inter_res_impropers = ([&] TMOL_DEVICE_FUNC(int tid) {
        evaluate_connection_impropers<Int, D, nt>(
            tid,
            block_type1,
            block_type2,
            rot_coord_offset1,
            rot_coord_offset2,
            conn_ind1,
            conn_ind2,
            atom_paths_from_conn,
            atom_is_rosetta,
            atom_unique_ids,
            atom_wildcard_ids,
            atom_cross_ids,
            hash_keys,
            score_subgraph);
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          eval_inter_res_impropers);
    }

    // Now let's accumulate the scores into main memory
    auto reduce_energies = ([&] TMOL_DEVICE_FUNC(int tid) {
      Real const cta_length_score =
          DeviceDispatch<D>::template reduce_in_workgroup<nt>(
              length_score, shared, tmol::portable::plus_t<Real>());
      Real const cta_angle_score =
          DeviceDispatch<D>::template reduce_in_workgroup<nt>(
              angle_score, shared, tmol::portable::plus_t<Real>());
      Real const cta_torsion_score =
          DeviceDispatch<D>::template reduce_in_workgroup<nt>(
              torsion_score, shared, tmol::portable::plus_t<Real>());
      Real const cta_improper_torsion_score =
          DeviceDispatch<D>::template reduce_in_workgroup<nt>(
              improper_torsion_score, shared, tmol::portable::plus_t<Real>());
      Real const cta_hxyl_torsion_score =
          DeviceDispatch<D>::template reduce_in_workgroup<nt>(
              hxyl_torsion_score, shared, tmol::portable::plus_t<Real>());

      if (tid == 0) {
        // straight assignment in the block-pair scoring case; atomic add
        // otherwise;
        if (output_block_pair_energies) {
          // We cannot just perform assignment here because it is possible that
          // a residue may form more than one bond to another residue,
          // e.g. i to i+1 disulfide
          if (cta_length_score != 0.0) {
            accumulate<D, EnergyReal>::add(
                V[0][pose_ind][block_ind1][block_ind2], cta_length_score);
          }
          if (cta_angle_score != 0.0) {
            accumulate<D, EnergyReal>::add(
                V[1][pose_ind][block_ind1][block_ind2], cta_angle_score);
          }
          if (cta_torsion_score != 0.0) {
            accumulate<D, EnergyReal>::add(
                V[2][pose_ind][block_ind1][block_ind2], cta_torsion_score);
          }
          if (cta_improper_torsion_score != 0.0) {
            accumulate<D, EnergyReal>::add(
                V[3][pose_ind][block_ind1][block_ind2],
                cta_improper_torsion_score);
          }
          if (cta_hxyl_torsion_score != 0.0) {
            accumulate<D, EnergyReal>::add(
                V[4][pose_ind][block_ind1][block_ind2], cta_hxyl_torsion_score);
          }
        } else {
          if (cta_length_score != 0.0) {
            accumulate<D, EnergyReal>::add(
                V[0][pose_ind][0][0], cta_length_score);
          }
          if (cta_angle_score != 0.0) {
            accumulate<D, EnergyReal>::add(
                V[1][pose_ind][0][0], cta_angle_score);
          }
          if (cta_torsion_score != 0.0) {
            accumulate<D, EnergyReal>::add(
                V[2][pose_ind][0][0], cta_torsion_score);
          }
          if (cta_improper_torsion_score != 0.0) {
            accumulate<D, EnergyReal>::add(
                V[3][pose_ind][0][0], cta_improper_torsion_score);
          }
          if (cta_hxyl_torsion_score != 0.0) {
            accumulate<D, EnergyReal>::add(
                V[4][pose_ind][0][0], cta_hxyl_torsion_score);
          }
        }
      }
    });
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(reduce_energies);
  });
  DeviceDispatch<D>::template foreach_pose_workgroup<launch_t>(
      mgr, n_poses, workgroups_per_pose, eval_subgraphs_for_interaction);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    typename GradReal = Real>
void cartbonded_pose_backward(
    ContextManager& mgr,

    TView<Vec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,

    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Vec<Int, 3>, 3, D> atom_paths_from_conn,
    TView<Int, 2, D> atom_unique_ids,
    TView<Int, 2, D> atom_wildcard_ids,
    TView<Int, 2, D> atom_is_rosetta,
    TView<Int, 1, D> block_type_is_fragment,
    TView<Int, 2, D> atom_cross_ids,
    TView<Vec<Int, 5>, 1, D> connection_hash_keys,
    TView<Vec<Int, 2>, 1, D> connection_spans,
    TView<Vec<Int, 5>, 1, D> connection_paths,
    TView<Vec<Int, 5>, 1, D> hash_keys,
    TView<Vec<Real, 7>, 1, D> hash_values,
    TView<Vec<Int, 4>, 1, D> cart_subgraphs,

    TView<Int, 1, D> cart_subgraph_offsets,

    TView<Vec<Int, 3>, 1, D> cart_subgraph_type_counts,

    TView<Vec<Int, 3>, 1, D> cart_subgraph_type_offsets,
    TView<Int, 1, D> cart_subgraph_param_indices,

    TView<Real, 4, D> dTdV,
    TView<Vec<GradReal, 3>, 2, D> dV_dx) {
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const max_n_conns = pose_stack_inter_block_connections.size(2);
  int const workgroups_per_pose = score::common::checked_dispatch_product(
      max_n_blocks,
      int64_t(max_n_conns) + 1,
      "Cartesian bonded derivative per-pose dispatch");
  int const n_block_types = cart_subgraph_offsets.size(0);
  int const n_subgraphs = cart_subgraphs.size(0);
  int const n_max_atoms_per_block = atom_unique_ids.size(1);

  assert(first_rot_block_type.size(0) == n_poses);
  assert(first_rot_block_type.size(1) == max_n_blocks);

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_connections.size(2) == max_n_conns);

  assert(atom_paths_from_conn.size(0) == n_block_types);
  assert(atom_paths_from_conn.size(1) == max_n_conns);
  assert(atom_paths_from_conn.size(2) == MAX_PATHS_FROM_CONN);

  assert(atom_unique_ids.size(0) == n_block_types);
  assert(atom_unique_ids.size(1) == n_max_atoms_per_block);

  assert(atom_wildcard_ids.size(0) == n_block_types);
  assert(atom_cross_ids.size(0) == n_block_types);
  assert(atom_wildcard_ids.size(1) == n_max_atoms_per_block);
  assert(atom_cross_ids.size(1) == n_max_atoms_per_block);

  assert(cart_subgraph_offsets.size(0) == n_block_types);
  assert(cart_subgraph_type_counts.size(0) == n_block_types);
  assert(cart_subgraph_type_offsets.size(0) == n_block_types);

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;

  auto eval_subgraphs_for_interaction = ([=] TMOL_DEVICE_FUNC(int cta) {
    int const pose_ind = cta / workgroups_per_pose;
    int const block_conn = cta % workgroups_per_pose;
    int const block_ind1 = block_conn / (max_n_conns + 1);
    int const conn_ind1 = block_conn % (max_n_conns + 1);
    int const block_type1 = first_rot_block_type[pose_ind][block_ind1];

    if (block_type1 == -1) {
      return;
    }

    int const rot_ind1 = first_rot_for_block[pose_ind][block_ind1];
    int const rot_coord_offset1 = rot_coord_offset[rot_ind1];

    auto score_subgraph = ([=] TMOL_DEVICE_FUNC(
                               Vec<Int, 4> atoms,
                               Int param_index,
                               int pose_ind,
                               int block_ind1,
                               int block_ind2) {
      Real length_score = 0.0;
      Real angle_score = 0.0;
      Real torsion_score = 0.0;
      Real improper_torsion_score = 0.0;
      Real hxyl_torsion_score = 0.0;
      Vec<Real, 7> params = hash_values[param_index];

      subgraph_type type = get_subgraph_type(atoms);

      int score_type = params[0];
      Real block_weight = dTdV[score_type][pose_ind][block_ind1][block_ind2];
      if (block_weight == 0) {
        return;
      }

      Vec<Real, 3> atom1 = rot_coords[atoms[0]];
      Vec<Real, 3> atom2 = rot_coords[atoms[1]];

      // Real score;
      switch (type) {
        case subgraph_type::length: {
          auto eval = cblength_V_dV(atom1, atom2, params[2], params[1]);
          accumulate_result<Real, Int, 2, D>(
              length_score,
              eval,
              atoms.template head<2>(),
              true,
              dV_dx[score_type],
              block_weight);

          break;
        }
        case subgraph_type::angle: {
          Vec<Real, 3> atom3 = rot_coords[atoms[2]];
          auto eval = cbangle_V_dV(atom1, atom2, atom3, params[2], params[1]);
          accumulate_result<Real, Int, 3, D>(
              angle_score,
              eval,
              atoms.template head<3>(),
              true,
              dV_dx[score_type],
              block_weight);

          break;
        }
        case subgraph_type::torsion: {
          Vec<Real, 3> atom3 = rot_coords[atoms[2]];
          Vec<Real, 3> atom4 = rot_coords[atoms[3]];
          auto eval = cbtorsion_V_dV(
              atom1,
              atom2,
              atom3,
              atom4,
              params[1],
              params[2],
              params[3],
              params[4],
              params[5],
              params[6]);
          Real& tor_score = (score_type == 3)   ? improper_torsion_score
                            : (score_type == 4) ? hxyl_torsion_score
                                                : torsion_score;
          accumulate_result<Real, Int, 4, D>(
              tor_score,
              eval,
              atoms.template head<4>(),
              true,
              dV_dx[score_type],
              block_weight);

          break;
        }
      }
    });

    if (conn_ind1 == max_n_conns) {
      // intra-residue!
      int subgraph_offset = cart_subgraph_offsets[block_type1];
      int subgraph_offset_next = block_type1 + 1 == n_block_types
                                     ? n_subgraphs
                                     : cart_subgraph_offsets[block_type1 + 1];
      int n_subgraphs = subgraph_offset_next - subgraph_offset;
      auto eval_intra_res_subgraphs = ([&] TMOL_DEVICE_FUNC(int tid) {
        for (int i = tid; i < n_subgraphs; i += nt) {
          int const subgraph_index = subgraph_offset + i;
          int const param_index = cart_subgraph_param_indices[subgraph_index];
          if (param_index != -1) {
            Vec<Int, 4> subgraph_atom_indices = atom_local_to_global_indices(
                cart_subgraphs[subgraph_index], rot_coord_offset1);
            score_subgraph(
                subgraph_atom_indices,
                param_index,
                pose_ind,
                block_ind1,
                block_ind1);
          }
        }
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          eval_intra_res_subgraphs);
    } else {
      // Inter residue!
      int const block_ind2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                            [0];
      int const conn_ind2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                            [1];
      if (block_ind2 == -1) {
        return;
      }
      if (block_ind1 > block_ind2) {
        // to avoid double counting, only have the lower-indexed block
        // handle the interaction
        return;
      }
      int const rot_ind2 = first_rot_for_block[pose_ind][block_ind2];
      int const block_type2 = first_rot_block_type[pose_ind][block_ind2];
      int const rot_coord_offset2 = rot_coord_offset[rot_ind2];

      // Use by-reference capture here so that we can write to the length,
      // angle, and torsion_score variables
      auto score_connection = ([&] TMOL_DEVICE_FUNC(
                                   Vec<Int, 4> atoms, int param_index) {
        score_subgraph(atoms, param_index, pose_ind, block_ind1, block_ind2);
      });
      auto eval_inter_res_subgraphs = ([&] TMOL_DEVICE_FUNC(int tid) {
        evaluate_connection_paths<Int, D, nt>(
            tid,
            block_type1,
            block_type2,
            rot_coord_offset1,
            rot_coord_offset2,
            conn_ind1,
            conn_ind2,
            atom_paths_from_conn,
            block_type_is_fragment,
            atom_is_rosetta,
            atom_unique_ids,
            atom_wildcard_ids,
            atom_cross_ids,
            hash_keys,
            connection_hash_keys,
            connection_spans,
            connection_paths,
            score_connection);
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          eval_inter_res_subgraphs);

      auto eval_inter_res_impropers = ([&] TMOL_DEVICE_FUNC(int tid) {
        evaluate_connection_impropers<Int, D, nt>(
            tid,
            block_type1,
            block_type2,
            rot_coord_offset1,
            rot_coord_offset2,
            conn_ind1,
            conn_ind2,
            atom_paths_from_conn,
            atom_is_rosetta,
            atom_unique_ids,
            atom_wildcard_ids,
            atom_cross_ids,
            hash_keys,
            score_connection);
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          eval_inter_res_impropers);
    }

    // We can skip the step where we accumulate into global memory because
    // we are only computing derivatives in this pass and don't need the score
    // a second time
  });
  DeviceDispatch<D>::template foreach_pose_workgroup<launch_t>(
      mgr, n_poses, workgroups_per_pose, eval_subgraphs_for_interaction);

}  // backward

}  // namespace potentials
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
