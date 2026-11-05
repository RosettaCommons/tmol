#pragma once

// LJ/LK kernel bodies.
//
// Separated from ljlk_pose_score.impl.hh -- which holds the dispatch entry
// points and therefore reaches the torch tensor layer -- so that a Metal
// translation unit can include just the kernels. Everything named here is
// freestanding: the portable math layer, the view-generic helpers in ljlk.hh,
// and the device-templated workgroup machinery.
//
// Note the absence of device_operations.hh. The bodies reach the workgroup
// primitives through their DeviceDispatch template parameter, which is only
// resolved at instantiation, so the interface declaration -- which returns
// TPack and therefore needs torch -- is the caller's concern, not theirs.
//
// The bodies are templated on their context and shared-memory types, so the
// CPU and CUDA builds pass TView-backed state while a `.metal` entry passes
// MetalView-backed state and runs the same code.

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/tile_atom_pair_evaluation.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/upper_triangle_indices.hh>
#include <tmol/score/common/workgroup_lane.hh>
#include <tmol/score/ljlk/potentials/ljlk.hh>
#include <tmol/score/ljlk/potentials/ljlk_ctx.hh>
#include <tmol/utility/portable/array.hh>
#include <tmol/utility/portable/device.hh>
#include <tmol/utility/portable/metal_view.hh>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

enum { ljlk_tile_size = 32, ljlk_max_n_conn = 12 };

using tmol::portable::Vec;
using tmol::score::common::WorkgroupLane;

// Shared-memory block for one LJ/LK workgroup.
//
// Hoisted to namespace scope so a kernel body can take it by reference rather
// than declare it inline. Metal permits `threadgroup` declarations only at
// kernel-entry scope, so the entry declares the storage and passes it down;
// CUDA and CPU are unaffected, since the caller declares it either way.
template <typename Real, int TILE, int MAX_CONN, typename launch_t>
union LJLKSharedMem {
  CTA_REAL_REDUCE_T_TYPEDEF;

#if !TMOL_PORTABLE_METAL
  LJLKSharedMem() {}
#endif
  LJLKBlockPairSharedData<Real, TILE, MAX_CONN> m;
  CTA_REAL_REDUCE_T_VARIABLE;
};

#define SCORE_INTER_LJ_ATOM_PAIR(atom_pair_func)                        \
  TMOL_DEVICE_FUNC(                                                     \
      int start_atom1,                                                  \
      int start_atom2,                                                  \
      int atom_tile_ind1,                                               \
      int atom_tile_ind2,                                               \
      TMOL_THREAD LJLKScoringData<Real> const& inter_dat) {             \
    int separation = interres_count_pair_separation<ljlk_tile_size>(    \
        inter_dat,                                                      \
        atom_tile_ind1,                                                 \
        atom_tile_ind2,                                                 \
        ctx.block_type_all_atoms_ligand_typed[inter_dat.r1.block_type]  \
            && ctx.block_type_all_atoms_ligand_typed[inter_dat.r2       \
                                                         .block_type]); \
    return atom_pair_func(                                              \
        atom_tile_ind1,                                                 \
        atom_tile_ind2,                                                 \
        start_atom1,                                                    \
        start_atom2,                                                    \
        inter_dat,                                                      \
        separation);                                                    \
  }

// SCORE_INTRA_LJ_ATOM_PAIR
// input argument:  a function with signature (
//     int atom_tile_idx1
//     int atom_tile_idx2
//     int start_atom1
//     int start_atom2
//     LJLKScoringData<Real> const &score_dat
//     int cp_separation)
//   ->tmol::portable::array<Real, 2>
// captures:
//    ctx.block_type_path_distance
#define SCORE_INTRA_LJ_ATOM_PAIR(atom_pair_func)                         \
  TMOL_DEVICE_FUNC(                                                      \
      int start_atom1,                                                   \
      int start_atom2,                                                   \
      int atom_tile_ind1,                                                \
      int atom_tile_ind2,                                                \
      TMOL_THREAD LJLKScoringData<Real> const& intra_dat) {              \
    int const atom_ind1 = start_atom1 + atom_tile_ind1;                  \
    int const atom_ind2 = start_atom2 + atom_tile_ind2;                  \
    int const separation =                                               \
        ctx.block_type_path_distance[intra_dat.r1.block_type][atom_ind1] \
                                    [atom_ind2];                         \
    return atom_pair_func(                                               \
        atom_tile_ind1,                                                  \
        atom_tile_ind2,                                                  \
        start_atom1,                                                     \
        start_atom2,                                                     \
        intra_dat,                                                       \
        separation);                                                     \
  }

// captures:
//    coords (TView<Vec<Real, 3>, 2, D>)
//    ctx.block_type_atom_types (TView<Int, 2, D>)
//    ctx.type_params (TView<LJLKTypeParams<Real>, 1, D>)
#define LOAD_BLOCK_COORDS_AND_PARAMS_INTO_SHARED   \
  TMOL_DEVICE_FUNC(                                \
      int pose_ind,                                \
      TMOL_THREAD LJLKSingleResData<Real>& r_dat,  \
      int n_atoms_to_load,                         \
      int start_atom) {                            \
    ljlk_load_block_coords_and_params_into_shared< \
        DeviceDispatch,                            \
        D,                                         \
        nt,                                        \
        Real,                                      \
        Int>(                                      \
        ctx.rot_coords,                            \
        ctx.block_type_atom_types,                 \
        ctx.type_params,                           \
        pose_ind,                                  \
        r_dat,                                     \
        n_atoms_to_load,                           \
        start_atom,                                \
        lane);                                     \
  }

// LOAD_BLOCK_INTO_SHARED
// captures:
//    coords (TView<Vec<Real, 3>, 2, D>)
//    ctx.block_type_atom_types (TView<Int, 2, D>)
//    ctx.type_params (TView<LJLKTypeParams<Real>, 1, D>)
//    ctx.block_type_path_distance (TView<Int, 3, D>)
#define LOAD_BLOCK_INTO_SHARED                    \
  TMOL_DEVICE_FUNC(                               \
      int pose_ind,                               \
      TMOL_THREAD LJLKSingleResData<Real>& r_dat, \
      int n_atoms_to_load,                        \
      int start_atom,                             \
      bool count_pair_striking_dist,              \
      TMOL_SHARED_PTR(int) conn_ats) {  \
    ljlk_load_block_into_shared<                  \
        DeviceDispatch,                           \
        D,                                        \
        nt,                                       \
        ljlk_tile_size,                           \
        Real,                                     \
        Int>(                                     \
        ctx.rot_coords,                           \
        ctx.block_type_atom_types,                \
        ctx.type_params,                          \
        ctx.block_type_path_distance,             \
        pose_ind,                                 \
        r_dat,                                    \
        n_atoms_to_load,                          \
        start_atom,                               \
        count_pair_striking_dist,                 \
        conn_ats,                                 \
        lane);                                    \
  }

// LOAD_TILE_INVARIANT_INTERRES_DATA
// captures:
//    pose_stack_block_coord_offset (TView<Vec<Real, 3>, 2, D>)
//    ctx.pose_stack_near_blocks (TView<Int, 4, D>)
//    ctx.block_type_n_interblock_bonds (TView<Int, 1, D>)
//    ctx.block_type_atoms_forming_chemical_bonds (TView<Int, 2, D>)
//    ctx.pose_stack_inter_block_bondsep (TView<int8_t, 5, D>)
//    ctx.global_params (TView<LJGlobalParams<Real>, 1, D>)
//    max_important_bond_separation (int)
#define LOAD_TILE_INVARIANT_INTERRES_DATA                                     \
  TMOL_DEVICE_FUNC(                                                           \
      int pose_ind,                                                           \
      int rot_ind1,                                                           \
      int rot_ind2,                                                           \
      int block_ind1,                                                         \
      int block_ind2,                                                         \
      int block_type1,                                                        \
      int block_type2,                                                        \
      int n_atoms1,                                                           \
      int n_atoms2,                                                           \
      TMOL_THREAD LJLKScoringData<Real>& inter_dat,                           \
      TMOL_SHARED_REF(shared_mem_union) shared) {                             \
    ljlk_load_tile_invariant_interres_data<DeviceDispatch, D, nt, Int, Real>( \
        ctx.rot_coord_offset,                                                 \
        ctx.pose_stack_near_blocks,                                           \
        ctx.block_type_n_interblock_bonds,                                    \
        ctx.block_type_atoms_forming_chemical_bonds,                          \
        ctx.pose_stack_inter_block_bondsep,                                   \
        ctx.global_params,                                                    \
        max_important_bond_separation,                                        \
        pose_ind,                                                             \
        rot_ind1,                                                             \
        rot_ind2,                                                             \
        block_ind1,                                                           \
        block_ind2,                                                           \
        block_type1,                                                          \
        block_type2,                                                          \
        n_atoms1,                                                             \
        n_atoms2,                                                             \
        inter_dat,                                                            \
        shared.m,                                                             \
        lane);                                                                \
  }

// LOAD_INTERRES1_TILE_DATA_TO_SHARED
// captures:
//    coords (TView<Vec<Real, 3>, 2, D>)
//    ctx.block_type_atom_types (TView<Int, 2, D>)
//    ctx.type_params (TView<LJLKTypeParams<Real>, 1, D>)
//    ctx.block_type_path_distance (TView<Int, 3, D>)
#define LOAD_INTERRES1_TILE_DATA_TO_SHARED          \
  TMOL_DEVICE_FUNC(                                 \
      int tile_ind,                                 \
      int start_atom1,                              \
      int n_atoms_to_load1,                         \
      TMOL_THREAD LJLKScoringData<Real>& inter_dat, \
      TMOL_SHARED_REF(shared_mem_union) shared) {   \
    ljlk_load_interres1_tile_data_to_shared<        \
        DeviceDispatch,                             \
        D,                                          \
        nt,                                         \
        ljlk_tile_size,                             \
        ljlk_max_n_conn,                            \
        Real,                                       \
        Int>(                                       \
        ctx.rot_coords,                             \
        ctx.block_type_atom_types,                  \
        ctx.type_params,                            \
        ctx.block_type_path_distance,               \
        tile_ind,                                   \
        start_atom1,                                \
        n_atoms_to_load1,                           \
        inter_dat,                                  \
        shared.m,                                   \
        lane);                                      \
  }

// LOAD_INTERRES2_TILE_DATA_TO_SHARED
//   same as LOAD_INTERRES1_TILE_DATA_TO_SHARED but saves to inter_dat.r2
// captures:
//    coords (TView<Vec<Real, 3>, 2, D>)
//    ctx.block_type_atom_types (TView<Int, 2, D>)
//    ctx.type_params (TView<LJLKTypeParams<Real>, 1, D>)
//    ctx.block_type_path_distance (TView<Int, 3, D>)
#define LOAD_INTERRES2_TILE_DATA_TO_SHARED          \
  TMOL_DEVICE_FUNC(                                 \
      int tile_ind,                                 \
      int start_atom2,                              \
      int n_atoms_to_load2,                         \
      TMOL_THREAD LJLKScoringData<Real>& inter_dat, \
      TMOL_SHARED_REF(shared_mem_union) shared) {   \
    ljlk_load_interres2_tile_data_to_shared<        \
        DeviceDispatch,                             \
        D,                                          \
        nt,                                         \
        ljlk_tile_size,                             \
        ljlk_max_n_conn,                            \
        Real,                                       \
        Int>(                                       \
        ctx.rot_coords,                             \
        ctx.block_type_atom_types,                  \
        ctx.type_params,                            \
        ctx.block_type_path_distance,               \
        tile_ind,                                   \
        start_atom2,                                \
        n_atoms_to_load2,                           \
        inter_dat,                                  \
        shared.m,                                   \
        lane);                                      \
  }

// LOAD_INTERRES_DATA_FROM_SHARED
// captures:
//    nothing
#define LOAD_INTERRES_DATA_FROM_SHARED   \
  TMOL_DEVICE_FUNC(                      \
      int,                               \
      int,                               \
      TMOL_SHARED_REF(shared_mem_union), \
      TMOL_THREAD LJLKScoringData<Real>&) {}

// Fused LJ/LK traversal: LJ is evaluated for every atom pair and LK for the
// heavy subset inside the pair function, reusing geometry and count-pair work.
#define EVAL_INTERRES_ATOM_PAIR_SCORES_FUSED                    \
  TMOL_DEVICE_FUNC(                                             \
      TMOL_THREAD LJLKScoringData<Real>& inter_dat,             \
      int start_atom1,                                          \
      int start_atom2) {                                        \
    auto eval_scores_for_atom_pairs = ([&](int tid) {           \
      auto LJLK = tmol::score::common::InterResBlockEvaluation< \
          LJLKScoringData,                                      \
          AllAtomPairSelector,                                  \
          D,                                                    \
          ljlk_tile_size,                                       \
          nt,                                                   \
          3,                                                    \
          Real,                                                 \
          Int>::                                                \
          eval_interres_atom_pair(                              \
              tid,                                              \
              start_atom1,                                      \
              start_atom2,                                      \
              score_inter_ljlk_atom_pair,                       \
              inter_dat);                                       \
      inter_dat.total_ljatr += tmol::portable::get<0>(LJLK);    \
      inter_dat.total_ljrep += tmol::portable::get<1>(LJLK);    \
      inter_dat.total_lk += tmol::portable::get<2>(LJLK);       \
    });                                                         \
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(      \
        eval_scores_for_atom_pairs, lane);                      \
  }

// STORE_CALCULATED_ENERGIES
//    store energies if we are NOT computing per-blockpair
// captures:
//    ctx.output_block_pair_energies (bool)
//    ctx.output (TView<Real, 4, D>)
#define STORE_POSE_CALCULATED_ENERGIES                                         \
  TMOL_DEVICE_FUNC(                                                            \
      TMOL_THREAD LJLKScoringData<Real>& score_dat,                            \
      TMOL_SHARED_REF(shared_mem_union) shared) {                              \
    auto reduce_energies = ([&](int tid) {                                     \
      (void)cpu_pose_accum; /* nvcc: first-capture outside the */              \
      /* constexpr-if below, which may not introduce a capture */              \
      Real const cta_total_ljatr =                                             \
          DeviceDispatch<D>::template reduce_in_workgroup<nt>(                 \
              score_dat.total_ljatr,                                           \
              shared,                                                          \
              tmol::portable::plus_t<Real>(),                                  \
              lane);                                                           \
      Real const cta_total_ljrep =                                             \
          DeviceDispatch<D>::template reduce_in_workgroup<nt>(                 \
              score_dat.total_ljrep,                                           \
              shared,                                                          \
              tmol::portable::plus_t<Real>(),                                  \
              lane);                                                           \
      Real const cta_total_lk =                                                \
          DeviceDispatch<D>::template reduce_in_workgroup<nt>(                 \
              score_dat.total_lk,                                              \
              shared,                                                          \
              tmol::portable::plus_t<Real>(),                                  \
              lane);                                                           \
                                                                               \
      if (tid == 0) {                                                          \
        int const p = score_dat.pose_ind;                                      \
        if (ctx.output_block_pair_energies) {                                  \
          int const b1 = score_dat.block_ind1;                                 \
          int const b2 = score_dat.block_ind2;                                 \
          ctx.output[0][p][b1][b2] = cta_total_ljatr;                          \
          ctx.output[1][p][b1][b2] = cta_total_ljrep;                          \
          ctx.output[2][p][b1][b2] = cta_total_lk;                             \
        } else {                                                               \
          if constexpr (D == tmol::Device::CPU) {                              \
            cpu_pose_accum[3 * p] += cta_total_ljatr;                          \
            cpu_pose_accum[3 * p + 1] += cta_total_ljrep;                      \
            cpu_pose_accum[3 * p + 2] += cta_total_lk;                         \
          } else {                                                             \
            accumulate<D, Real>::add(ctx.output[0][p][0][0], cta_total_ljatr); \
            accumulate<D, Real>::add(ctx.output[1][p][0][0], cta_total_ljrep); \
            accumulate<D, Real>::add(ctx.output[2][p][0][0], cta_total_lk);    \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    });                                                                        \
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(                     \
        reduce_energies, lane);                                                \
  }

// STORE_ROTAMER_CALCULATED_ENERGIES
//    store energies if we ARE computing rotamer pair energies
// captures:
//    cta (int)
//    ctx.output (TView<Real, 4, D>)
#define STORE_ROTAMER_CALCULATED_ENERGIES                      \
  TMOL_DEVICE_FUNC(                                            \
      TMOL_THREAD LJLKScoringData<Real>& score_dat,            \
      TMOL_SHARED_REF(shared_mem_union) shared) {              \
    auto reduce_energies = ([&](int tid) {                     \
      Real const cta_total_ljatr =                             \
          DeviceDispatch<D>::template reduce_in_workgroup<nt>( \
              score_dat.total_ljatr,                           \
              shared,                                          \
              tmol::portable::plus_t<Real>(),                  \
              lane);                                           \
      Real const cta_total_ljrep =                             \
          DeviceDispatch<D>::template reduce_in_workgroup<nt>( \
              score_dat.total_ljrep,                           \
              shared,                                          \
              tmol::portable::plus_t<Real>(),                  \
              lane);                                           \
      Real const cta_total_lk =                                \
          DeviceDispatch<D>::template reduce_in_workgroup<nt>( \
              score_dat.total_lk,                              \
              shared,                                          \
              tmol::portable::plus_t<Real>(),                  \
              lane);                                           \
      if (tid == 0) {                                          \
        ctx.output[0][cta] = cta_total_ljatr;                  \
        ctx.output[1][cta] = cta_total_ljrep;                  \
        ctx.output[2][cta] = cta_total_lk;                     \
      }                                                        \
    });                                                        \
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(     \
        reduce_energies, lane);                                \
  }

// LOAD_TILE_INVARIANT_INTRARES_DATA
// captures:
//    pose_stack_block_coord_offset (TView<Int, 2, D>)
//    ctx.global_params (TView<LJGlobalParams<Real>, 1, D>)
//    max_important_bond_separation (int)
#define LOAD_TILE_INVARIANT_INTRARES_DATA                                     \
  TMOL_DEVICE_FUNC(                                                           \
      int pose_ind,                                                           \
      int rot_ind1,                                                           \
      int block_ind1,                                                         \
      int block_type1,                                                        \
      int n_atoms1,                                                           \
      TMOL_THREAD LJLKScoringData<Real>& intra_dat,                           \
      TMOL_SHARED_REF(shared_mem_union) shared) {                             \
    ljlk_load_tile_invariant_intrares_data<DeviceDispatch, D, nt, Int, Real>( \
        ctx.rot_coord_offset,                                                 \
        ctx.global_params,                                                    \
        max_important_bond_separation,                                        \
        pose_ind,                                                             \
        rot_ind1,                                                             \
        block_ind1,                                                           \
        block_type1,                                                          \
        n_atoms1,                                                             \
        intra_dat,                                                            \
        shared.m,                                                             \
        lane);                                                                \
  }

// LOAD_INTRARES1_TILE_DATA_TO_SHARED
// captures:
//    coords (TView<Vec<Real, 3>, 2, D>)
//    ctx.block_type_atom_types (TView<Int, 2, D>)
//    ctx.type_params (TView<LJLKTypeParams<Real>, 1, D>)
#define LOAD_INTRARES1_TILE_DATA_TO_SHARED          \
  TMOL_DEVICE_FUNC(                                 \
      int tile_ind,                                 \
      int start_atom1,                              \
      int n_atoms_to_load1,                         \
      TMOL_THREAD LJLKScoringData<Real>& intra_dat, \
      TMOL_SHARED_REF(shared_mem_union) shared) {   \
    ljlk_load_intrares1_tile_data_to_shared<        \
        DeviceDispatch,                             \
        D,                                          \
        nt,                                         \
        ljlk_tile_size,                             \
        ljlk_max_n_conn,                            \
        Real,                                       \
        Int>(                                       \
        ctx.rot_coords,                             \
        ctx.block_type_atom_types,                  \
        ctx.type_params,                            \
        tile_ind,                                   \
        start_atom1,                                \
        n_atoms_to_load1,                           \
        intra_dat,                                  \
        shared.m,                                   \
        lane);                                      \
  }

// LOAD_INTRARES2_TILE_DATA_TO_SHARED
//    same as LOAD_INTRARES1_TILE_DATA_TO_SHARED but assign to intra_dat.r2
// captures:
//    coords (TView<Vec<Real, 3>, 2, D>)
//    ctx.block_type_atom_types (TView<Int, 2, D>)
//    ctx.type_params (TView<LJLKTypeParams<Real>, 1, D>)
#define LOAD_INTRARES2_TILE_DATA_TO_SHARED          \
  TMOL_DEVICE_FUNC(                                 \
      int tile_ind,                                 \
      int start_atom2,                              \
      int n_atoms_to_load2,                         \
      TMOL_THREAD LJLKScoringData<Real>& intra_dat, \
      TMOL_SHARED_REF(shared_mem_union) shared) {   \
    ljlk_load_intrares2_tile_data_to_shared<        \
        DeviceDispatch,                             \
        D,                                          \
        nt,                                         \
        ljlk_tile_size,                             \
        ljlk_max_n_conn,                            \
        Real,                                       \
        Int>(                                       \
        ctx.rot_coords,                             \
        ctx.block_type_atom_types,                  \
        ctx.type_params,                            \
        tile_ind,                                   \
        start_atom2,                                \
        n_atoms_to_load2,                           \
        intra_dat,                                  \
        shared.m,                                   \
        lane);                                      \
  }

// LOAD_INTRARES_DATA_FROM_SHARED
// captures:
//     nothing
#define LOAD_INTRARES_DATA_FROM_SHARED                \
  TMOL_DEVICE_FUNC(                                   \
      int tile_ind1,                                  \
      int tile_ind2,                                  \
      TMOL_SHARED_REF(shared_mem_union) shared,       \
      TMOL_THREAD LJLKScoringData<Real>& intra_dat) { \
    ljlk_load_intrares_data_from_shared(              \
        tile_ind1, tile_ind2, shared.m, intra_dat);   \
  }

#define EVAL_INTRARES_ATOM_PAIR_SCORES_FUSED                    \
  TMOL_DEVICE_FUNC(                                             \
      TMOL_THREAD LJLKScoringData<Real>& intra_dat,             \
      int start_atom1,                                          \
      int start_atom2) {                                        \
    auto eval_scores_for_atom_pairs = ([&](int tid) {           \
      auto LJLK = tmol::score::common::IntraResBlockEvaluation< \
          LJLKScoringData,                                      \
          AllAtomPairSelector,                                  \
          D,                                                    \
          ljlk_tile_size,                                       \
          nt,                                                   \
          3,                                                    \
          Real,                                                 \
          Int>::                                                \
          eval_intrares_atom_pairs(                             \
              tid,                                              \
              start_atom1,                                      \
              start_atom2,                                      \
              score_intra_ljlk_atom_pair,                       \
              intra_dat);                                       \
      intra_dat.total_ljatr += tmol::portable::get<0>(LJLK);    \
      intra_dat.total_ljrep += tmol::portable::get<1>(LJLK);    \
      intra_dat.total_lk += tmol::portable::get<2>(LJLK);       \
    });                                                         \
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(      \
        eval_scores_for_atom_pairs, lane);                      \
  }
// end of macro definitions

enum class LJLKMode { Energy, EnergyAndDerivatives, Backward };

template <
    LJLKMode Mode,
    bool Rotamer,
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    typename launch_t,
    typename Ctx,
    typename Shared,
    typename PoseAccum>
TMOL_HOST_DEVICE_FUNC void ljlk_evaluate_pair(
    TMOL_THREAD Ctx const& ctx,
    TMOL_SHARED_REF(Shared) shared,
    int cta,
    PoseAccum cpu_pose_accum,
    WorkgroupLane<D> lane = {}) {
  CTA_LAUNCH_T_PARAMS;
  typedef LJLKSharedMem<Real, ljlk_tile_size, ljlk_max_n_conn, launch_t>
      shared_mem_union;
  // Pair arithmetic needs only these three views. Keeping the full launch
  // context out of this frequently copied closure avoids copying topology
  // metadata for every tile while preserving the same native operations.
  auto const type_params = ctx.type_params;
  auto const coordinate_derivatives = ctx.dV_dcoords;
  auto const upstream_weights = ctx.dTdV;
  // Backward accumulates one weighted gradient lane. Resolve its subview
  // once per pair workgroup, not once for every atom pair.
  auto weighted_derivatives = decltype(ctx.dV_dcoords[0]){};
  if constexpr (Mode == LJLKMode::Backward)
    weighted_derivatives = ctx.dV_dcoords[0];
  auto atom_pair_ljlk_fn =
      ([=] TMOL_DEVICE_FUNC(
           int atom_tile_ind1,
           int atom_tile_ind2,
           int start_atom1,
           int start_atom2,
           TMOL_THREAD LJLKScoringData<Real> const& score_dat,
           int cp_separation) {
        (void)type_params;
        (void)coordinate_derivatives;
        (void)upstream_weights;
        (void)weighted_derivatives;
        (void)cta;  // NVCC captures outside constexpr branches.
        // Each mode calls the same mature scalar physics and shares all
        // topology, tile loading, count-pair exclusions and atom traversal.
        if constexpr (Mode == LJLKMode::Backward) {
          Real weights[3];
          for (int term = 0; term < 3; ++term) {
            if constexpr (Rotamer)
              weights[term] = upstream_weights[term][cta];
            else
              weights[term] =
                  upstream_weights[term][score_dat.pose_ind]
                                  [score_dat.block_ind1][score_dat.block_ind2];
          }
          ljlk_atom_derivs<Real, D>(
              atom_tile_ind1,
              atom_tile_ind2,
              start_atom1,
              start_atom2,
              score_dat,
              cp_separation,
              weights[0],
              weights[1],
              weights[2],
              weighted_derivatives,
              type_params);
          return tmol::portable::array<Real, 3>{0, 0, 0};
        } else if constexpr (Mode == LJLKMode::EnergyAndDerivatives) {
          return ljlk_atom_energy_and_derivs_full<Real, D>(
              atom_tile_ind1,
              atom_tile_ind2,
              start_atom1,
              start_atom2,
              score_dat,
              cp_separation,
              coordinate_derivatives,
              type_params);
        } else {
          return ljlk_atom_energy<Real, D>(
              atom_tile_ind1,
              atom_tile_ind2,
              score_dat,
              cp_separation,
              type_params);
        }
      });

  auto score_inter_ljlk_atom_pair =
      ([=] SCORE_INTER_LJ_ATOM_PAIR(atom_pair_ljlk_fn));
  auto score_intra_ljlk_atom_pair =
      ([=] SCORE_INTRA_LJ_ATOM_PAIR(atom_pair_ljlk_fn));

  auto load_block_coords_and_params_into_shared =
      ([=] LOAD_BLOCK_COORDS_AND_PARAMS_INTO_SHARED);

  auto load_block_into_shared = ([=] LOAD_BLOCK_INTO_SHARED);

  int const max_important_bond_separation = 4;

  int pose_ind, block_ind1, block_ind2, rot_ind1, rot_ind2;
  if constexpr (Rotamer) {
    pose_ind = ctx.dispatch_indices[0][cta];
    rot_ind1 = ctx.dispatch_indices[1][cta];
    rot_ind2 = ctx.dispatch_indices[2][cta];
    if (rot_ind1 < 0 || rot_ind2 < 0) return;
    block_ind1 = ctx.block_ind_for_rot[rot_ind1];
    block_ind2 = ctx.block_ind_for_rot[rot_ind2];
  } else {
    pose_ind = cta / ctx.max_n_upper_triangle_inds;
    auto pair = common::upper_triangle_inds_from_linear_index(
        cta % ctx.max_n_upper_triangle_inds, ctx.max_n_blocks + 1);
    block_ind1 = common::get<0>(pair);
    block_ind2 = common::get<1>(pair) - 1;
    if (!ctx.use_shared_compact_block_neighbors
        && ctx.scratch_rot_neighbors[pose_ind][block_ind1][block_ind2] == 0)
      return;
    if constexpr (Mode == LJLKMode::Backward) {
      if (ctx.dTdV[0][pose_ind][block_ind1][block_ind2] == 0
          && ctx.dTdV[1][pose_ind][block_ind1][block_ind2] == 0
          && ctx.dTdV[2][pose_ind][block_ind1][block_ind2] == 0)
        return;
    }
    rot_ind1 = ctx.rot_offset_for_block[pose_ind][block_ind1];
    rot_ind2 = ctx.rot_offset_for_block[pose_ind][block_ind2];
    if (rot_ind1 < 0 || rot_ind2 < 0) return;
  }

  int const block_type1 = ctx.block_type_ind_for_rot[rot_ind1];
  int const block_type2 = ctx.block_type_ind_for_rot[rot_ind2];

  if (block_type1 < 0 || block_type2 < 0) {
    return;
  }

  int const n_atoms1 = ctx.block_type_n_atoms[block_type1];
  int const n_atoms2 = ctx.block_type_n_atoms[block_type2];

  auto load_tile_invariant_interres_data =
      ([=] LOAD_TILE_INVARIANT_INTERRES_DATA);

  auto load_interres1_tile_data_to_shared =
      ([=] LOAD_INTERRES1_TILE_DATA_TO_SHARED);

  auto load_interres2_tile_data_to_shared =
      ([=] LOAD_INTERRES2_TILE_DATA_TO_SHARED);

  auto load_interres_data_from_shared = ([=] LOAD_INTERRES_DATA_FROM_SHARED);

  auto eval_interres_atom_pair_scores =
      ([=] EVAL_INTERRES_ATOM_PAIR_SCORES_FUSED);

  auto store_calculated_energies =
      ([=] TMOL_DEVICE_FUNC(
           TMOL_THREAD LJLKScoringData<Real>& score_dat,
           TMOL_SHARED_REF(shared_mem_union) storage) {
        // Keep captures visible to NVCC outside constexpr branches.
        (void)ctx;
        (void)cta;
        (void)cpu_pose_accum;
        (void)lane;
        if constexpr (Mode != LJLKMode::Backward) {
          if constexpr (Rotamer) {
            auto store = ([=] STORE_ROTAMER_CALCULATED_ENERGIES);
            store(score_dat, storage);
          } else {
            auto store = ([=] STORE_POSE_CALCULATED_ENERGIES);
            store(score_dat, storage);
          }
        }
      });

  auto load_tile_invariant_intrares_data =
      ([=] LOAD_TILE_INVARIANT_INTRARES_DATA);

  auto load_intrares1_tile_data_to_shared =
      ([=] LOAD_INTRARES1_TILE_DATA_TO_SHARED);

  auto load_intrares2_tile_data_to_shared =
      ([=] LOAD_INTRARES2_TILE_DATA_TO_SHARED);

  auto load_intrares_data_from_shared = ([=] LOAD_INTRARES_DATA_FROM_SHARED);

  auto eval_intrares_atom_pair_scores =
      ([=] EVAL_INTRARES_ATOM_PAIR_SCORES_FUSED);

  tmol::score::common::tile_evaluate_rot_pair<
      DeviceDispatch,
      D,
      LJLKScoringData<Real>,
      LJLKScoringData<Real>,
      Real,
      ljlk_tile_size>(
      shared,
      pose_ind,
      rot_ind1,
      rot_ind2,
      block_ind1,
      block_ind2,
      block_type1,
      block_type2,
      n_atoms1,
      n_atoms2,
      load_tile_invariant_interres_data,
      load_interres1_tile_data_to_shared,
      load_interres2_tile_data_to_shared,
      load_interres_data_from_shared,
      eval_interres_atom_pair_scores,
      store_calculated_energies,
      load_tile_invariant_intrares_data,
      load_intrares1_tile_data_to_shared,
      load_intrares2_tile_data_to_shared,
      load_intrares_data_from_shared,
      eval_intrares_atom_pair_scores,
      store_calculated_energies,
      lane);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    typename launch_t,
    typename Ctx,
    typename Shared,
    typename PoseAccum>
TMOL_HOST_DEVICE_FUNC void ljlk_eval_energies_by_block(
    TMOL_THREAD Ctx const& ctx,
    TMOL_SHARED_REF(Shared) shared,
    int cta,
    PoseAccum cpu_pose_accum,
    WorkgroupLane<D> lane = {}) {
  ljlk_evaluate_pair<
      LJLKMode::Energy,
      false,
      DeviceDispatch,
      D,
      Real,
      Int,
      launch_t>(ctx, shared, cta, cpu_pose_accum, lane);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    typename launch_t,
    typename Ctx,
    typename Shared,
    typename PoseAccum>
TMOL_HOST_DEVICE_FUNC void ljlk_eval_rotamer_energies_by_block(
    TMOL_THREAD Ctx const& ctx,
    TMOL_SHARED_REF(Shared) shared,
    int cta,
    PoseAccum cpu_pose_accum,
    WorkgroupLane<D> lane = {}) {
  ljlk_evaluate_pair<
      LJLKMode::Energy,
      true,
      DeviceDispatch,
      D,
      Real,
      Int,
      launch_t>(ctx, shared, cta, cpu_pose_accum, lane);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    typename launch_t,
    typename Ctx,
    typename Shared,
    typename PoseAccum>
TMOL_HOST_DEVICE_FUNC void ljlk_eval_energies(
    TMOL_THREAD Ctx const& ctx,
    TMOL_SHARED_REF(Shared) shared,
    int cta,
    PoseAccum cpu_pose_accum,
    WorkgroupLane<D> lane = {}) {
  ljlk_evaluate_pair<
      LJLKMode::EnergyAndDerivatives,
      false,
      DeviceDispatch,
      D,
      Real,
      Int,
      launch_t>(ctx, shared, cta, cpu_pose_accum, lane);
}

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol

// Keep implementation macros private to this header.
#undef SCORE_INTER_LJ_ATOM_PAIR
#undef SCORE_INTRA_LJ_ATOM_PAIR
#undef LOAD_BLOCK_COORDS_AND_PARAMS_INTO_SHARED
#undef LOAD_BLOCK_INTO_SHARED
#undef LOAD_TILE_INVARIANT_INTERRES_DATA
#undef LOAD_INTERRES1_TILE_DATA_TO_SHARED
#undef LOAD_INTERRES2_TILE_DATA_TO_SHARED
#undef LOAD_INTERRES_DATA_FROM_SHARED
#undef EVAL_INTERRES_ATOM_PAIR_SCORES_FUSED
#undef STORE_POSE_CALCULATED_ENERGIES
#undef STORE_ROTAMER_CALCULATED_ENERGIES
#undef LOAD_TILE_INVARIANT_INTRARES_DATA
#undef LOAD_INTRARES1_TILE_DATA_TO_SHARED
#undef LOAD_INTRARES2_TILE_DATA_TO_SHARED
#undef LOAD_INTRARES_DATA_FROM_SHARED
#undef EVAL_INTRARES_ATOM_PAIR_SCORES_FUSED
