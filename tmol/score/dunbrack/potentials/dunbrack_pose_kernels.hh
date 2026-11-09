#pragma once
#include <tmol/utility/tensor/context_manager.hh>

#include <tmol/utility/portable/matrix.hh>
#include <type_traits>

#include <tmol/utility/tensor/TensorAccessor.h>

#include <tmol/score/unresolved_atom_types.hh>
#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/uaid_util.hh>
#include <tmol/score/common/warp_segreduce.hh>
#include <tmol/score/common/warp_stride_reduce.hh>

// Complete shared pose science. Bindings own output and scratch allocation.
// Forward scratch may be uninitialized: each active residue assigns all
// consumed dihedral indices, values, derivatives and table assignments before
// use. Term derivative scratch is overwritten before each contribution.
// Padded/unassigned residues do not read scratch. Only V and dV_dx must start
// at zero, since scoring accumulates into those output buffers.

#include "potentials.hh"

namespace tmol {
namespace score {
namespace dunbrack {
namespace potentials {

using tmol::portable::Vec;

// Output layout is the only difference between dense pose/block scores and
// sparse rotamer entries. Preserve per-entry accumulation order in both modes.
template <Device D, size_t Rank, typename EnergyReal, typename Real>
TMOL_HOST_DEVICE_FUNC void emit_dunbrack_energy(
    TView<EnergyReal, Rank, D> V,
    int term,
    int pose,
    int block,
    int rotamer,
    bool block_pairs,
    Real value) {
  if constexpr (Rank == 2) {
    common::accumulate<D, EnergyReal>::add(
        V[term][block_pairs ? rotamer : pose], value);
  } else {
    if (block_pairs)
      V[term][pose][block][block] = value;
    else
      common::accumulate<D, EnergyReal>::add(V[term][pose][0][0], value);
  }
}

// Evaluate each derivative (including its score weight) in Real, then widen
// only its accumulation. CUDA float callers cast the completed sum once.
template <Device D, typename GradReal, typename Real>
TMOL_HOST_DEVICE_FUNC void emit_dunbrack_derivative(
    Vec<GradReal, 3>& target, Vec<Real, 3> value) {
  for (int axis = 0; axis < 3; ++axis)
    common::accumulate<D, GradReal>::add(target[axis], GradReal(value[axis]));
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    typename SizeInt = int64_t,
    size_t OutRank = 4,
    typename EnergyReal = Real,
    typename GradReal = Real>
void dunbrack_pose_forward(
    ContextManager& mgr,
    TView<Vec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 3, D> block_type_atom_downstream_of_conn,
    TView<Real, 3, D> rotameric_neglnprob_tables,
    TView<Vec<SizeInt, 2>, 1, D> rotprob_table_sizes,
    TView<Vec<SizeInt, 2>, 1, D> rotprob_table_strides,
    TView<Real, 3, D> rotameric_mean_tables,
    TView<Real, 3, D> rotameric_sdev_tables,
    TView<Vec<SizeInt, 2>, 1, D> rotmean_table_sizes,
    TView<Vec<SizeInt, 2>, 1, D> rotmean_table_strides,
    TView<Vec<Real, 2>, 1, D> rotameric_bb_start,
    TView<Vec<Real, 2>, 1, D> rotameric_bb_step,
    TView<Vec<Real, 2>, 1, D> rotameric_bb_periodicity,
    TView<Int, 1, D> rotameric_rotind2tableind,
    TView<Int, 1, D> semirotameric_rotind2tableind,
    TView<Real, 4, D> semirotameric_tables,
    TView<Vec<SizeInt, 3>, 1, D> semirot_table_sizes,
    TView<Vec<SizeInt, 3>, 1, D> semirot_table_strides,
    TView<Vec<Real, 3>, 1, D> semirot_start,
    TView<Vec<Real, 3>, 1, D> semirot_step,
    TView<Vec<Real, 3>, 1, D> semirot_periodicity,
    TView<Int, 1, D> block_n_dihedrals,
    TView<UnresolvedAtomID<Int>, 3, D> block_dih_uaids,
    TView<Int, 1, D> block_rotamer_table_set,
    TView<Int, 1, D> block_rotameric_index,
    TView<Int, 1, D> block_semirotameric_index,
    TView<Int, 1, D> block_n_chi,
    TView<Int, 1, D> block_n_rotameric_chi,
    TView<Int, 1, D> block_probability_table_offset,
    TView<Int, 1, D> block_mean_table_offset,
    TView<Int, 1, D> block_rotamer_index_to_table_index,
    TView<Int, 1, D> block_semirotameric_tableset_offset,
    TView<Int, 1, D> block_is_mirrored,
    bool output_block_pair_energies,
    bool compute_derivs,
    TView<EnergyReal, OutRank, D> V,
    TView<Vec<GradReal, 3>, 2, D> dV_dx,
    TView<Vec<Int, 4>, 2, D> dihedral_atom_inds,
    TView<Real, 2, D> dihedral_values,
    TView<tmol::portable::Mat<Real, 4, 3>, 2, D> dihedral_deriv,
    TView<tmol::portable::Mat<Real, 4, 3>, 2, D> dterm_dtor_xyz,
    TView<Int, 1, D> rotameric_rottable_assignment,
    TView<Int, 1, D> semirotameric_rottable_assignment,
    TView<Int, 1, D> pose_ind_for_rot = {},
    TView<Int, 1, D> block_ind_for_rot = {},
    TView<Int, 1, D> block_type_ind_for_rot = {},
    TView<Int, 2, D> dispatch_indices = {}) {
  static_assert(OutRank == 4 || OutRank == 2);

  int const n_atoms = rot_coords.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const n_block_types = block_n_dihedrals.size(0);

  int const max_n_dih = block_dih_uaids.size(1);
  int const max_n_conns = pose_stack_inter_block_connections.size(2);

  int const DIH_N_ATOMS = 4;

  assert(first_rot_block_type.size(0) == n_poses);
  assert(first_rot_block_type.size(1) == max_n_blocks);

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_connections.size(2) == max_n_conns);

  assert(block_type_atom_downstream_of_conn.size(0) == n_block_types);
  assert(block_type_atom_downstream_of_conn.size(1) == max_n_conns);
  //     block_type_atom_downstream_of_conn.size(2) == max number of atoms in
  //     any block type

  assert(block_n_dihedrals.size(0) == n_block_types);

  assert(block_dih_uaids.size(0) == n_block_types);
  assert(block_dih_uaids.size(1) == max_n_dih);
  assert(block_dih_uaids.size(2) == DIH_N_ATOMS);

  assert(block_rotamer_table_set.size(0) == n_block_types);
  assert(block_rotameric_index.size(0) == n_block_types);
  assert(block_semirotameric_index.size(0) == n_block_types);
  assert(block_n_chi.size(0) == n_block_types);
  assert(block_n_rotameric_chi.size(0) == n_block_types);
  assert(block_probability_table_offset.size(0) == n_block_types);
  assert(block_mean_table_offset.size(0) == n_block_types);
  assert(block_rotamer_index_to_table_index.size(0) == n_block_types);
  assert(block_semirotameric_tableset_offset.size(0) == n_block_types);
  assert(block_is_mirrored.size(0) == n_block_types);

  bool const accumulate_derivs = compute_derivs && !output_block_pair_energies;
  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;

  auto func = ([=] TMOL_DEVICE_FUNC(int ind, auto deriv_tag) {
    constexpr bool calc_derivs = decltype(deriv_tag)::value;
    int const pose_index =
        OutRank == 2 ? pose_ind_for_rot[ind] : ind / max_n_blocks;
    int const block_index =
        OutRank == 2 ? block_ind_for_rot[ind] : ind % max_n_blocks;
    int const block_type_index =
        OutRank == 2 ? block_type_ind_for_rot[ind]
                     : first_rot_block_type[pose_index][block_index];
    if (block_type_index == -1) {
      // in non-rotamer scoring (both whole-pose and block-pair scoring)
      // we will launch n_poses * max_n_blocks_per_pose threads
      // with some threads being launched for blocks with -1 block type.
      return;
    }

    int const rotamer_index =
        OutRank == 2 ? ind : first_rot_for_block[pose_index][block_index];
    // Ordinary constant condition lets NVCC capture these views before its
    // constexpr-if capture restriction; the unused rank branch is eliminated.
    if (OutRank == 2 && output_block_pair_energies) {
      dispatch_indices[0][rotamer_index] = pose_index;
      dispatch_indices[1][rotamer_index] = rotamer_index;
      dispatch_indices[2][rotamer_index] = rotamer_index;
    }
    if (block_rotamer_table_set[block_type_index] == -1) return;
    if (rotamer_index == -1) {
      // This should never happen
      return;
    }

    for (int ii = 0; ii < block_n_dihedrals[block_type_index]; ii++) {
      auto dih_uaids = block_dih_uaids[block_type_index][ii];
      bool fail = false;
      for (int jj = 0; jj < DIH_N_ATOMS; jj++) {
        UnresolvedAtomID<Int> uaid = dih_uaids[jj];

        if (uaid.atom_id == -1 && uaid.conn_id == -1) {  // Dihedral undefined
          fail = true;
          break;
        }

        dihedral_atom_inds[rotamer_index][ii][jj] =
            resolve_rotamer_atom_from_uaid(
                uaid,
                rotamer_index,
                block_index,
                pose_index,
                rot_coord_offset,
                first_rot_for_block,
                first_rot_block_type,
                pose_stack_inter_block_connections,
                block_type_atom_downstream_of_conn);
        if (dihedral_atom_inds[rotamer_index][ii][jj] == -1) {
          // UAID resolution failed
          fail = true;
          break;
        }
      }
      if (fail) {  // if the dihedral resolution failed, let's fill the cached
                   // value with -1s since we might have partially filled it
                   // above
        dihedral_atom_inds[rotamer_index][ii] =
            tmol::portable::Vec<Int, 4>::Constant(-1);
      }

      // a terminus has no phi or no psi, so a default stands in for it. It is
      // a torsion like any other and must be negated for a mirrored residue,
      // or the two forms read different points of the same table.
      const Real mirror = block_is_mirrored[block_type_index] ? -1.0 : 1.0;
      const Real PHI_DEFAULT = mirror * -60.0 * M_PI / 180;
      const Real PSI_DEFAULT = mirror * 60.0 * M_PI / 180;

      Real dih_default = (ii == 0)   ? PHI_DEFAULT
                         : (ii == 1) ? PSI_DEFAULT
                                     : 0.0;

      if (calc_derivs) {
        measure_dihedral_V_dV(
            TensorAccessor<Vec<Real, 3>, 1, D>(rot_coords),
            dihedral_atom_inds[rotamer_index][ii],
            dih_default,
            dihedral_values[rotamer_index][ii],
            dihedral_deriv[rotamer_index][ii]);
      } else {
        measure_dihedral_V(
            TensorAccessor<Vec<Real, 3>, 1, D>(rot_coords),
            dihedral_atom_inds[rotamer_index][ii],
            dih_default,
            dihedral_values[rotamer_index][ii]);
      }
    }

    Int local_rotameric, local_semirotameric;
    Int& rotameric_assignment =
        OutRank == 2 ? local_rotameric
                     : rotameric_rottable_assignment[rotamer_index];
    Int& semirotameric_assignment =
        OutRank == 2 ? local_semirotameric
                     : semirotameric_rottable_assignment[rotamer_index];
    // Templated on there being 2 backbone dihedrals for canonical aas.
    classify_rotamer_for_block<2>(
        dihedral_values[rotamer_index],
        block_n_rotameric_chi[block_type_index],
        block_rotamer_index_to_table_index[block_type_index],
        rotameric_rotind2tableind,
        semirotameric_rotind2tableind,
        rotameric_assignment,
        semirotameric_assignment);

    if (block_rotameric_index[block_type_index] != -1) {
      Real prob = rotameric_chi_probability_for_block(
          rotameric_neglnprob_tables,
          rotprob_table_sizes,
          rotprob_table_strides,
          rotameric_bb_start,
          rotameric_bb_step,
          rotameric_bb_periodicity,
          block_probability_table_offset[block_type_index],
          block_rotamer_table_set[block_type_index],
          dihedral_values[rotamer_index],
          rotameric_assignment,
          calc_derivs ? dterm_dtor_xyz[rotamer_index]
                      : TensorAccessor<CoordQuad<Real>, 1, D>(),
          calc_derivs ? dihedral_deriv[rotamer_index]
                      : TensorAccessor<CoordQuad<Real>, 1, D>(),
          calc_derivs);

      emit_dunbrack_energy<D>(
          V,
          0,
          pose_index,
          block_index,
          rotamer_index,
          output_block_pair_energies,
          prob);

      if (calc_derivs) {
        // Note that we will accumulate all of the dV_dx derivatives
        // into the phi and psi definiing atoms of the _first rotamers_
        // of residues i+1 and i-1 respectively. This is dedicedly weird
        // unless there is only one rotmaer for each residue
        Vec<Int, DIH_N_ATOMS> phi_ats = dihedral_atom_inds[rotamer_index][0];
        Vec<Int, DIH_N_ATOMS> psi_ats = dihedral_atom_inds[rotamer_index][1];
        for (int j = 0; j < DIH_N_ATOMS; ++j) {
          if (phi_ats[j] != -1)
            emit_dunbrack_derivative<D>(
                dV_dx[0][phi_ats[j]], dterm_dtor_xyz[rotamer_index][0].row(j));
          if (psi_ats[j] != -1)
            emit_dunbrack_derivative<D>(
                dV_dx[0][psi_ats[j]], dterm_dtor_xyz[rotamer_index][1].row(j));
        }
      }
    }

    Real rotameric_chi_dev_penalty = 0.0;
    for (int ii = 0; ii < block_n_rotameric_chi[block_type_index]; ii++) {
      // deviation from chi for chis within this block
      auto Erotdev = block_deviation_penalty_for_chi(
          // DB tables
          rotameric_mean_tables,
          rotameric_sdev_tables,
          rotmean_table_sizes,
          rotmean_table_strides,
          rotameric_bb_start,
          rotameric_bb_step,
          rotameric_bb_periodicity,
          // Block type info
          block_rotamer_table_set[block_type_index],
          block_mean_table_offset[block_type_index],
          block_n_chi[block_type_index],
          // Block info
          dihedral_values[rotamer_index],
          ii,
          rotameric_assignment,
          // Out
          calc_derivs ? dterm_dtor_xyz[rotamer_index]
                      : TensorAccessor<CoordQuad<Real>, 1, D>(),
          calc_derivs ? dihedral_deriv[rotamer_index]
                      : TensorAccessor<CoordQuad<Real>, 1, D>(),
          calc_derivs);
      if constexpr (OutRank == 2)
        emit_dunbrack_energy<D>(
            V,
            1,
            pose_index,
            block_index,
            rotamer_index,
            output_block_pair_energies,
            Erotdev);
      else
        rotameric_chi_dev_penalty += Erotdev;

      if (calc_derivs) {
        Vec<Int, DIH_N_ATOMS> tor0_ats = dihedral_atom_inds[rotamer_index][0];
        Vec<Int, DIH_N_ATOMS> tor1_ats = dihedral_atom_inds[rotamer_index][1];
        Vec<Int, DIH_N_ATOMS> tor2_ats =
            dihedral_atom_inds[rotamer_index][2 + ii];
        for (int j = 0; j < DIH_N_ATOMS; ++j) {
          if (tor0_ats[j] != -1)
            emit_dunbrack_derivative<D>(
                dV_dx[1][tor0_ats[j]], dterm_dtor_xyz[rotamer_index][0].row(j));
          if (tor1_ats[j] != -1)
            emit_dunbrack_derivative<D>(
                dV_dx[1][tor1_ats[j]], dterm_dtor_xyz[rotamer_index][1].row(j));
          if (tor2_ats[j] != -1)
            emit_dunbrack_derivative<D>(
                dV_dx[1][tor2_ats[j]], dterm_dtor_xyz[rotamer_index][2].row(j));
        }
      }
    }
    if constexpr (OutRank == 4) {
      if (rotameric_chi_dev_penalty != 0.0)
        emit_dunbrack_energy<D>(
            V,
            1,
            pose_index,
            block_index,
            rotamer_index,
            output_block_pair_energies,
            rotameric_chi_dev_penalty);
    }

    if (block_semirotameric_index[block_type_index] != -1) {
      auto Esemi = block_semirotameric_energy(
          semirotameric_tables,
          semirot_table_sizes,
          semirot_table_strides,
          semirot_start,
          semirot_step,
          semirot_periodicity,

          block_semirotameric_index[block_type_index],
          block_semirotameric_tableset_offset[block_type_index],
          block_n_chi[block_type_index] + 1,

          dihedral_values[rotamer_index],
          semirotameric_assignment,

          calc_derivs ? dterm_dtor_xyz[rotamer_index]
                      : TensorAccessor<CoordQuad<Real>, 1, D>(),
          calc_derivs ? dihedral_deriv[rotamer_index]
                      : TensorAccessor<CoordQuad<Real>, 1, D>(),
          calc_derivs);

      emit_dunbrack_energy<D>(
          V,
          2,
          pose_index,
          block_index,
          rotamer_index,
          output_block_pair_energies,
          Esemi);

      if (calc_derivs) {
        int last = block_n_chi[block_type_index] + 1;  // = +2 - 1
        Vec<Int, DIH_N_ATOMS> tor0_ats = dihedral_atom_inds[rotamer_index][0];
        Vec<Int, DIH_N_ATOMS> tor1_ats = dihedral_atom_inds[rotamer_index][1];
        Vec<Int, DIH_N_ATOMS> tor2_ats =
            dihedral_atom_inds[rotamer_index][last];
        for (int j = 0; j < DIH_N_ATOMS; ++j) {
          if (tor0_ats[j] != -1)
            emit_dunbrack_derivative<D>(
                dV_dx[2][tor0_ats[j]], dterm_dtor_xyz[rotamer_index][0].row(j));
          if (tor1_ats[j] != -1)
            emit_dunbrack_derivative<D>(
                dV_dx[2][tor1_ats[j]], dterm_dtor_xyz[rotamer_index][1].row(j));
          if (tor2_ats[j] != -1)
            emit_dunbrack_derivative<D>(
                dV_dx[2][tor2_ats[j]], dterm_dtor_xyz[rotamer_index][2].row(j));
        }
      }
    }
  });

  // Specialize the two paths so unused derivative scratch and arithmetic
  // disappear from score-only kernels without adding per-residue branches.
  if (accumulate_derivs) {
    auto eval =
        ([=] TMOL_DEVICE_FUNC(int ind) { func(ind, std::true_type{}); });
    if constexpr (OutRank == 2) {
      DeviceDispatch<D>::template forall<launch_t>(mgr, n_rots, eval);
    } else {
      DeviceDispatch<D>::template forall_grouped<launch_t>(
          mgr, n_poses, max_n_blocks, eval);
    }
  } else {
    auto eval =
        ([=] TMOL_DEVICE_FUNC(int ind) { func(ind, std::false_type{}); });
    if constexpr (OutRank == 2) {
      if constexpr (D == Device::CPU) {
        if (output_block_pair_energies) {
          DeviceDispatch<D>::template foreach_independent_workgroup<launch_t>(
              mgr, n_rots, eval);
          return;
        }
      }
      DeviceDispatch<D>::template forall<launch_t>(mgr, n_rots, eval);
    } else {
      DeviceDispatch<D>::template forall_grouped<launch_t>(
          mgr, n_poses, max_n_blocks, eval);
    }
  }
}

template <size_t Rank, typename Real, Device D>
TMOL_HOST_DEVICE_FUNC Real dunbrack_score_weight(
    TView<Real, Rank, D> weights, int term, int pose, int block, int rotamer) {
  if constexpr (Rank == 2)
    return weights[term][rotamer];
  else
    return weights[term][pose][block][block];
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    typename SizeInt = int64_t,
    size_t OutRank = 4,
    typename GradReal = Real>
void dunbrack_pose_backward(
    ContextManager& mgr,
    TView<Vec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 3, D> block_type_atom_downstream_of_conn,
    TView<Real, 3, D> rotameric_neglnprob_tables,
    TView<Vec<SizeInt, 2>, 1, D> rotprob_table_sizes,
    TView<Vec<SizeInt, 2>, 1, D> rotprob_table_strides,
    TView<Real, 3, D> rotameric_mean_tables,
    TView<Real, 3, D> rotameric_sdev_tables,
    TView<Vec<SizeInt, 2>, 1, D> rotmean_table_sizes,
    TView<Vec<SizeInt, 2>, 1, D> rotmean_table_strides,
    TView<Vec<Real, 2>, 1, D> rotameric_bb_start,
    TView<Vec<Real, 2>, 1, D> rotameric_bb_step,
    TView<Vec<Real, 2>, 1, D> rotameric_bb_periodicity,
    TView<Int, 1, D> rotameric_rotind2tableind,
    TView<Int, 1, D> semirotameric_rotind2tableind,
    TView<Real, 4, D> semirotameric_tables,
    TView<Vec<SizeInt, 3>, 1, D> semirot_table_sizes,
    TView<Vec<SizeInt, 3>, 1, D> semirot_table_strides,
    TView<Vec<Real, 3>, 1, D> semirot_start,
    TView<Vec<Real, 3>, 1, D> semirot_step,
    TView<Vec<Real, 3>, 1, D> semirot_periodicity,
    TView<Int, 1, D> block_n_dihedrals,
    TView<UnresolvedAtomID<Int>, 3, D> block_dih_uaids,
    TView<Int, 1, D> block_rotamer_table_set,
    TView<Int, 1, D> block_rotameric_index,
    TView<Int, 1, D> block_semirotameric_index,
    TView<Int, 1, D> block_n_chi,
    TView<Int, 1, D> block_n_rotameric_chi,
    TView<Int, 1, D> block_probability_table_offset,
    TView<Int, 1, D> block_mean_table_offset,
    TView<Int, 1, D> block_rotamer_index_to_table_index,
    TView<Int, 1, D> block_semirotameric_tableset_offset,
    TView<Int, 1, D> block_is_mirrored,
    TView<Real, OutRank, D> dTdV,
    TView<Vec<GradReal, 3>, 2, D> dV_dx,
    TView<Vec<Int, 4>, 2, D> dihedral_atom_inds,
    TView<Real, 2, D> dihedral_values,
    TView<tmol::portable::Mat<Real, 4, 3>, 2, D> dihedral_deriv,
    TView<tmol::portable::Mat<Real, 4, 3>, 2, D> dterm_dtor_xyz,
    TView<Int, 1, D> pose_ind_for_rot = {},
    TView<Int, 1, D> block_ind_for_rot = {},
    TView<Int, 1, D> block_type_ind_for_rot = {}) {
  static_assert(OutRank == 4 || OutRank == 2);
  int const n_atoms = rot_coords.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const n_block_types = block_n_dihedrals.size(0);

  int const max_n_dih = block_dih_uaids.size(1);
  int const max_n_conns = pose_stack_inter_block_connections.size(2);

  int const DIH_N_ATOMS = 4;

  assert(first_rot_block_type.size(0) == n_poses);
  assert(first_rot_block_type.size(1) == max_n_blocks);

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_connections.size(2) == max_n_conns);

  assert(block_type_atom_downstream_of_conn.size(0) == n_block_types);
  assert(block_type_atom_downstream_of_conn.size(1) == max_n_conns);
  //     block_type_atom_downstream_of_conn.size(2) == max number of atoms in
  //     any block type

  assert(block_n_dihedrals.size(0) == n_block_types);

  assert(block_dih_uaids.size(0) == n_block_types);
  assert(block_dih_uaids.size(1) == max_n_dih);
  assert(block_dih_uaids.size(2) == DIH_N_ATOMS);

  assert(block_rotamer_table_set.size(0) == n_block_types);
  assert(block_rotameric_index.size(0) == n_block_types);
  assert(block_semirotameric_index.size(0) == n_block_types);
  assert(block_n_chi.size(0) == n_block_types);
  assert(block_n_rotameric_chi.size(0) == n_block_types);
  assert(block_probability_table_offset.size(0) == n_block_types);
  assert(block_mean_table_offset.size(0) == n_block_types);
  assert(block_rotamer_index_to_table_index.size(0) == n_block_types);
  assert(block_semirotameric_tableset_offset.size(0) == n_block_types);
  assert(block_is_mirrored.size(0) == n_block_types);

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;

  auto func = ([=] TMOL_DEVICE_FUNC(int ind) {
    int const pose_index =
        OutRank == 2 ? pose_ind_for_rot[ind] : ind / max_n_blocks;
    int const block_index =
        OutRank == 2 ? block_ind_for_rot[ind] : ind % max_n_blocks;
    int const block_type_index =
        OutRank == 2 ? block_type_ind_for_rot[ind]
                     : first_rot_block_type[pose_index][block_index];
    int const rotamer_index =
        OutRank == 2 ? ind : first_rot_for_block[pose_index][block_index];
    if (block_type_index == -1) {
      // in non-rotamer scoring (both whole-pose and block-pair scoring)
      // we will launch n_poses * max_n_blocks_per_pose threads
      // with some threads being launched for blocks with -1 block type.
      return;
    }

    if (OutRank == 4
        && dunbrack_score_weight(
               dTdV, 0, pose_index, block_index, rotamer_index)
               == 0
        && dunbrack_score_weight(
               dTdV, 1, pose_index, block_index, rotamer_index)
               == 0
        && dunbrack_score_weight(
               dTdV, 2, pose_index, block_index, rotamer_index)
               == 0) {
      return;
    }

    if (block_rotamer_table_set[block_type_index] == -1) return;
    if (rotamer_index == -1) {
      // This should never happen
      return;
    }

    for (int ii = 0; ii < block_n_dihedrals[block_type_index]; ii++) {
      auto dih_uaids = block_dih_uaids[block_type_index][ii];
      bool fail = false;
      for (int jj = 0; jj < DIH_N_ATOMS; jj++) {
        UnresolvedAtomID<Int> uaid = dih_uaids[jj];

        if (uaid.atom_id == -1 && uaid.conn_id == -1) {  // Dihedral undefined
          fail = true;
          break;
        }

        dihedral_atom_inds[rotamer_index][ii][jj] =
            resolve_rotamer_atom_from_uaid(
                uaid,
                rotamer_index,
                block_index,
                pose_index,
                rot_coord_offset,
                first_rot_for_block,
                first_rot_block_type,
                pose_stack_inter_block_connections,
                block_type_atom_downstream_of_conn);
        if (dihedral_atom_inds[rotamer_index][ii][jj] == -1) {
          // UAID resolution failed
          fail = true;
          break;
        }
      }
      if (fail) {  // if the dihedral resolution failed, let's fill the cached
                   // value with -1s since we might have partially filled it
                   // above
        dihedral_atom_inds[rotamer_index][ii] =
            tmol::portable::Vec<Int, 4>::Constant(-1);
      }

      // a terminus has no phi or no psi, so a default stands in for it. It is
      // a torsion like any other and must be negated for a mirrored residue,
      // or the two forms read different points of the same table.
      const Real mirror = block_is_mirrored[block_type_index] ? -1.0 : 1.0;
      const Real PHI_DEFAULT = mirror * -60.0 * M_PI / 180;
      const Real PSI_DEFAULT = mirror * 60.0 * M_PI / 180;

      Real dih_default = (ii == 0)   ? PHI_DEFAULT
                         : (ii == 1) ? PSI_DEFAULT
                                     : 0.0;

      measure_dihedral_V_dV(
          TensorAccessor<Vec<Real, 3>, 1, D>(rot_coords),
          dihedral_atom_inds[rotamer_index][ii],
          dih_default,
          dihedral_values[rotamer_index][ii],
          dihedral_deriv[rotamer_index][ii]);
    }

    Int rotameric_rottable_assignment, semirotameric_rottable_assignment;
    // Templated on there being 2 backbone dihedrals for canonical aas.
    classify_rotamer_for_block<2>(
        dihedral_values[rotamer_index],
        block_n_rotameric_chi[block_type_index],
        block_rotamer_index_to_table_index[block_type_index],
        rotameric_rotind2tableind,
        semirotameric_rotind2tableind,
        rotameric_rottable_assignment,
        semirotameric_rottable_assignment);

    if (block_rotameric_index[block_type_index] != -1) {
      Real prob = rotameric_chi_probability_for_block(
          rotameric_neglnprob_tables,
          rotprob_table_sizes,
          rotprob_table_strides,
          rotameric_bb_start,
          rotameric_bb_step,
          rotameric_bb_periodicity,
          block_probability_table_offset[block_type_index],
          block_rotamer_table_set[block_type_index],
          dihedral_values[rotamer_index],
          rotameric_rottable_assignment,
          dterm_dtor_xyz[rotamer_index],
          dihedral_deriv[rotamer_index]);

      Vec<Int, DIH_N_ATOMS> phi_ats = dihedral_atom_inds[rotamer_index][0];
      Vec<Int, DIH_N_ATOMS> psi_ats = dihedral_atom_inds[rotamer_index][1];
      Real block_weight_0 = dunbrack_score_weight(
          dTdV, 0, pose_index, block_index, rotamer_index);
      for (int j = 0; j < DIH_N_ATOMS; ++j) {
        if (phi_ats[j] != -1)
          emit_dunbrack_derivative<D>(
              dV_dx[0][phi_ats[j]],
              dterm_dtor_xyz[rotamer_index][0].row(j) * block_weight_0);
        if (psi_ats[j] != -1)
          emit_dunbrack_derivative<D>(
              dV_dx[0][psi_ats[j]],
              dterm_dtor_xyz[rotamer_index][1].row(j) * block_weight_0);
      }
    }

    for (int ii = 0; ii < block_n_rotameric_chi[block_type_index]; ii++) {
      // deviation from chi for chis within this block
      auto Erotdev = block_deviation_penalty_for_chi(
          // DB tables
          rotameric_mean_tables,
          rotameric_sdev_tables,
          rotmean_table_sizes,
          rotmean_table_strides,
          rotameric_bb_start,
          rotameric_bb_step,
          rotameric_bb_periodicity,
          // Block type info
          block_rotamer_table_set[block_type_index],
          block_mean_table_offset[block_type_index],
          block_n_chi[block_type_index],
          // Block info
          dihedral_values[rotamer_index],
          ii,
          rotameric_rottable_assignment,
          // Out
          dterm_dtor_xyz[rotamer_index],
          dihedral_deriv[rotamer_index]);

      Real block_weight_1 = dunbrack_score_weight(
          dTdV, 1, pose_index, block_index, rotamer_index);
      Vec<Int, DIH_N_ATOMS> tor0_ats = dihedral_atom_inds[rotamer_index][0];
      Vec<Int, DIH_N_ATOMS> tor1_ats = dihedral_atom_inds[rotamer_index][1];
      Vec<Int, DIH_N_ATOMS> tor2_ats =
          dihedral_atom_inds[rotamer_index][2 + ii];
      for (int j = 0; j < DIH_N_ATOMS; ++j) {
        if (tor0_ats[j] != -1)
          emit_dunbrack_derivative<D>(
              dV_dx[1][tor0_ats[j]],
              dterm_dtor_xyz[rotamer_index][0].row(j) * block_weight_1);
        if (tor1_ats[j] != -1)
          emit_dunbrack_derivative<D>(
              dV_dx[1][tor1_ats[j]],
              dterm_dtor_xyz[rotamer_index][1].row(j) * block_weight_1);
        if (tor2_ats[j] != -1)
          emit_dunbrack_derivative<D>(
              dV_dx[1][tor2_ats[j]],
              dterm_dtor_xyz[rotamer_index][2].row(j) * block_weight_1);
      }
    }

    if (block_semirotameric_index[block_type_index] != -1) {
      auto Esemi = block_semirotameric_energy(
          semirotameric_tables,
          semirot_table_sizes,
          semirot_table_strides,
          semirot_start,
          semirot_step,
          semirot_periodicity,

          block_semirotameric_index[block_type_index],
          block_semirotameric_tableset_offset[block_type_index],
          block_n_chi[block_type_index] + 1,

          dihedral_values[rotamer_index],
          semirotameric_rottable_assignment,

          dterm_dtor_xyz[rotamer_index],
          dihedral_deriv[rotamer_index]);

      Real block_weight_2 = dunbrack_score_weight(
          dTdV, 2, pose_index, block_index, rotamer_index);

      int last = block_n_chi[block_type_index] + 1;
      Vec<Int, DIH_N_ATOMS> tor0_ats = dihedral_atom_inds[rotamer_index][0];
      Vec<Int, DIH_N_ATOMS> tor1_ats = dihedral_atom_inds[rotamer_index][1];
      Vec<Int, DIH_N_ATOMS> tor2_ats = dihedral_atom_inds[rotamer_index][last];
      for (int j = 0; j < DIH_N_ATOMS; ++j) {
        if (tor0_ats[j] != -1)
          emit_dunbrack_derivative<D>(
              dV_dx[2][tor0_ats[j]],
              dterm_dtor_xyz[rotamer_index][0].row(j) * block_weight_2);
        if (tor1_ats[j] != -1)
          emit_dunbrack_derivative<D>(
              dV_dx[2][tor1_ats[j]],
              dterm_dtor_xyz[rotamer_index][1].row(j) * block_weight_2);
        if (tor2_ats[j] != -1)
          emit_dunbrack_derivative<D>(
              dV_dx[2][tor2_ats[j]],
              dterm_dtor_xyz[rotamer_index][2].row(j) * block_weight_2);
      }
    }
  });

  if constexpr (OutRank == 2)
    DeviceDispatch<D>::template forall<launch_t>(mgr, n_rots, func);
  else
    DeviceDispatch<D>::template forall_grouped<launch_t>(
        mgr, n_poses, max_n_blocks, func);
}

}  // namespace potentials
}  // namespace dunbrack
}  // namespace score
}  // namespace tmol
