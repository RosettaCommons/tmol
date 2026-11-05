#pragma once
#include <tmol/utility/portable/matrix.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tuple>
#include <tmol/score/common/segmented_scan_workspace.hh>

namespace tmol {
namespace pack {
namespace rotamer {
namespace dunbrack {
using tmol::portable::Vec;

// Immutable views of library, molecular topology and requested sampling.
// Unused legacy ABI arguments remain in f, outside this scientific contract.
template <typename Real, typename Int, Device D>
struct DunbrackSamplingInputs {
  TView<Vec<Real, 3>, 1, D> coords;
  TView<Real, 3, D> rotameric_prob_tables;
  TView<Vec<int64_t, 2>, 1, D> rotprob_table_sizes;
  TView<Vec<int64_t, 2>, 1, D> rotprob_table_strides;
  TView<Real, 3, D> rotameric_mean_tables;
  TView<Real, 3, D> rotameric_sdev_tables;
  TView<Vec<int64_t, 2>, 1, D> rotmean_table_sizes;
  TView<Vec<int64_t, 2>, 1, D> rotmean_table_strides;
  TView<Int, 1, D> rotameric_meansdev_tableset_offsets;
  TView<Vec<Real, 2>, 1, D> rotameric_bb_start;
  TView<Vec<Real, 2>, 1, D> rotameric_bb_step;
  TView<Vec<Real, 2>, 1, D> rotameric_bb_periodicity;
  TView<Vec<Real, 2>, 1, D> rotameric_bb_source_start;
  TView<bool, 1, D> rotameric_bb_is_mirrored;
  TView<int64_t, 1, D> n_rotamers_for_tableset;
  TView<Int, 1, D> n_rotamers_for_tableset_offsets;
  TView<int64_t, 3, D> sorted_rotamer_2_rotamer;
  TView<Int, 1, D> nchi_for_tableset;
  TView<Int, 1, D> ndihe_for_res;
  TView<Int, 1, D> dihedral_offset_for_res;
  TView<Vec<Int, 4>, 1, D> dihedral_atom_inds;
  TView<Int, 2, D> bubl_and_rottable_set_for_buildable_restype;
  TView<Int, 2, D> chi_expansion_for_buildable_restype;
  TView<Real, 3, D> non_dunbrack_expansion_for_buildable_restype;
  TView<Int, 2, D> non_dunbrack_expansion_counts_for_buildable_restype;
  TView<Real, 1, D> prob_cumsum_limit_for_buildable_restype;
  TView<Int, 1, D> nchi_for_buildable_restype;
};

template <typename Real, typename Int, Device D>
struct DunbrackSamplingPlan {
  TView<Real, 1, D> backbone_dihedrals;
  TView<Int, 1, D> n_expansions_for_brt;
  TView<Int, 2, D> expansion_dim_prods_for_brt;
  TView<Int, 1, D> n_rotamers_to_build_per_brt;
};

// Possible counts/offsets are prepared from host library metadata first.
// Every other scratch element is overwritten by prepare_sampling_plan.
template <typename Real, typename Int, Device D>
struct DunbrackSamplingScratch {
  TView<Int, 1, D> n_possible_rotamers_per_brt;
  TView<Int, 1, D> possible_rotamer_offset_for_brt;
  TView<Int, 1, D> brt_for_possible_rotamer;
  TView<Int, 1, D> boundaries;
  TView<Real, 1, D> rotamer_probability;
  TView<Real, 1, D> probability_prefix;
  TView<Int, 1, D> keep;
  TView<Int, 1, D> count_prefix;
  score::common::SegmentedScanWorkspace<Real, D> probability_workspace;
  score::common::SegmentedScanWorkspace<Int, D> count_workspace;
};

template <
    template <tmol::Device> class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    template <typename, size_t, tmol::Device> class Pack,
    typename Context>
// One canonical sampling implementation; execution primitives, buffer ownership
// and context are supplied by the frontend. The eager f entry returns dynamic
// output sizes after checked prefix scans synchronize their totals to the host.
struct NativeDunbrackChiSampler {
  using Inputs = DunbrackSamplingInputs<Real, Int, D>;
  using Plan = DunbrackSamplingPlan<Real, Int, D>;
  using Scratch = DunbrackSamplingScratch<Real, Int, D>;

  // No allocation or host synchronization. Frontends own all buffers and
  // execution context scratch, including nonsegmented scan workspace.
  static void prepare_sampling_plan(
      Context& mgr, Inputs const& in, Plan const& plan, Scratch const& scratch);
  static void emit_samples(
      Context& mgr,
      Inputs const& in,
      Plan const& plan,
      TView<Int, 1, D> offsets,
      TView<Int, 1, D> brt,
      TView<Int, 1, D> boundaries,
      TView<Real, 2, D> chi);

  static TMOL_HOST_DEVICE_FUNC void backbone_lookup_coordinate(
      Real dihedral,
      Real start,
      Real source_start,
      Real step,
      Real period,
      Real& coordinate,
      Int& bin,
      bool mirrored,
      Int axis);

  static TMOL_HOST_DEVICE_FUNC Int checked_count_product(Int a, Int b);

  static Int checked_count_offsets(
      Context& mgr, TView<Int, 1, D> counts, TView<Int, 1, D> offsets);

  // Dynamic output allocation barrier: identical count overflow and budget
  // validation on the host and in eager framework execution.
  static Int checked_sample_offsets(
      Context& mgr,
      TView<Int, 1, D> counts,
      TView<Int, 1, D> offsets,
      int64_t max_samples_per_restype);

  static auto
  f(Context& mgr,
    TView<Vec<Real, 3>, 1, D> coords,

    TView<Real, 3, D> rotameric_prob_tables,
    TView<Vec<int64_t, 2>, 1, D> rotprob_table_sizes,
    TView<Vec<int64_t, 2>, 1, D> rotprob_table_strides,
    TView<Real, 3, D> rotameric_mean_tables,
    TView<Real, 3, D> rotameric_sdev_tables,
    TView<Vec<int64_t, 2>, 1, D> rotmean_table_sizes,
    TView<Vec<int64_t, 2>, 1, D> rotmean_table_strides,
    TView<Int, 1, D> rotameric_meansdev_tableset_offsets,
    TView<Vec<Real, 2>, 1, D> rotameric_bb_start,        // ntable-set entries
    TView<Vec<Real, 2>, 1, D> rotameric_bb_step,         // ntable-set entries
    TView<Vec<Real, 2>, 1, D> rotameric_bb_periodicity,  // ntable-set entries
    TView<Vec<Real, 2>, 1, D> rotameric_bb_source_start,
    TView<bool, 1, D> rotameric_bb_is_mirrored,
    TView<Real, 4, D> /*semirotameric_tables*/,              // n-semirot-tabset
    TView<Vec<int64_t, 3>, 1, D> /*semirot_table_sizes*/,    // n-semirot-tabset
    TView<Vec<int64_t, 3>, 1, D> /*semirot_table_strides*/,  // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> /*semirot_start*/,             // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> /*semirot_step*/,              // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> /*semirot_periodicity*/,       // n-semirot-tabset
    TView<Int, 1, D> rotameric_rotind2tableind,
    TView<Int, 1, D> semirotameric_rotind2tableind,
    TView<Int, 1, D> all_chi_rotind2tableind,
    TView<Int, 1, D> all_chi_rotind2tableind_offsets,

    TView<int64_t, 1, D> n_rotamers_for_tableset,
    TView<Int, 1, D> n_rotamers_for_tableset_offsets,
    TView<int64_t, 3, D> sorted_rotamer_2_rotamer,
    TView<Int, 1, D> nchi_for_tableset,
    TView<Int, 2, D> rotwells,

    TView<Int, 1, D> ndihe_for_res,               // nbubl x 1
    TView<Int, 1, D> dihedral_offset_for_res,     // nbubl x 1
    TView<Vec<Int, 4>, 1, D> dihedral_atom_inds,  // ndihe x 4

    TView<Int, 2, D>
        bubl_and_rottable_set_for_buildable_restype,  // n-buildable-restypes
                                                      // x 2
    TView<Int, 2, D> chi_expansion_for_buildable_restype,
    TView<Real, 3, D> non_dunbrack_expansion_for_buildable_restype,
    TView<Int, 2, D> non_dunbrack_expansion_counts_for_buildable_restype,
    TView<Real, 1, D> prob_cumsum_limit_for_buildable_restype,
    TView<Int, 1, D> nchi_for_buildable_restype,  // inc. hydroxyl chi, e.g.
    int64_t max_samples_per_restype

    )
      -> std::tuple<
          Pack<Int, 1, D>,
          Pack<Int, 1, D>,
          Pack<Int, 1, D>,
          Pack<Real, 2, D> >;

  static void compute_backbone_dihedrals(
      Context& mgr,
      TView<Vec<Real, 3>, 1, D> coords,
      TView<Int, 1, D> ndihe_for_res,
      TView<Int, 1, D> dihedral_offset_for_res,
      TView<Vec<Int, 4>, 1, D> dihedral_atom_inds,
      TView<Real, 1, D> backbone_dihedrals);

  static void determine_n_possible_rots(
      Context& mgr,
      TView<Int, 2, D> bubl_and_rottable_set_for_buildable_restype,
      TView<int64_t, 1, D> n_rotamers_for_tableset,
      TView<Int, 1, D> n_possible_rotamers_per_brt);

  static void fill_in_brt_for_possrots(
      Context& mgr,
      TView<Int, 1, D> possible_rotamer_offset_for_brt,
      TView<Int, 1, D> brt_for_possible_rotamer);

  static void interpolate_probabilities_for_possible_rotamers(
      Context& mgr,
      TView<Real, 3, D> rotameric_prob_tables,
      TView<Vec<int64_t, 2>, 1, D> rotprob_table_sizes,
      TView<Vec<int64_t, 2>, 1, D> rotprob_table_strides,
      TView<Vec<Real, 2>, 1, D> rotameric_bb_start,
      TView<Vec<Real, 2>, 1, D> rotameric_bb_step,
      TView<Vec<Real, 2>, 1, D> rotameric_bb_periodicity,
      TView<Vec<Real, 2>, 1, D> rotameric_bb_source_start,
      TView<bool, 1, D> rotameric_bb_is_mirrored,
      TView<Int, 1, D> n_rotamers_for_tableset_offsets,
      TView<int64_t, 3, D> sorted_rotamer_2_rotamer,
      TView<Int, 2, D> bubl_and_rottable_set_for_buildable_restype,
      TView<Int, 1, D> brt_for_possible_rotamer,
      TView<Int, 1, D> possible_rotamer_offset_for_brt,
      TView<Real, 1, D> backbone_dihedrals,
      TView<Real, 1, D> rotamer_probability);

  static void determine_n_base_rotamers_to_build(
      Context& mgr,
      TView<Real, 1, D> prob_cumsum_limit_for_buildable_restype,
      TView<Int, 1, D> n_possible_rotamers_per_brt,
      TView<Int, 1, D> brt_for_possible_rotamer,
      TView<Int, 1, D> possible_rotamer_offset_for_brt,
      TView<Real, 1, D> rotamer_probability,
      TView<Int, 1, D> n_rotamers_to_build_per_brt);

  static void determine_n_base_rotamers_to_build_into(
      Context& mgr,
      TView<Real, 1, D> prob_cumsum_limit_for_buildable_restype,
      TView<Int, 1, D> n_possible_rotamers_per_brt,
      TView<Int, 1, D> brt_for_possible_rotamer,
      TView<Int, 1, D> possible_rotamer_offset_for_brt,
      TView<Real, 1, D> rotamer_probability,
      TView<Int, 1, D> n_rotamers_to_build_per_brt,
      TView<Real, 1, D> rotamer_probability_cumsum,
      TView<Int, 1, D> build_possible_rotamer,
      TView<Int, 1, D> count_rotamers_to_build,
      score::common::SegmentedScanWorkspace<Real, D> probability_workspace,
      score::common::SegmentedScanWorkspace<Int, D> count_workspace);

  static Int count_expanded_rotamers(
      Context& mgr,
      TView<Int, 1, D> nchi_for_buildable_restype,
      TView<Int, 2, D> bubl_and_rottable_set_for_buildable_restype,
      TView<Int, 1, D> nchi_for_tableset,
      TView<Int, 2, D> chi_expansion_for_buildable_restype,
      TView<Int, 2, D> non_dunbrack_expansion_counts_for_buildable_restype,
      TView<Int, 1, D> n_expansions_for_brt,
      TView<Int, 2, D> expansion_dim_prods_for_brt,
      TView<Int, 1, D> n_rotamers_to_build_per_brt,
      TView<Int, 1, D> n_rotamers_to_build_per_brt_offsets);

  static void count_expanded_rotamers_into(
      Context& mgr,
      TView<Int, 1, D> nchi_for_buildable_restype,
      TView<Int, 2, D> bubl_and_rottable_set_for_buildable_restype,
      TView<Int, 1, D> nchi_for_tableset,
      TView<Int, 2, D> chi_expansion_for_buildable_restype,
      TView<Int, 2, D> non_dunbrack_expansion_counts_for_buildable_restype,
      TView<Int, 1, D> n_expansions_for_brt,
      TView<Int, 2, D> expansion_dim_prods_for_brt,
      TView<Int, 1, D> n_rotamers_to_build_per_brt);

  static void map_from_rotamer_index_to_brt(
      Context& mgr,
      TView<Int, 1, D> n_rotamers_to_build_per_brt_offsets,
      TView<Int, 1, D> brt_for_rotamer);

  static void map_from_rotamer_index_to_brt_into(
      Context& mgr,
      TView<Int, 1, D> n_rotamers_to_build_per_brt_offsets,
      TView<Int, 1, D> brt_for_rotamer,
      TView<Int, 1, D> brt_for_rotamer_start);

  static void sample_chi_for_rotamers(
      Context& mgr,
      TView<Real, 3, D> rotameric_mean_tables,
      TView<Real, 3, D> rotameric_sdev_tables,
      TView<Vec<int64_t, 2>, 1, D> rotmean_table_sizes,
      TView<Vec<int64_t, 2>, 1, D> rotmean_table_strides,
      TView<Int, 1, D> rotameric_meansdev_tableset_offsets,
      TView<Vec<Real, 2>, 1, D> rotameric_bb_start,
      TView<Vec<Real, 2>, 1, D> rotameric_bb_step,
      TView<Vec<Real, 2>, 1, D> rotameric_bb_periodicity,
      TView<Vec<Real, 2>, 1, D> rotameric_bb_source_start,
      TView<bool, 1, D> rotameric_bb_is_mirrored,

      TView<int64_t, 3, D> sorted_rotamer_2_rotamer,
      TView<Int, 1, D> nchi_for_tableset,

      TView<Int, 2, D> bubl_and_rottable_set_for_buildable_restype,
      TView<Int, 2, D> chi_expansion_for_buildable_restype,
      TView<Real, 3, D> non_dunbrack_expansion_for_buildable_restype,
      TView<Int, 1, D> nchi_for_buildable_restype,

      TView<Real, 1, D> backbone_dihedrals,

      TView<Int, 1, D> n_rotamers_to_build_per_brt_offsets,
      TView<Int, 1, D> brt_for_rotamer,
      TView<Int, 1, D> n_expansions_for_brt,
      TView<Int, 1, D> n_rotamers_for_tableset_offsets,

      TView<Int, 2, D> expansion_dim_prods_for_brt,
      TView<Real, 2, D> chi_for_rotamers);
};
}  // namespace dunbrack
}  // namespace rotamer
}  // namespace pack
}  // namespace tmol
