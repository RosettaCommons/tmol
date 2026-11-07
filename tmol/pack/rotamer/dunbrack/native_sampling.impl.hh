#pragma once
#include "native_sampling.hh"
#include <tmol/utility/portable/reduce_ops.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/utility/portable/math.hh>
#include <tmol/numeric/bspline_compiled/bspline_kernel.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace tmol {
namespace pack {
namespace rotamer {
namespace dunbrack {

template <typename... Message>
inline void sampling_require(bool valid, Message const&... message) {
  if (!valid) {
    std::ostringstream text;
    (text << ... << message);
    throw std::runtime_error(text.str());
  }
}

template <
    template <tmol::Device> class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    template <typename, size_t, tmol::Device> class Pack,
    typename Context>
void NativeDunbrackChiSampler<Dispatch, D, Real, Int, Pack, Context>::
    prepare_sampling_plan(
        Context& mgr,
        Inputs const& in,
        Plan const& plan,
        Scratch const& scratch) {
  sampling_require(
      in.rotameric_bb_is_mirrored.size(0) == in.rotameric_bb_start.size(0),
      "Dunbrack reflection metadata must cover every table set");
  sampling_require(
      in.rotameric_bb_source_start.size(0) == in.rotameric_bb_start.size(0),
      "Dunbrack source origins must cover every table set");
  compute_backbone_dihedrals(
      mgr,
      in.coords,
      in.ndihe_for_res,
      in.dihedral_offset_for_res,
      in.dihedral_atom_inds,
      plan.backbone_dihedrals);
  map_from_rotamer_index_to_brt_into(
      mgr,
      scratch.possible_rotamer_offset_for_brt,
      scratch.brt_for_possible_rotamer,
      scratch.boundaries);
  interpolate_probabilities_for_possible_rotamers(
      mgr,
      in.rotameric_prob_tables,
      in.rotprob_table_sizes,
      in.rotprob_table_strides,
      in.rotameric_bb_start,
      in.rotameric_bb_step,
      in.rotameric_bb_periodicity,
      in.rotameric_bb_source_start,
      in.rotameric_bb_is_mirrored,
      in.n_rotamers_for_tableset_offsets,
      in.sorted_rotamer_2_rotamer,
      in.bubl_and_rottable_set_for_buildable_restype,
      scratch.brt_for_possible_rotamer,
      scratch.possible_rotamer_offset_for_brt,
      plan.backbone_dihedrals,
      scratch.rotamer_probability);
  determine_n_base_rotamers_to_build_into(
      mgr,
      in.prob_cumsum_limit_for_buildable_restype,
      scratch.n_possible_rotamers_per_brt,
      scratch.brt_for_possible_rotamer,
      scratch.possible_rotamer_offset_for_brt,
      scratch.rotamer_probability,
      plan.n_rotamers_to_build_per_brt,
      scratch.probability_prefix,
      scratch.keep,
      scratch.count_prefix,
      scratch.probability_workspace,
      scratch.count_workspace);
  count_expanded_rotamers_into(
      mgr,
      in.nchi_for_buildable_restype,
      in.bubl_and_rottable_set_for_buildable_restype,
      in.nchi_for_tableset,
      in.chi_expansion_for_buildable_restype,
      in.non_dunbrack_expansion_counts_for_buildable_restype,
      plan.n_expansions_for_brt,
      plan.expansion_dim_prods_for_brt,
      plan.n_rotamers_to_build_per_brt);
}

template <
    template <tmol::Device> class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    template <typename, size_t, tmol::Device> class Pack,
    typename Context>
void NativeDunbrackChiSampler<Dispatch, D, Real, Int, Pack, Context>::
    emit_samples(
        Context& mgr,
        Inputs const& in,
        Plan const& plan,
        TView<Int, 1, D> offsets,
        TView<Int, 1, D> brt,
        TView<Int, 1, D> boundaries,
        TView<Real, 2, D> chi) {
  LAUNCH_BOX_32;
  map_from_rotamer_index_to_brt_into(mgr, offsets, brt, boundaries);
  int const width = chi.size(1);
  Dispatch<D>::template forall<launch_t>(
      mgr, chi.size(0), [=] TMOL_DEVICE_FUNC(int r) {
        for (int c = 0; c < width; ++c) chi[r][c] = Real(0);
      });
  sample_chi_for_rotamers(
      mgr,
      in.rotameric_mean_tables,
      in.rotameric_sdev_tables,
      in.rotmean_table_sizes,
      in.rotmean_table_strides,
      in.rotameric_meansdev_tableset_offsets,
      in.rotameric_bb_start,
      in.rotameric_bb_step,
      in.rotameric_bb_periodicity,
      in.rotameric_bb_source_start,
      in.rotameric_bb_is_mirrored,

      in.sorted_rotamer_2_rotamer,
      in.nchi_for_tableset,

      in.bubl_and_rottable_set_for_buildable_restype,
      in.chi_expansion_for_buildable_restype,
      in.non_dunbrack_expansion_for_buildable_restype,
      in.nchi_for_buildable_restype,

      plan.backbone_dihedrals,

      offsets,
      brt,
      plan.n_expansions_for_brt,
      in.n_rotamers_for_tableset_offsets,

      plan.expansion_dim_prods_for_brt,
      chi);
}

template <
    template <tmol::Device> class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    template <typename, size_t, tmol::Device> class Pack,
    typename Context>
void NativeDunbrackChiSampler<Dispatch, D, Real, Int, Pack, Context>::
    compute_backbone_dihedrals(
        Context& mgr,
        TView<Vec<Real, 3>, 1, D> coords,
        TView<Int, 1, D> ndihe_for_res,
        TView<Int, 1, D> dihedral_offset_for_res,
        TView<Vec<Int, 4>, 1, D> dihedral_atom_inds,
        TView<Real, 1, D> backbone_dihedrals) {
  LAUNCH_BOX_32;
  auto compute_dihedrals = [=] TMOL_DEVICE_FUNC(int bubl) {
    for (int dihe_ind = 0; dihe_ind < ndihe_for_res[bubl]; ++dihe_ind) {
      int i = dihedral_offset_for_res[bubl] + dihe_ind;
      Int at0 = dihedral_atom_inds[i][0];
      Int at1 = dihedral_atom_inds[i][1];
      Int at2 = dihedral_atom_inds[i][2];
      Int at3 = dihedral_atom_inds[i][3];
      Real dihe = std::numeric_limits<Real>::quiet_NaN();
      if (at0 >= 0 && at1 >= 0 && at2 >= 0 && at3 >= 0) {
        dihe = score::common::dihedral_angle<Real>::V(
            coords[at0], coords[at1], coords[at2], coords[at3]);
      }
      backbone_dihedrals[i] = dihe;
    }
  };

  Dispatch<D>::template forall<launch_t>(
      mgr, dihedral_offset_for_res.size(0), compute_dihedrals);
}

template <
    template <tmol::Device> class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    template <typename, size_t, tmol::Device> class Pack,
    typename Context>
TMOL_HOST_DEVICE_FUNC void
NativeDunbrackChiSampler<Dispatch, D, Real, Int, Pack, Context>::
    backbone_lookup_coordinate(
        Real dihedral,
        Real start,
        Real source_start,
        Real step,
        Real period,
        Real& coordinate,
        Int& bin,
        bool mirrored,
        Int axis) {
  // Missing terminal torsions use the source library's neutral angles.
  // A reflected table resolves its own defaults, even when L and D types
  // are both considered at one input residue.
  if (dihedral != dihedral) {
    dihedral = axis == 0 ? Real(-M_PI / 3.0) : Real(M_PI / 3.0);
  } else if (mirrored) {
    dihedral = -dihedral;
  }
  Real wrapped = dihedral - (mirrored ? source_start : start);
  while (wrapped < 0) wrapped += period;
  while (wrapped >= period) wrapped -= period;
  coordinate = wrapped / step;
  // Division can round a value just below the endpoint up to the bin count.
  Int const n_bins = Int(period / step + Real(0.5));
  if (coordinate >= n_bins) coordinate = 0;
  bin = Int(coordinate);
  if (mirrored) {
    // The generated grid reflects point k to (origin - k) modulo n_bins.
    // Select the source cell first: independently flooring the reflected
    // coordinate chooses the adjacent cell, and ceil is unstable at grid
    // boundaries because the two coordinate calculations round separately.
    Int const origin = Int(std::round(-(source_start + start) / step));
    coordinate = Real(origin) - coordinate;
    while (coordinate < 0) coordinate += n_bins;
    while (coordinate >= n_bins) coordinate -= n_bins;
    bin = (origin - bin) % n_bins;
    if (bin < 0) bin += n_bins;
  }
}

template <
    template <tmol::Device> class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    template <typename, size_t, tmol::Device> class Pack,
    typename Context>
TMOL_HOST_DEVICE_FUNC Int
NativeDunbrackChiSampler<Dispatch, D, Real, Int, Pack, Context>::
    checked_count_product(Int a, Int b) {
  if (a < 0 || b < 0) return -1;
  if constexpr (sizeof(Int) < sizeof(int64_t)) {
    int64_t const product = int64_t(a) * int64_t(b);
    return product > std::numeric_limits<Int>::max() ? Int(-1) : Int(product);
  } else {
    if (b != 0 && a > std::numeric_limits<Int>::max() / b) return -1;
    return a * b;
  }
}

template <
    template <tmol::Device> class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    template <typename, size_t, tmol::Device> class Pack,
    typename Context>
Int NativeDunbrackChiSampler<Dispatch, D, Real, Int, Pack, Context>::
    checked_count_offsets(
        Context& mgr, TView<Int, 1, D> counts, TView<Int, 1, D> offsets) {
  sampling_require(
      counts.size(0) == offsets.size(0),
      "Dunbrack count/offset shape mismatch");
  if (counts.size(0) == 0) return 0;
  // The serial primitive preserves the caller's exclusive-scan identity;
  // this checked count scan always uses zero, independent of buffer contents.
  if constexpr (D == Device::CPU) offsets[0] = 0;
  auto add = [] TMOL_DEVICE_FUNC(Int a, Int b) -> Int {
    if (a < 0 || b < 0 || a > std::numeric_limits<Int>::max() - b) return -1;
    return a + b;
  };
  Int const total = Dispatch<D>::template scan_and_return_total<
      tmol::portable::scan_type::exclusive>(
      mgr, counts.data(), offsets.data(), counts.size(0), add);
  sampling_require(
      total >= 0,
      "Dunbrack sampling count exceeds index capacity ",
      std::numeric_limits<Int>::max(),
      " or contains a negative count");
  return total;
}

template <
    template <tmol::Device> class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    template <typename, size_t, tmol::Device> class Pack,
    typename Context>
Int NativeDunbrackChiSampler<Dispatch, D, Real, Int, Pack, Context>::
    checked_sample_offsets(
        Context& mgr,
        TView<Int, 1, D> counts,
        TView<Int, 1, D> offsets,
        int64_t max_samples_per_restype) {
  Int const n_rotamers = checked_count_offsets(mgr, counts, offsets);
  if (max_samples_per_restype > 0 && counts.size(0) != 0) {
    Int const maximum = Dispatch<D>::reduce(
        mgr, counts.data(), counts.size(0), portable::maximum_t<Int>());
    sampling_require(
        maximum <= max_samples_per_restype,
        "Sampling budget ",
        max_samples_per_restype,
        " cannot fit the required ",
        maximum,
        " Dunbrack library/extra-chi states");
  }
  return n_rotamers;
}

template <
    template <tmol::Device> class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    template <typename, size_t, tmol::Device> class Pack,
    typename Context>
auto NativeDunbrackChiSampler<Dispatch, D, Real, Int, Pack, Context>::f(
    Context& mgr,
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
        Pack<Real, 2, D> > {
  Int const n_brt = bubl_and_rottable_set_for_buildable_restype.size(0);
  if (n_brt == 0) {
    auto empty_i = Pack<Int, 1, D>::zeros(0);
    return {empty_i, empty_i, empty_i, Pack<Real, 2, D>::zeros({0, 0})};
  }
  Inputs in{
      coords,
      rotameric_prob_tables,
      rotprob_table_sizes,
      rotprob_table_strides,
      rotameric_mean_tables,
      rotameric_sdev_tables,
      rotmean_table_sizes,
      rotmean_table_strides,
      rotameric_meansdev_tableset_offsets,
      rotameric_bb_start,
      rotameric_bb_step,
      rotameric_bb_periodicity,
      rotameric_bb_source_start,
      rotameric_bb_is_mirrored,
      n_rotamers_for_tableset,
      n_rotamers_for_tableset_offsets,
      sorted_rotamer_2_rotamer,
      nchi_for_tableset,
      ndihe_for_res,
      dihedral_offset_for_res,
      dihedral_atom_inds,
      bubl_and_rottable_set_for_buildable_restype,
      chi_expansion_for_buildable_restype,
      non_dunbrack_expansion_for_buildable_restype,
      non_dunbrack_expansion_counts_for_buildable_restype,
      prob_cumsum_limit_for_buildable_restype,
      nchi_for_buildable_restype};
  auto possible_counts = Pack<Int, 1, D>::empty(n_brt);
  auto possible_offsets = Pack<Int, 1, D>::empty(n_brt);
  determine_n_possible_rots(
      mgr,
      in.bubl_and_rottable_set_for_buildable_restype,
      in.n_rotamers_for_tableset,
      possible_counts.view);
  Int const n_possible =
      checked_count_offsets(mgr, possible_counts.view, possible_offsets.view);
  Int const max_n_chi = Dispatch<D>::reduce(
      mgr,
      nchi_for_buildable_restype.data(),
      n_brt,
      portable::maximum_t<Int>());
  auto backbone = Pack<Real, 1, D>::empty(ndihe_for_res.size(0) * 2);
  auto expansions = Pack<Int, 1, D>::empty(n_brt);
  auto products = Pack<Int, 2, D>::empty({n_brt, max_n_chi});
  auto counts = Pack<Int, 1, D>::empty(n_brt);
  Plan plan{backbone.view, expansions.view, products.view, counts.view};
  {
    auto possible_brt = Pack<Int, 1, D>::empty(n_possible);
    auto possible_boundary = Pack<Int, 1, D>::empty(n_possible);
    auto probability = Pack<Real, 1, D>::empty(n_possible);
    auto probability_prefix = Pack<Real, 1, D>::empty(n_possible);
    auto keep = Pack<Int, 1, D>::empty(n_possible);
    auto count_prefix = Pack<Int, 1, D>::empty(n_possible);
    score::common::OwnedSegmentedScanWorkspace<Real, D, Pack> probability_ws(
        n_possible, n_brt);
    score::common::OwnedSegmentedScanWorkspace<Int, D, Pack> count_ws(
        n_possible, n_brt);
    Scratch scratch{
        possible_counts.view,
        possible_offsets.view,
        possible_brt.view,
        possible_boundary.view,
        probability.view,
        probability_prefix.view,
        keep.view,
        count_prefix.view,
        probability_ws.views(),
        count_ws.views()};
    prepare_sampling_plan(mgr, in, plan, scratch);
  }
  auto offsets = Pack<Int, 1, D>::empty(n_brt);
  Int const n_rotamers = checked_sample_offsets(
      mgr, counts.view, offsets.view, max_samples_per_restype);
  auto brt = Pack<Int, 1, D>::empty(n_rotamers);
  auto boundaries = Pack<Int, 1, D>::empty(n_rotamers);
  auto chi = Pack<Real, 2, D>::empty({n_rotamers, max_n_chi});
  emit_samples(
      mgr, in, plan, offsets.view, brt.view, boundaries.view, chi.view);
  return {counts, offsets, brt, chi};
}

template <
    template <tmol::Device> class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    template <typename, size_t, tmol::Device> class Pack,
    typename Context>
void NativeDunbrackChiSampler<Dispatch, D, Real, Int, Pack, Context>::
    determine_n_possible_rots(
        Context& mgr,
        TView<Int, 2, D> bubl_and_rottable_set_for_buildable_restype,
        TView<int64_t, 1, D> n_rotamers_for_tableset,
        TView<Int, 1, D> n_possible_rotamers_per_brt) {
  LAUNCH_BOX_32;

  Int const n_brt = bubl_and_rottable_set_for_buildable_restype.size(0);
  assert(n_possible_rotamers_per_brt.size(0) == n_brt);
  auto lambda_determine_n_possible_rots = [=] TMOL_DEVICE_FUNC(int brt) {
    Int rottable_set = bubl_and_rottable_set_for_buildable_restype[brt][1];
    // fd  a residue with no rotamer library builds from its own chi samples
    //     alone: one base rotamer, which those samples then expand
    int64_t const count =
        rottable_set < 0 ? 1 : n_rotamers_for_tableset[rottable_set];
    n_possible_rotamers_per_brt[brt] =
        count < 0 || count > std::numeric_limits<Int>::max() ? Int(-1)
                                                             : Int(count);
  };

  Dispatch<D>::template forall<launch_t>(
      mgr, n_brt, lambda_determine_n_possible_rots);
}

template <
    template <tmol::Device> class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    template <typename, size_t, tmol::Device> class Pack,
    typename Context>
void NativeDunbrackChiSampler<Dispatch, D, Real, Int, Pack, Context>::
    fill_in_brt_for_possrots(
        Context& mgr,
        TView<Int, 1, D> possible_rotamer_offset_for_brt,
        TView<Int, 1, D> brt_for_possible_rotamer) {
  map_from_rotamer_index_to_brt(
      mgr, possible_rotamer_offset_for_brt, brt_for_possible_rotamer);
}

template <
    template <tmol::Device> class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    template <typename, size_t, tmol::Device> class Pack,
    typename Context>
void NativeDunbrackChiSampler<Dispatch, D, Real, Int, Pack, Context>::
    interpolate_probabilities_for_possible_rotamers(
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
        TView<Real, 1, D> rotamer_probability) {
  LAUNCH_BOX_32;

  // std::cout << "interpolate_probabilities_for_possible_rotamers" <<
  // std::endl;
  int const n_possible_rotamers = brt_for_possible_rotamer.size(0);

  auto calculate_possible_rotamer_probability = [=] TMOL_DEVICE_FUNC(
                                                    int possible_rotamer) {
    // Compute the probability of the ith possible rotamer
    int const brt = brt_for_possible_rotamer[possible_rotamer];
    int const bbi = bubl_and_rottable_set_for_buildable_restype[brt][0];
    int const table_set = bubl_and_rottable_set_for_buildable_restype[brt][1];
    int const sorted_rotno =
        possible_rotamer - possible_rotamer_offset_for_brt[brt];

    if (table_set < 0) {
      // no library: the single base rotamer is the only one to build
      rotamer_probability[possible_rotamer] = 1.0;
      return;
    }

    Vec<Real, 2> bbdihe;
    Vec<Int, 2> bin_index;
    for (int ii = 0; ii < 2; ++ii) {
      backbone_lookup_coordinate(
          backbone_dihedrals[2 * bbi + ii],
          rotameric_bb_start[table_set][ii],
          rotameric_bb_source_start[table_set][ii],
          rotameric_bb_step[table_set][ii],
          rotameric_bb_periodicity[table_set][ii],
          bbdihe[ii],
          bin_index[ii],
          rotameric_bb_is_mirrored[table_set],
          ii);
    }

    // Look up the index of the rotamer: we know where the rotamer is in
    // sorted order but not which chi values this represents. Also look
    // up the "table index" of this rotamer, which for some AAs
    // (e.g. LYS) is not the same as the rotamer index because some
    // rotamers are too strained to be considered (e.g. g+,g+,g+,g+)
    Int const tableset_offset = n_rotamers_for_tableset_offsets[table_set];
    Int const rot_table_ind =
        sorted_rotamer_2_rotamer[bin_index[0]][bin_index[1]]
                                [tableset_offset + sorted_rotno]
        + tableset_offset;

    // Now we know which rotamer we'll be building: time to look up
    // (interpolate) the rotamer's probability from the
    // rotameric_prob_tables

    TensorAccessor<Real, 2, D> rotprob_slice(
        rotameric_prob_tables.data()
            + rot_table_ind * rotameric_prob_tables.stride(0),
        rotprob_table_sizes[rot_table_ind].data(),
        rotprob_table_strides[rot_table_ind].data());
    auto prob_and_derivs = tmol::numeric::bspline::
        ndspline_kernel<2, 3, D, Real, Int>::interpolate(rotprob_slice, bbdihe);
    rotamer_probability[possible_rotamer] =
        score::common::get<0>(prob_and_derivs);
  };
  Dispatch<D>::template forall<launch_t>(
      mgr, n_possible_rotamers, calculate_possible_rotamer_probability);
}

template <
    template <tmol::Device> class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    template <typename, size_t, tmol::Device> class Pack,
    typename Context>
void NativeDunbrackChiSampler<Dispatch, D, Real, Int, Pack, Context>::
    determine_n_base_rotamers_to_build(
        Context& mgr,
        TView<Real, 1, D> prob_cumsum_limit_for_buildable_restype,
        TView<Int, 1, D> n_possible_rotamers_per_brt,
        TView<Int, 1, D> brt_for_possible_rotamer,
        TView<Int, 1, D> possible_rotamer_offset_for_brt,
        TView<Real, 1, D> rotamer_probability,
        TView<Int, 1, D> n_rotamers_to_build_per_brt) {
  int const count = rotamer_probability.size(0);
  int const segments = possible_rotamer_offset_for_brt.size(0);
  auto probability_prefix = Pack<Real, 1, D>::empty(count);
  auto keep = Pack<Int, 1, D>::empty(count);
  auto count_prefix = Pack<Int, 1, D>::empty(count);
  score::common::OwnedSegmentedScanWorkspace<Real, D, Pack> probability_ws(
      count, segments);
  score::common::OwnedSegmentedScanWorkspace<Int, D, Pack> count_ws(
      count, segments);
  determine_n_base_rotamers_to_build_into(
      mgr,
      prob_cumsum_limit_for_buildable_restype,
      n_possible_rotamers_per_brt,
      brt_for_possible_rotamer,
      possible_rotamer_offset_for_brt,
      rotamer_probability,
      n_rotamers_to_build_per_brt,
      probability_prefix.view,
      keep.view,
      count_prefix.view,
      probability_ws.views(),
      count_ws.views());
}

template <
    template <tmol::Device> class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    template <typename, size_t, tmol::Device> class Pack,
    typename Context>
void NativeDunbrackChiSampler<Dispatch, D, Real, Int, Pack, Context>::
    determine_n_base_rotamers_to_build_into(
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
        score::common::SegmentedScanWorkspace<Int, D> count_workspace) {
  LAUNCH_BOX_32;

  int const n_brt = n_rotamers_to_build_per_brt.size(0);
  int const n_possible_rotamers = rotamer_probability.size(0);
  assert(prob_cumsum_limit_for_buildable_restype.size(0) == n_brt);
  assert(brt_for_possible_rotamer.size(0) == n_possible_rotamers);
  assert(possible_rotamer_offset_for_brt.size(0) == n_brt);

  // OK
  // Now we perform (exclusive) segmented scans on the possible-rotamer
  // probabilities to get the cumulative sums of the probabilities so
  // that we can decide where to put the cutoff
  Dispatch<D>::template segmented_scan_into<
      tmol::portable::scan_type::exclusive>(
      mgr,
      rotamer_probability.data(),
      possible_rotamer_offset_for_brt.data(),
      rotamer_probability_cumsum.data(),
      n_possible_rotamers,
      n_brt,
      tmol::portable::plus_t<Real>(),
      Real(0),
      probability_workspace);

  // And with the cumulative sum, we can now decide which rotamers we will
  // build
  auto decide_on_possible_rotamer = [=] TMOL_DEVICE_FUNC(int possible_rotamer) {
    int const rt = brt_for_possible_rotamer[possible_rotamer];
    int keep = rotamer_probability_cumsum[possible_rotamer]
               <= prob_cumsum_limit_for_buildable_restype[rt];
    build_possible_rotamer[possible_rotamer] = keep;
  };

  Dispatch<D>::template forall<launch_t>(
      mgr, n_possible_rotamers, decide_on_possible_rotamer);

  // *in*clusive segmented scan on the build_possible_rotamer array
  Dispatch<D>::template segmented_scan_into<
      tmol::portable::scan_type::inclusive>(
      mgr,
      build_possible_rotamer.data(),
      possible_rotamer_offset_for_brt.data(),
      count_rotamers_to_build.data(),
      n_possible_rotamers,
      n_brt,
      tmol::portable::plus_t<Int>(),
      Int(0),
      count_workspace);

  auto count_rots_to_build_per_brt = [=] TMOL_DEVICE_FUNC(int brt) {
    Int const offset = possible_rotamer_offset_for_brt[brt];
    Int const npossible = n_possible_rotamers_per_brt[brt];
    Int const last = offset + npossible - 1;
    Int const brt_count = npossible == 0 ? 0 : count_rotamers_to_build[last];
    n_rotamers_to_build_per_brt[brt] = brt_count;
  };

  Dispatch<D>::template forall<launch_t>(
      mgr, n_brt, count_rots_to_build_per_brt);
}

template <
    template <tmol::Device> class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    template <typename, size_t, tmol::Device> class Pack,
    typename Context>
Int NativeDunbrackChiSampler<Dispatch, D, Real, Int, Pack, Context>::
    count_expanded_rotamers(
        Context& mgr,
        TView<Int, 1, D> nchi_for_buildable_restype,
        TView<Int, 2, D> bubl_and_rottable_set_for_buildable_restype,
        TView<Int, 1, D> nchi_for_tableset,
        TView<Int, 2, D> chi_expansion_for_buildable_restype,
        TView<Int, 2, D> non_dunbrack_expansion_counts_for_buildable_restype,
        TView<Int, 1, D> n_expansions_for_brt,
        TView<Int, 2, D> expansion_dim_prods_for_brt,
        TView<Int, 1, D> n_rotamers_to_build_per_brt,
        TView<Int, 1, D> n_rotamers_to_build_per_brt_offsets) {
  count_expanded_rotamers_into(
      mgr,
      nchi_for_buildable_restype,
      bubl_and_rottable_set_for_buildable_restype,
      nchi_for_tableset,
      chi_expansion_for_buildable_restype,
      non_dunbrack_expansion_counts_for_buildable_restype,
      n_expansions_for_brt,
      expansion_dim_prods_for_brt,
      n_rotamers_to_build_per_brt);
  return checked_count_offsets(
      mgr, n_rotamers_to_build_per_brt, n_rotamers_to_build_per_brt_offsets);
}

template <
    template <tmol::Device> class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    template <typename, size_t, tmol::Device> class Pack,
    typename Context>
void NativeDunbrackChiSampler<Dispatch, D, Real, Int, Pack, Context>::
    count_expanded_rotamers_into(
        Context& mgr,
        TView<Int, 1, D> nchi_for_buildable_restype,
        TView<Int, 2, D> bubl_and_rottable_set_for_buildable_restype,
        TView<Int, 1, D> nchi_for_tableset,
        TView<Int, 2, D> chi_expansion_for_buildable_restype,
        TView<Int, 2, D> non_dunbrack_expansion_counts_for_buildable_restype,
        TView<Int, 1, D> n_expansions_for_brt,
        TView<Int, 2, D> expansion_dim_prods_for_brt,
        TView<Int, 1, D> n_rotamers_to_build_per_brt) {
  LAUNCH_BOX_32;

  int const n_brt = nchi_for_buildable_restype.size(0);
  int const max_nchi = expansion_dim_prods_for_brt.size(1);

  assert(bubl_and_rottable_set_for_buildable_restype.size(0) == n_brt);
  assert(chi_expansion_for_buildable_restype.size(0) == n_brt);
  assert(n_expansions_for_brt.size(0) == n_brt);
  assert(expansion_dim_prods_for_brt.size(0) == n_brt);
  assert(n_rotamers_to_build_per_brt.size(0) == n_brt);

  auto count_expansions_for_brt = [=] TMOL_DEVICE_FUNC(int brt) {
    Int const nchi = nchi_for_buildable_restype[brt];
    Int const table_set = bubl_and_rottable_set_for_buildable_restype[brt][1];
    Int const n_dun_chi = table_set < 0 ? 0 : nchi_for_tableset[table_set];
    Int n_expansions = 1;

    Int const n_chi = nchi_for_buildable_restype[brt];
    for (int ii = n_chi - 1; ii >= n_dun_chi; --ii) {
      expansion_dim_prods_for_brt[brt][ii] = n_expansions;
      Int ii_expansion =
          non_dunbrack_expansion_counts_for_buildable_restype[brt][ii];
      if (ii_expansion != 0) {
        n_expansions = checked_count_product(n_expansions, ii_expansion);
      }
    }

    for (int ii = n_dun_chi - 1; ii >= 0; --ii) {
      expansion_dim_prods_for_brt[brt][ii] = n_expansions;
      // for now, only consider +/- 1 standard deviation sampling
      if (chi_expansion_for_buildable_restype[brt][ii]) {
        n_expansions = checked_count_product(n_expansions, Int(3));
      }
    }

    n_expansions_for_brt[brt] = n_expansions;
    n_rotamers_to_build_per_brt[brt] =
        checked_count_product(n_rotamers_to_build_per_brt[brt], n_expansions);
  };

  Dispatch<D>::template forall<launch_t>(mgr, n_brt, count_expansions_for_brt);
}

template <
    template <tmol::Device> class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    template <typename, size_t, tmol::Device> class Pack,
    typename Context>
void NativeDunbrackChiSampler<Dispatch, D, Real, Int, Pack, Context>::
    map_from_rotamer_index_to_brt(
        Context& mgr,
        TView<Int, 1, D> n_rotamers_to_build_per_brt_offsets,
        TView<Int, 1, D> brt_for_rotamer) {
  auto boundary = Pack<Int, 1, D>::empty(brt_for_rotamer.size(0));
  map_from_rotamer_index_to_brt_into(
      mgr, n_rotamers_to_build_per_brt_offsets, brt_for_rotamer, boundary.view);
}

template <
    template <tmol::Device> class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    template <typename, size_t, tmol::Device> class Pack,
    typename Context>
void NativeDunbrackChiSampler<Dispatch, D, Real, Int, Pack, Context>::
    map_from_rotamer_index_to_brt_into(
        Context& mgr,
        TView<Int, 1, D> n_rotamers_to_build_per_brt_offsets,
        TView<Int, 1, D> brt_for_rotamer,
        TView<Int, 1, D> brt_for_rotamer_start) {
  LAUNCH_BOX_32;

  int const n_rotamers = brt_for_rotamer.size(0);
  int const n_brt = n_rotamers_to_build_per_brt_offsets.size(0);

  sampling_require(
      brt_for_rotamer_start.size(0) == n_rotamers,
      "Dunbrack boundary workspace has the wrong size");
  Dispatch<D>::template forall<launch_t>(
      mgr, n_rotamers, [=] TMOL_DEVICE_FUNC(int i) {
        brt_for_rotamer_start[i] = 0;
      });
  auto mark_rot_brt_boundary_beginnings = [=] TMOL_DEVICE_FUNC(int brt) {
    Int const offset = n_rotamers_to_build_per_brt_offsets[brt];
    Int const end = brt + 1 < n_brt
                        ? n_rotamers_to_build_per_brt_offsets[brt + 1]
                        : n_rotamers;
    if (offset < end) brt_for_rotamer_start[offset] = brt;
  };

  Dispatch<D>::template forall<launch_t>(
      mgr, n_brt, mark_rot_brt_boundary_beginnings);

  // Now scan on max and record the restype for each rotamer
  Dispatch<D>::template scan<tmol::portable::scan_type::inclusive>(
      mgr,
      brt_for_rotamer_start.data(),
      brt_for_rotamer.data(),
      n_rotamers,
      tmol::portable::maximum_t<Int>());
}

template <
    template <tmol::Device> class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    template <typename, size_t, tmol::Device> class Pack,
    typename Context>
void NativeDunbrackChiSampler<Dispatch, D, Real, Int, Pack, Context>::
    sample_chi_for_rotamers(
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
        TView<Real, 2, D> chi_for_rotamers) {
  LAUNCH_BOX_32;

  int const n_rotamers = chi_for_rotamers.size(0);
  int const max_n_chi = chi_for_rotamers.size(1);

  // Now we can construct the chi samples.
  // One thread per rotamer
  // each rotamer figures out from its index
  //  -- which restype it's building for
  //  -- its base rotamer index
  //  -- which expanded rotamer for the base rotamer it is

  auto sample_chi_for_rotamer = [=] TMOL_DEVICE_FUNC(int rotamer) {
    int const brt = brt_for_rotamer[rotamer];
    int const res = bubl_and_rottable_set_for_buildable_restype[brt][0];
    int const table_set = bubl_and_rottable_set_for_buildable_restype[brt][1];
    int const expanded_rotamer_for_brt =
        rotamer - n_rotamers_to_build_per_brt_offsets[brt];
    int const n_expansions = n_expansions_for_brt[brt];
    int const base_rotno = expanded_rotamer_for_brt / n_expansions;
    int const expanded_rotno = expanded_rotamer_for_brt % n_expansions;

    // fd  with no rotamer library there is no table to read and no backbone
    //     to read it at: every chi comes from the residue's own samples
    bool const has_library = table_set >= 0;

    Vec<Real, 2> bbdihe;
    Vec<Int, 2> bin_index;
    bbdihe[0] = bbdihe[1] = 0;
    bin_index[0] = bin_index[1] = 0;
    if (has_library) {
      for (int ii = 0; ii < 2; ++ii) {
        backbone_lookup_coordinate(
            backbone_dihedrals[2 * res + ii],
            rotameric_bb_start[table_set][ii],
            rotameric_bb_source_start[table_set][ii],
            rotameric_bb_step[table_set][ii],
            rotameric_bb_periodicity[table_set][ii],
            bbdihe[ii],
            bin_index[ii],
            rotameric_bb_is_mirrored[table_set],
            ii);
      }
    }

    // Look up the index of the rotamer: we know where the rotamer is in
    // sorted order
    // but not which chi values this represents.
    Int const n_dun_chi = has_library ? nchi_for_tableset[table_set] : 0;
    Int const n_chi = nchi_for_buildable_restype[brt];
    Int const rot_table_start =
        has_library ? n_dun_chi
                              * sorted_rotamer_2_rotamer
                                  [bin_index[0]][bin_index[1]]
                                  [n_rotamers_for_tableset_offsets[table_set]
                                   + base_rotno]
                          + rotameric_meansdev_tableset_offsets[table_set]
                    : 0;

    // OK: we have the phi & psi values, and we have the rotamer index
    // Now we can start calculating the chi.

    // How do we do that?
    // For each chi,
    //  if it is a dunbrack chi:
    //    we will look up the mean value, and if
    //    we are expanding this chi, and the expansion
    //    index calls for the standard devation, then
    //    we will also look up the standard devation.
    //    then we will record the perturbed chi value
    //  else
    //    we will look up the non_dunbrack_expansion

    int expanded_rot_remainder = expanded_rotno;
    for (int ii = 0; ii < n_chi; ++ii) {
      int const ii_dim_prods = expansion_dim_prods_for_brt[brt][ii];
      int const ii_expansion = expanded_rot_remainder / ii_dim_prods;
      expanded_rot_remainder = expanded_rot_remainder % ii_dim_prods;

      Real ii_chi;
      if (ii < n_dun_chi) {
        TensorAccessor<Real, 2, D> rotmean_slice(
            rotameric_mean_tables.data()
                + (rot_table_start + ii) * rotameric_mean_tables.stride(0),
            rotmean_table_sizes[rot_table_start + ii].data(),
            rotmean_table_strides[rot_table_start + ii].data());

        auto mean_and_derivs =
            tmol::numeric::bspline::ndspline_kernel<2, 3, D, Real, Int>::
                interpolate(rotmean_slice, bbdihe);
        ii_chi = score::common::get<0>(mean_and_derivs);
        if (chi_expansion_for_buildable_restype[brt][ii]) {
          // OK! we expand this chi; so retrieve the standard deviation
          TensorAccessor<Real, 2, D> rotsdev_slice(
              rotameric_sdev_tables.data()
                  + (rot_table_start + ii) * rotameric_sdev_tables.stride(0),
              rotmean_table_sizes[rot_table_start + ii].data(),
              rotmean_table_strides[rot_table_start + ii].data());
          auto sdev_and_derivs =
              tmol::numeric::bspline::ndspline_kernel<2, 3, D, Real, Int>::
                  interpolate(rotsdev_slice, bbdihe);
          Real sdev = score::common::get<0>(sdev_and_derivs);
          if (ii_expansion == 0) {
            ii_chi -= sdev;
          } else if (ii_expansion == 2) {
            ii_chi += sdev;
          }
        }
      } else {
        ii_chi =
            non_dunbrack_expansion_for_buildable_restype[brt][ii][ii_expansion];
      }
      chi_for_rotamers[rotamer][ii] = ii_chi;
    }
  };

  Dispatch<D>::template forall<launch_t>(
      mgr, n_rotamers, sample_chi_for_rotamer);
}

}  // namespace dunbrack
}  // namespace rotamer
}  // namespace pack
}  // namespace tmol
