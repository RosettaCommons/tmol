#pragma once

// The state the per-atom grid-scoring kernel reads.
//
// Templated on the view, so the field list is declared once: the host and
// CUDA builds instantiate it with TView, a Metal translation unit -- which
// cannot include the torch tensor layer -- with tmol::portable::MetalView.

#include <tmol/backends/metal/score/common/mps_layout.hh>
#include <tmol/utility/portable/device.hh>
#include <tmol/utility/portable/matrix.hh>

namespace tmol {
namespace score {
namespace gridscore {
namespace potentials {

// One work item per (pose, grid-scored atom).
template <template <typename, int> class View, typename Real, typename Int>
struct GridScoreAtomsCtxT {
  View<tmol::portable::Vec<Real, 3>, 2> coords;
  View<Int, 1> grid_atoms;
  // [table pose, grid atom]; batch pose p reads row p % rows. Type -1 marks a
  // slot this pose's ligand has no atom for.
  View<Int, 2> atom_types;
  View<Real, 2> charges;
  View<Real, 4> atr;
  View<Real, 4> rep;
  View<Real, 4> sol;
  View<Real, 4> elec;
  // Heavy-atom part of the gridded LK-ball total; empty when not gridded.
  View<Real, 4> lk_ball;
  View<Real, 1> origin;
  View<Real, 1> weights;
  View<Real, 2> energy;
  View<tmol::portable::Vec<Real, 3>, 2> gradient;

  template <typename F>
  TMOL_HOST_DEVICE_FUNC void visit_views(F f) {
    f(coords);
    f(grid_atoms);
    f(atom_types);
    f(charges);
    f(atr);
    f(rep);
    f(sol);
    f(elec);
    f(lk_ball);
    f(origin);
    f(weights);
    f(energy);
    f(gradient);
  }

  enum { n_views = 13 };

  // Launch scalars, carried in the metadata buffer.
  int n_items = {};
  int n_grid_atoms = {};
  int n_types = {};
  bool with_lk_ball = {};
  Real voxel_spacing = {};
  Real out_of_bounds_penalty = {};

  template <typename F>
  TMOL_HOST_DEVICE_FUNC void visit_scalars(F f) {
    f(n_items);
    f(n_grid_atoms);
    f(n_types);
    f(with_lk_ball);
    f(voxel_spacing);
    f(out_of_bounds_penalty);
  }
};

}  // namespace potentials
}  // namespace gridscore
}  // namespace score
}  // namespace tmol
