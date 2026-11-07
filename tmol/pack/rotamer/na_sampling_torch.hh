#pragma once
#include <tmol/utility/tensor/native.hh>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
namespace tmol::pack::rotamer {
template <template <tmol::Device> class Ops, tmol::Device D>
struct NaSamplingTorchDispatch {
  using Tensor = tmol::native::Tensor;
  static Tensor pucker(
      ContextManager&,
      Tensor coords,
      Tensor rings,
      Tensor valid,
      double temperature);
  // inputs: types, base, pucker, n_combos, active, chi_atom, proton,
  // well_syn, means, sdev, steps. Float32 parameter/sample convention.
  static Tensor counts(
      ContextManager&,
      std::vector<Tensor> inputs,
      bool sample_syn,
      int64_t bases_per_polymer,
      double max_syn_well,
      double syn_mean);
  static std::tuple<Tensor, Tensor, Tensor, Tensor> fill(
      ContextManager&,
      std::vector<Tensor> inputs,
      Tensor counts,
      int64_t n_rotamers,
      bool sample_syn,
      int64_t bases_per_polymer,
      double max_syn_well,
      double syn_mean);
};
}  // namespace tmol::pack::rotamer
