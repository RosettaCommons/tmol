// Host setup ABI. Every topology operation is supplied by the shared planner.
#pragma once
#include "host_array_ffi.hh"
#include <tmol/kinematics/compiled/host_pose_kinematics.hh>
#include <tmol/kinematics/compiled/native_kinematics.hh>
#include <tmol/pack/rotamer/dunbrack/host_sampling.hh>
#include <tmol/score/hbond/potentials/native_hbond_bases.hh>
#include <cstdio>

namespace tmol::jax {
// Host allocation/dispatch only; the inverse operation is the runtime kernel.
template <typename Real>
auto host_inverse_kinematics(
    ContextManager& context,
    TView<kinematics::KinCoord<Real>, 1, Device::CPU> coords,
    TView<int32_t, 1, Device::CPU> parent,
    TView<int32_t, 1, Device::CPU> frame_x,
    TView<int32_t, 1, Device::CPU> frame_y,
    TView<int32_t, 1, Device::CPU> frame_z,
    TView<int32_t, 1, Device::CPU> kinds) {
  auto count = coords.size(0);
  for (auto index : {parent, frame_x, frame_y, frame_z, kinds})
    if (index.size(0) != count)
      throw std::invalid_argument(
          "inverse forest and coordinate extents differ");
  for (int32_t i = 0; i < count; ++i) {
    for (auto index : {parent, frame_x, frame_y, frame_z})
      if (index[i] < 0 || index[i] >= count)
        throw std::invalid_argument("inverse forest index is out of bounds");
    if (kinds[i] < 0 || kinds[i] > 2)
      throw std::invalid_argument("invalid inverse forest node type");
  }
  using Transform = kinematics::KinTransform<Real>;
  using Dofs = kinematics::KinDofs<Real>;
  auto scratch = OwnedHostTensor<Transform, 1, Device::CPU>::full(
      {count}, Transform::Zero());
  auto result =
      OwnedHostTensor<Dofs, 1, Device::CPU>::full({count}, Dofs::Zero());
  kinematics::inverse_kinematics_native<
      kinematics::HostPoseOperations,
      Device::CPU,
      Real,
      int32_t>(
      context,
      coords,
      parent,
      frame_x,
      frame_y,
      frame_z,
      kinds,
      scratch.view,
      result.view);
  return result;
}
// Only array ownership/transport is local; layout and budget semantics are
// shared.
inline auto sampling_host_layout(
    ContextManager&,
    TView<int32_t, 2, Device::CPU> types,
    TView<int64_t, 1, Device::CPU> library_counts,
    TView<int32_t, 1, Device::CPU> chi_counts) {
  auto layout = pack::rotamer::dunbrack::prepare_dunbrack_sampling_layout(
      types, library_counts, chi_counts);
  auto dimensions = OwnedHostTensor<int32_t, 1, Device::CPU>::empty({2});
  dimensions.view[0] = layout.n_possible_rotamers;
  dimensions.view[1] = layout.max_n_chi;
  return std::make_tuple(
      layout.possible_counts, layout.possible_offsets, dimensions);
}
inline auto sampling_host_offsets(
    ContextManager& context,
    TView<int32_t, 1, Device::CPU> counts,
    int64_t budget) {
  auto offsets =
      OwnedHostTensor<int32_t, 1, Device::CPU>::empty({counts.size(0)});
  auto total = OwnedHostTensor<int32_t, 1, Device::CPU>::empty({1});
  total.view[0] = pack::rotamer::dunbrack::HostDunbrackSampling<
      float>::checked_sample_offsets(context, counts, offsets.view, budget);
  return std::make_tuple(offsets, total);
}

// Typed host storage for the canonical topology-only hydrogen-bond prepass.
// Atom/bond traversal and unresolved-base semantics remain in the shared
// kernel.
template <template <Device> class Ops, typename... RotamerTypes>
auto hbond_host_indices(
    ContextManager& context,
    int64_t n_coordinates,
    TView<int32_t, 1, Device::CPU> offsets,
    TView<int32_t, 2, Device::CPU> first_rot,
    TView<int32_t, 2, Device::CPU> first_type,
    TView<int32_t, 1, Device::CPU> block_for_rot,
    TView<int32_t, 1, Device::CPU> pose_for_rot,
    TView<Vec<int32_t, 2>, 3, Device::CPU> connections,
    TView<int32_t, 1, Device::CPU> n_atoms,
    TView<int32_t, 1, Device::CPU> n_connections,
    TView<int32_t, 2, Device::CPU> connection_atoms,
    TView<int32_t, 1, Device::CPU> n_bonds,
    TView<Vec<int32_t, 3>, 2, Device::CPU> bonds,
    TView<Vec<int32_t, 2>, 2, Device::CPU> bond_ranges,
    TView<int32_t, 2, Device::CPU> is_hydrogen,
    TView<int32_t, 2, Device::CPU> n_donors,
    TView<int32_t, 2, Device::CPU> n_acceptors,
    TView<int32_t, 3, Device::CPU> donor_indices,
    TView<int32_t, 3, Device::CPU> acceptor_indices,
    TView<int32_t, 3, Device::CPU> hybridizations,
    RotamerTypes... explicit_types) {
  using namespace score::hbond::potentials;
  using I = int32_t;
  constexpr auto D = Device::CPU;
  auto rots = offsets.size(0);
  static_assert(sizeof...(RotamerTypes) <= 1);
  TView<I, 1, D> types;
  if constexpr (sizeof...(RotamerTypes))
    types = std::get<0>(std::make_tuple(explicit_types...));
  else {
    if (first_type.size(0) * first_type.size(1) != rots)
      throw std::invalid_argument("invalid whole-pose HBond type layout");
    types = contiguous_view<I, 1, D>(first_type.data(), {rots});
  }
  if (n_coordinates < 0 || n_coordinates > INT32_MAX / 3
      || first_rot.size(0) != first_type.size(0)
      || first_rot.size(1) != first_type.size(1) || types.size(0) != rots
      || block_for_rot.size(0) != rots || pose_for_rot.size(0) != rots)
    throw std::invalid_argument("invalid hydrogen-bond topology layout");
  auto result = OwnedHostTensor<I, 2, D>::full({n_coordinates, 3}, -1);
  HBondBasesCtx<float, I, D> bases{};
  bases.rot_coord_offset = offsets;
  bases.first_rot_for_block = first_rot;
  bases.first_rot_block_type = first_type;
  bases.block_ind_for_rot = block_for_rot;
  bases.pose_ind_for_rot = pose_for_rot;
  bases.block_type_ind_for_rot = types;
  bases.pose_stack_inter_residue_connections = connections;
  bases.block_type_n_atoms = n_atoms;
  bases.block_type_n_interblock_bonds = n_connections;
  bases.block_type_atoms_forming_chemical_bonds = connection_atoms;
  bases.block_type_n_all_bonds = n_bonds;
  bases.block_type_all_bonds = bonds;
  bases.block_type_atom_all_bond_ranges = bond_ranges;
  bases.block_type_atom_is_hydrogen = is_hydrogen;
  bases.block_type_tile_n_donH = n_donors;
  bases.block_type_tile_n_acc = n_acceptors;
  bases.block_type_tile_donH_inds = donor_indices;
  bases.block_type_tile_acc_inds = acceptor_indices;
  bases.block_type_tile_hybridization = hybridizations;
  bases.derived_atom_inds = result.view;
  launch_hbond_bases<Ops, D, float, I, false>(context, bases);
  return result;
}
}  // namespace tmol::jax

extern "C" size_t tmol_kinematic_host_array_size() {
  return sizeof(tmol::jax::HostArray);
}

extern "C" int tmol_kinematic_host_prepare(
    int32_t operation,
    const tmol::jax::HostArray* arrays,
    int64_t array_count,
    const int64_t* scalars,
    int64_t scalar_count,
    void** result,
    char* error,
    size_t error_size) {
  if (!result) return 1;
  *result = nullptr;
  try {
    using Planner = tmol::kinematics::HostPoseKinematics<>;
    tmol::ContextManager context;
    tmol::jax::HostArguments inputs(arrays, array_count, scalars, scalar_count);
    auto output = std::make_unique<tmol::jax::HostResult>();
    // The generic invoker derives argument types directly from each method.
#define TMOL_HOST_OPERATION(ID, NAME)                                 \
  case ID:                                                            \
    tmol::jax::invoke_host(&Planner::NAME, context, inputs, *output); \
    break
    switch (operation) {
      TMOL_HOST_OPERATION(0, get_kfo_indices_for_atoms);
      TMOL_HOST_OPERATION(1, get_kfo_atom_parents);
      TMOL_HOST_OPERATION(2, get_children);
      TMOL_HOST_OPERATION(3, get_id_and_frame_xyz);
      TMOL_HOST_OPERATION(4, calculate_ff_edge_delays);
      TMOL_HOST_OPERATION(5, get_jump_atom_indices);
      TMOL_HOST_OPERATION(6, get_block_parent_connectivity_from_toposort);
      TMOL_HOST_OPERATION(7, get_scans2);
      TMOL_HOST_OPERATION(8, create_minimizer_map);
      TMOL_HOST_OPERATION(10, assemble_pose);
      case 11:
        if (array_count < 1)
          throw std::invalid_argument("missing inverse coordinates");
        if (arrays[0].scalar_type == 3)
          tmol::jax::invoke_host(
              &tmol::jax::host_inverse_kinematics<float>,
              context,
              inputs,
              *output);
        else if (arrays[0].scalar_type == 4)
          tmol::jax::invoke_host(
              &tmol::jax::host_inverse_kinematics<double>,
              context,
              inputs,
              *output);
        else
          throw std::invalid_argument("inverse coordinates must be float32/64");
        break;
      case 12:
        if (array_count < 1)
          throw std::invalid_argument("missing sampling coordinates");
        if (arrays[0].scalar_type == 3)
          tmol::jax::invoke_host(
              &tmol::pack::rotamer::dunbrack::HostDunbrackSampling<float>::f,
              context,
              inputs,
              *output);
        else if (arrays[0].scalar_type == 4)
          tmol::jax::invoke_host(
              &tmol::pack::rotamer::dunbrack::HostDunbrackSampling<double>::f,
              context,
              inputs,
              *output);
        else
          throw std::invalid_argument(
              "sampling coordinates must be float32/64");
        break;
      case 13:
        tmol::jax::invoke_host(
            &tmol::jax::sampling_host_layout, context, inputs, *output);
        break;
      case 14:
        tmol::jax::invoke_host(
            &tmol::jax::sampling_host_offsets, context, inputs, *output);
        break;
      case 15:
        tmol::jax::invoke_host(
            &tmol::jax::hbond_host_indices<tmol::jax::CpuOps>,
            context,
            inputs,
            *output);
        break;
      case 16:
        tmol::jax::invoke_host(
            &tmol::jax::hbond_host_indices<
                tmol::jax::CpuOps,
                tmol::TView<int32_t, 1, tmol::Device::CPU>>,
            context,
            inputs,
            *output);
        break;
      default:
        throw std::invalid_argument("unknown native pose operation");
    }
#undef TMOL_HOST_OPERATION
    *result = output.release();
    return 0;
  } catch (const std::exception& exception) {
    if (error && error_size)
      std::snprintf(error, error_size, "%s", exception.what());
  } catch (...) {
    if (error && error_size)
      std::snprintf(error, error_size, "unknown native pose preparation error");
  }
  return 1;
}

extern "C" int64_t tmol_kinematic_host_result_count(const void* result) {
  if (!result) return -1;
  return static_cast<const tmol::jax::HostResult*>(result)->arrays.size();
}

extern "C" int tmol_kinematic_host_result_array(
    const void* result, int64_t index, tmol::jax::HostArray* array) {
  if (!result || !array || index < 0) return 1;
  const auto& values =
      static_cast<const tmol::jax::HostResult*>(result)->arrays;
  if (size_t(index) >= values.size()) return 1;
  *array = values[index].array;
  return 0;
}

extern "C" void tmol_kinematic_host_release(void* result) {
  delete static_cast<tmol::jax::HostResult*>(result);
}
