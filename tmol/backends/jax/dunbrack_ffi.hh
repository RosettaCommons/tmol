// Typed XLA views for the shared whole-pose Dunbrack kernels.
#pragma once
#include "workspace_ffi.hh"
#include <tmol/score/dunbrack/potentials/dunbrack_pose_kernels.hh>

namespace tmol::jax {
template <typename Real>
struct DunbrackWorkspace {
  using I = int32_t;
  using Derivative = tmol::portable::Mat<Real, 4, 3>;
  ByteWorkspace storage;
  Vec<I, 4>* atoms;
  Real* values;
  Derivative* dihedral_derivatives;
  Derivative* term_derivatives;
  I* rotameric_assignment;
  I* semirotameric_assignment;

  DunbrackWorkspace(
      size_t rots, size_t dihedrals, bool derivatives, void* data = nullptr)
      : storage(data) {
    atoms = storage.take<Vec<I, 4>>(rots * dihedrals);
    values = storage.take<Real>(rots * dihedrals);
    auto derivative_rots = derivatives ? rots : 0;
    dihedral_derivatives =
        storage.take<Derivative>(derivative_rots * dihedrals);
    term_derivatives = storage.take<Derivative>(derivative_rots * 3);
    rotameric_assignment = storage.take<I>(rots);
    semirotameric_assignment = storage.take<I>(rots);
  }
};

struct DunbrackKernel {
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      ffi::AnyBuffer energy,
      const ffi::AnyBuffer* gradient) {
    if (a.size() != 36)
      throw std::invalid_argument("Dunbrack FFI expects 36 operands");
    auto real = a[0].element_type();
    if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
      throw std::invalid_argument("Dunbrack coordinates must be float32/64");
    auto integer = ffi::DataType::S32;
    require_shape(a[0], real, {-1, -1, 3});
    auto poses = a[0].dimensions()[0], atoms = a[0].dimensions()[1];
    require_shape(a[3], integer, {poses, -1});
    auto blocks = a[3].dimensions()[1];
    require_shape(a[1], integer, {poses * blocks});
    require_shape(a[2], integer, {poses, blocks});
    require_shape(a[4], integer, {poses, blocks, -1, 2});
    require_shape(a[24], integer, {-1});
    auto types = a[24].dimensions()[0];
    require_shape(a[5], integer, {types, a[4].dimensions()[2], -1});
    require_shape(a[6], real, {-1, -1, -1});
    for (int index : {7, 8})
      require_shape(a[index], integer, {a[6].dimensions()[0], 2});
    require_shape(a[9], real, {-1, -1, -1});
    require_shape(
        a[10],
        real,
        {a[9].dimensions()[0], a[9].dimensions()[1], a[9].dimensions()[2]});
    for (int index : {11, 12})
      require_shape(a[index], integer, {a[9].dimensions()[0], 2});
    require_shape(a[13], real, {-1, 2});
    for (int index : {14, 15})
      require_shape(a[index], real, {a[13].dimensions()[0], 2});
    for (int index : {16, 17}) require_shape(a[index], integer, {-1});
    require_shape(a[18], real, {-1, -1, -1, -1});
    for (int index : {19, 20})
      require_shape(a[index], integer, {a[18].dimensions()[0], 3});
    require_shape(a[21], real, {-1, 3});
    for (int index : {22, 23})
      require_shape(a[index], real, {a[21].dimensions()[0], 3});
    require_shape(a[25], integer, {types, -1, 4, 3});
    for (int index = 26; index != 36; ++index)
      require_shape(a[index], integer, {types});
    require_shape(energy, real, {3, poses});
    if (gradient) require_shape(*gradient, real, {3, poses, atoms, 3});
    if (poses * atoms > std::numeric_limits<int32_t>::max())
      throw std::invalid_argument("Dunbrack atom layout exceeds int32");
  }

  static size_t workspace_bytes(
      const std::vector<ffi::AnyBuffer>& a, bool derivative) {
    auto rots = a[1].dimensions()[0], dihedrals = a[25].dimensions()[1];
    return a[0].element_type() == ffi::DataType::F32
               ? DunbrackWorkspace<float>(rots, dihedrals, derivative)
                     .storage.size_bytes()
               : DunbrackWorkspace<double>(rots, dihedrals, derivative)
                     .storage.size_bytes();
  }

  template <template <Device> class Ops, Device D, typename Real>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      ffi::AnyBuffer energy,
      const ffi::AnyBuffer* gradient,
      ffi::AnyBuffer workspace) {
    using I = int32_t;
    using UAID = tmol::UnresolvedAtomID<I>;
    using Derivative = typename DunbrackWorkspace<Real>::Derivative;
    static_assert(sizeof(UAID) == 3 * sizeof(I));
    auto poses = a[0].dimensions()[0], atoms = a[0].dimensions()[1];
    auto rots = a[1].dimensions()[0], dihedrals = a[25].dimensions()[1];
    auto derivative_rots = gradient ? rots : 0;
    DunbrackWorkspace<Real> work(
        rots, dihedrals, gradient != nullptr, workspace.untyped_data());
    ContextManager context;
    score::dunbrack::potentials::dunbrack_pose_forward<Ops, D, Real, I, I>(
        context,
        contiguous_view<Vec<Real, 3>, 1, D>(
            a[0].untyped_data(), {poses * atoms}),
        buffer_view<I, 1, D>(a[1]),
        buffer_view<I, 2, D>(a[2]),
        buffer_view<I, 2, D>(a[3]),
        buffer_view<Vec<I, 2>, 3, D>(a[4]),
        buffer_view<I, 3, D>(a[5]),
        buffer_view<Real, 3, D>(a[6]),
        buffer_view<Vec<I, 2>, 1, D>(a[7]),
        buffer_view<Vec<I, 2>, 1, D>(a[8]),
        buffer_view<Real, 3, D>(a[9]),
        buffer_view<Real, 3, D>(a[10]),
        buffer_view<Vec<I, 2>, 1, D>(a[11]),
        buffer_view<Vec<I, 2>, 1, D>(a[12]),
        buffer_view<Vec<Real, 2>, 1, D>(a[13]),
        buffer_view<Vec<Real, 2>, 1, D>(a[14]),
        buffer_view<Vec<Real, 2>, 1, D>(a[15]),
        buffer_view<I, 1, D>(a[16]),
        buffer_view<I, 1, D>(a[17]),
        buffer_view<Real, 4, D>(a[18]),
        buffer_view<Vec<I, 3>, 1, D>(a[19]),
        buffer_view<Vec<I, 3>, 1, D>(a[20]),
        buffer_view<Vec<Real, 3>, 1, D>(a[21]),
        buffer_view<Vec<Real, 3>, 1, D>(a[22]),
        buffer_view<Vec<Real, 3>, 1, D>(a[23]),
        buffer_view<I, 1, D>(a[24]),
        buffer_view<UAID, 3, D>(a[25]),
        buffer_view<I, 1, D>(a[26]),
        buffer_view<I, 1, D>(a[27]),
        buffer_view<I, 1, D>(a[28]),
        buffer_view<I, 1, D>(a[29]),
        buffer_view<I, 1, D>(a[30]),
        buffer_view<I, 1, D>(a[31]),
        buffer_view<I, 1, D>(a[32]),
        buffer_view<I, 1, D>(a[33]),
        buffer_view<I, 1, D>(a[34]),
        buffer_view<I, 1, D>(a[35]),
        false,
        gradient != nullptr,
        contiguous_view<Real, 4, D>(energy.untyped_data(), {3, poses, 1, 1}),
        contiguous_view<Vec<Real, 3>, 2, D>(
            gradient ? gradient->untyped_data() : nullptr,
            {3, gradient ? poses * atoms : 0}),
        contiguous_view<Vec<I, 4>, 2, D>(work.atoms, {rots, dihedrals}),
        contiguous_view<Real, 2, D>(work.values, {rots, dihedrals}),
        contiguous_view<Derivative, 2, D>(
            work.dihedral_derivatives, {derivative_rots, dihedrals}),
        contiguous_view<Derivative, 2, D>(
            work.term_derivatives, {derivative_rots, 3}),
        contiguous_view<I, 1, D>(work.rotameric_assignment, {rots}),
        contiguous_view<I, 1, D>(work.semirotameric_assignment, {rots}));
  }
};
}  // namespace tmol::jax
