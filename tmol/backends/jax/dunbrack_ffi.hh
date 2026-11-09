// Typed XLA views for the shared whole-pose Dunbrack kernels.
#pragma once
#include "energy_accumulator_ffi.hh"
#include "graph_topology_ffi.hh"
#include <tuple>
#include <tmol/score/dunbrack/potentials/dunbrack_pose_kernels.hh>

namespace tmol::jax {
template <typename Real>
struct DunbrackWorkspace {
  using I = int32_t;
  using Derivative = tmol::portable::Mat<Real, 4, 3>;
  ByteWorkspace storage;
  Vec<I, 4>* atoms;
  Real* values;
  double* energy = nullptr;
  double* gradient_accumulation = nullptr;
  Derivative* dihedral_derivatives;
  Derivative* term_derivatives;
  I* rotameric_assignment;
  I* semirotameric_assignment;

  DunbrackWorkspace(
      size_t rots,
      size_t dihedrals,
      bool derivatives,
      void* data = nullptr,
      size_t poses = 0,
      size_t gradient_values = 0)
      : storage(data) {
    atoms = storage.take<Vec<I, 4>>(rots * dihedrals);
    values = storage.take<Real>(rots * dihedrals);
    auto derivative_rots = derivatives ? rots : 0;
    dihedral_derivatives =
        storage.take<Derivative>(derivative_rots * dihedrals);
    term_derivatives = storage.take<Derivative>(derivative_rots * 3);
    rotameric_assignment = storage.take<I>(rots);
    semirotameric_assignment = storage.take<I>(rots);
    if constexpr (std::is_same_v<Real, float>)
      energy = storage.take<double>(3 * poses);
    if constexpr (std::is_same_v<Real, float>)
      gradient_accumulation = storage.take<double>(gradient_values);
  }
};

template <bool PreciseGradient = false>
struct DunbrackKernelT {
  // Shared forward evaluation writes each scratch value before reading it.
  static constexpr bool zero_workspace = false;
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
    validate_tables(a);
    require_shape(energy, real, {3, poses});
    if (gradient) require_shape(*gradient, real, {3, poses, atoms, 3});
    if (poses * atoms > std::numeric_limits<int32_t>::max())
      throw std::invalid_argument("Dunbrack atom layout exceeds int32");
  }

  static void validate_tables(const std::vector<ffi::AnyBuffer>& a) {
    auto real = a[0].element_type(), integer = ffi::DataType::S32;
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
  }

  static size_t workspace_bytes(
      const std::vector<ffi::AnyBuffer>& a, bool derivative) {
    auto rots = a[1].dimensions()[0], dihedrals = a[25].dimensions()[1];
    return a[0].element_type() == ffi::DataType::F32
               ? DunbrackWorkspace<float>(
                     rots,
                     dihedrals,
                     derivative,
                     nullptr,
                     a[0].dimensions().size() == 3 ? a[0].dimensions()[0] : 0,
                     PreciseGradient && derivative
                         ? gradient_values(
                               a[0].size_bytes() / (3 * sizeof(float)))
                         : 0)
                     .storage.size_bytes()
               : DunbrackWorkspace<double>(rots, dihedrals, derivative)
                     .storage.size_bytes();
  }

  static size_t gradient_values(size_t atoms) {
    return score::common::checked_dispatch_product(
        atoms, 9, "Dunbrack gradient values");
  }

  template <Device D, typename Real>
  static auto arguments(const std::vector<ffi::AnyBuffer>& a) {
    using I = int32_t;
    using UAID = tmol::UnresolvedAtomID<I>;
    static_assert(sizeof(UAID) == 3 * sizeof(I));
    return std::make_tuple(
        contiguous_view<Vec<Real, 3>, 1, D>(
            a[0].untyped_data(),
            {static_cast<int64_t>(a[0].size_bytes() / sizeof(Vec<Real, 3>))}),
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
        buffer_view<I, 1, D>(a[35]));
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
        rots,
        dihedrals,
        gradient != nullptr,
        workspace.untyped_data(),
        poses,
        PreciseGradient && gradient ? gradient_values(poses * atoms) : 0);
    ContextManager context;
    EnergyAccumulator<Ops, D, Real> total(context, energy, work.energy);
    using EnergyReal = typename decltype(total)::Value;
    using GradReal = std::conditional_t<PreciseGradient, EnergyReal, Real>;
    std::optional<EnergyAccumulator<Ops, D, Real>> derivative;
    if constexpr (PreciseGradient) {
      if (gradient)
        derivative.emplace(context, *gradient, work.gradient_accumulation);
    }
    auto* gradient_data =
        derivative ? derivative->template view<3>({3, poses * atoms, 3}).data()
                   : (gradient ? gradient->untyped_data() : nullptr);
    std::apply(
        [&](auto... operands) {
          score::dunbrack::potentials::dunbrack_pose_forward<
              Ops,
              D,
              Real,
              I,
              I,
              4,
              EnergyReal,
              GradReal>(
              context,
              operands...,
              false,
              gradient != nullptr,
              total.template view<4>({3, poses, 1, 1}),
              contiguous_view<Vec<GradReal, 3>, 2, D>(
                  gradient_data, {3, gradient ? poses * atoms : 0}),
              contiguous_view<Vec<I, 4>, 2, D>(work.atoms, {rots, dihedrals}),
              contiguous_view<Real, 2, D>(work.values, {rots, dihedrals}),
              contiguous_view<Derivative, 2, D>(
                  work.dihedral_derivatives, {derivative_rots, dihedrals}),
              contiguous_view<Derivative, 2, D>(
                  work.term_derivatives, {derivative_rots, 3}),
              contiguous_view<I, 1, D>(work.rotameric_assignment, {rots}),
              contiguous_view<I, 1, D>(work.semirotameric_assignment, {rots}));
        },
        arguments<D, Real>(a));
    total.finish(context);
    if (derivative) derivative->finish(context);
  }
};
using DunbrackKernel = DunbrackKernelT<false>;
using DunbrackPreciseKernel = DunbrackKernelT<true>;

template <bool PreciseGradient = false>
struct RotamerDunbrackStageT {
  static int fill(int64_t, size_t index) { return index == 1 ? -1 : 0; }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t backward,
      int64_t,
      int64_t) {
    if ((backward != 0 && backward != 1) || a.size() != 39 + backward
        || out.size() != 3)
      throw std::invalid_argument("invalid sparse Dunbrack operands");
    auto real = a[0].element_type(), integer = ffi::DataType::S32;
    if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
      throw std::invalid_argument("Dunbrack coordinates must be float32/64");
    require_shape(a[0], real, {-1, 3});
    require_shape(a[1], integer, {-1});
    auto atoms = a[0].dimensions()[0], rots = a[1].dimensions()[0];
    require_shape(a[2], integer, {-1, -1});
    auto poses = a[2].dimensions()[0], blocks = a[2].dimensions()[1];
    require_shape(a[3], integer, {poses, blocks});
    require_shape(a[4], integer, {poses, blocks, -1, 2});
    DunbrackKernel::validate_tables(a);
    for (int i = 36; i != 39; ++i) require_shape(a[i], integer, {rots});
    if (backward) {
      require_shape(a[39], real, {3, rots});
      require_shape(out[0], real, {3, atoms, 3});
    } else
      require_shape(out[0], real, {3, rots});
    require_shape(
        out[1],
        ffi::DataType::U8,
        {static_cast<int64_t>(
            DunbrackKernelT<PreciseGradient>::workspace_bytes(a, backward))});
    require_shape(out[2], integer, {3, backward ? 0 : rots});
    if (atoms > INT32_MAX || rots > INT32_MAX)
      throw std::invalid_argument("Dunbrack conformer layout exceeds int32");
  }
  template <template <Device> class Ops, Device D, typename Real>
  static void launch(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      bool backward) {
    using I = int32_t;
    using Derivative = typename DunbrackWorkspace<Real>::Derivative;
    auto atoms = a[0].dimensions()[0], rots = a[1].dimensions()[0];
    auto dihedrals = a[25].dimensions()[1];
    DunbrackWorkspace<Real> work(
        rots,
        dihedrals,
        backward,
        out[1].untyped_data(),
        0,
        PreciseGradient && backward ? DunbrackKernel::gradient_values(atoms)
                                    : 0);
    auto scratch = std::make_tuple(
        contiguous_view<Vec<I, 4>, 2, D>(work.atoms, {rots, dihedrals}),
        contiguous_view<Real, 2, D>(work.values, {rots, dihedrals}),
        contiguous_view<Derivative, 2, D>(
            work.dihedral_derivatives, {backward ? rots : 0, dihedrals}),
        contiguous_view<Derivative, 2, D>(
            work.term_derivatives, {backward ? rots : 0, 3}));
    auto origins = std::make_tuple(
        buffer_view<I, 1, D>(a[36]),
        buffer_view<I, 1, D>(a[37]),
        buffer_view<I, 1, D>(a[38]));
    ContextManager context;
    using Accumulator = EnergyAccumulator<Ops, D, Real>;
    using GradReal =
        std::conditional_t<PreciseGradient, typename Accumulator::Value, Real>;
    std::optional<Accumulator> derivative;
    if constexpr (PreciseGradient) {
      if (backward)
        derivative.emplace(context, out[0], work.gradient_accumulation);
    }
    auto* gradient_data =
        derivative ? derivative->template view<3>({3, atoms, 3}).data()
                   : out[0].untyped_data();
    std::apply(
        [&](auto... operands) {
          std::apply(
              [&](auto... temporary) {
                std::apply(
                    [&](auto... maps) {
                      if (backward) {
                        score::dunbrack::potentials::dunbrack_pose_backward<
                            Ops,
                            D,
                            Real,
                            I,
                            I,
                            2,
                            GradReal>(
                            context,
                            operands...,
                            buffer_view<Real, 2, D>(a[39]),
                            contiguous_view<Vec<GradReal, 3>, 2, D>(
                                gradient_data, {3, atoms}),
                            temporary...,
                            maps...);
                      } else {
                        score::dunbrack::potentials::
                            dunbrack_pose_forward<Ops, D, Real, I, I, 2>(
                                context,
                                operands...,
                                true,
                                false,
                                buffer_view<Real, 2, D>(out[0]),
                                contiguous_view<Vec<Real, 3>, 2, D>(
                                    nullptr, {3, 0}),
                                temporary...,
                                contiguous_view<I, 1, D>(
                                    work.rotameric_assignment, {rots}),
                                contiguous_view<I, 1, D>(
                                    work.semirotameric_assignment, {rots}),
                                maps...,
                                buffer_view<I, 2, D>(out[2]));
                      }
                    },
                    origins);
              },
              scratch);
        },
        DunbrackKernel::arguments<D, Real>(a));
    if (derivative) derivative->finish(context);
  }
  template <template <Device> class Ops, Device D>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t backward,
      int64_t,
      int64_t) {
    if (a[0].element_type() == ffi::DataType::F32)
      launch<Ops, D, float>(a, out, backward);
    else
      launch<Ops, D, double>(a, out, backward);
  }
};
using RotamerDunbrackStage = RotamerDunbrackStageT<false>;
using RotamerDunbrackPreciseStage = RotamerDunbrackStageT<true>;
}  // namespace tmol::jax
