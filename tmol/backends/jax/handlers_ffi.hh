// One ABI contract for CPU/CUDA handlers; execution policy stays in each TU.
#pragma once
#ifdef __CUDACC__
#define TMOL_JAX_CONTEXT xla::ffi::PlatformStream<cudaStream_t>
#define TMOL_JAX_TRAITS {xla::ffi::Traits::kCmdBufferCompatible}
#else
#define TMOL_JAX_CONTEXT xla::ffi::ThreadPool
#define TMOL_JAX_TRAITS \
  {                     \
  }
#endif

#define TMOL_JAX_ENERGY(NAME, FN, ATTRS) \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(         \
      NAME,                              \
      FN,                                \
      xla::ffi::Ffi::Bind()              \
          .Ctx<TMOL_JAX_CONTEXT>()       \
          .RemainingArgs()               \
          .Ret<xla::ffi::AnyBuffer>()    \
          .RemainingRets() ATTRS,        \
      TMOL_JAX_TRAITS)
#define TMOL_JAX_STAGE(NAME, FN, ATTRS)  \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(         \
      NAME,                              \
      FN,                                \
      xla::ffi::Ffi::Bind()              \
          .Ctx<TMOL_JAX_CONTEXT>()       \
          .RemainingArgs()               \
          .RemainingRets()               \
          .Attr<int64_t>("operation")    \
          .Attr<int64_t>("extent")       \
          .Attr<int64_t>("width") ATTRS, \
      TMOL_JAX_TRAITS)

TMOL_JAX_ENERGY(tmol_genbonded, tmol::jax::run<tmol::jax::GenBondedKernel>, );

TMOL_JAX_ENERGY(tmol_cartbonded, tmol::jax::run<tmol::jax::CartBondedKernel>, );
TMOL_JAX_ENERGY(
    tmol_cartbonded_precise,
    tmol::jax::run<tmol::jax::CartBondedPreciseKernel>, );

TMOL_JAX_ENERGY(
    tmol_backbone_torsion, tmol::jax::run<tmol::jax::BackboneTorsionKernel>, );

TMOL_JAX_ENERGY(
    tmol_metal_coordination,
    tmol::jax::run<tmol::jax::MetalCoordinationKernel>, );

TMOL_JAX_ENERGY(
    tmol_na_torsion,
    (tmol::jax::run<tmol::jax::NaTorsionKernel, double, double>),
    .Attr<double>("pucker_temperature").Attr<double>("bin_blend_sdev"));

TMOL_JAX_ENERGY(tmol_dunbrack, tmol::jax::run<tmol::jax::DunbrackKernel>, );

TMOL_JAX_ENERGY(
    tmol_nonbonded,
    (tmol::jax::run<tmol::jax::NonbondedKernel<false>, double>),
    .Attr<double>("reach"));

TMOL_JAX_ENERGY(
    tmol_nonbonded_weighted,
    (tmol::jax::run<tmol::jax::NonbondedKernel<true>, double>),
    .Attr<double>("reach"));

TMOL_JAX_ENERGY(
    tmol_nonbonded_precise,
    (tmol::jax::run<tmol::jax::NonbondedKernel<false, true>, double>),
    .Attr<double>("reach"));

TMOL_JAX_ENERGY(
    tmol_nonbonded_weighted_precise,
    (tmol::jax::run<tmol::jax::NonbondedKernel<true, true>, double>),
    .Attr<double>("reach"));

TMOL_JAX_ENERGY(
    tmol_constraint_torsion,
    tmol::jax::run<tmol::jax::ConstraintTorsionKernel>, );

TMOL_JAX_ENERGY(
    tmol_precise_torsion, tmol::jax::run<tmol::jax::PreciseTorsionKernel>, );

TMOL_JAX_ENERGY(
    tmol_ljlk,
    (tmol::jax::run<tmol::jax::IndependentNonbondedKernel<false>, double>),
    .Attr<double>("reach"));

TMOL_JAX_ENERGY(
    tmol_elec,
    (tmol::jax::run<tmol::jax::IndependentNonbondedKernel<true>, double>),
    .Attr<double>("reach"));

TMOL_JAX_ENERGY(
    tmol_inverse_kinematics,
    tmol::jax::run<tmol::jax::InverseKinematicsKernel>, );

TMOL_JAX_ENERGY(
    tmol_lk_ball,
    (tmol::jax::run<tmol::jax::LKBallKernel, double>),
    .Attr<double>("reach"));

TMOL_JAX_ENERGY(
    tmol_lk_ball_precise,
    (tmol::jax::run<tmol::jax::LKBallPreciseKernel, double>),
    .Attr<double>("reach"));
TMOL_JAX_ENERGY(
    tmol_lk_ball_weighted_precise,
    (tmol::jax::run<tmol::jax::LKBallWeightedPreciseKernel, double>),
    .Attr<double>("reach"));

TMOL_JAX_ENERGY(tmol_disulfide, tmol::jax::run<tmol::jax::DisulfideKernel>, );

TMOL_JAX_ENERGY(tmol_hbond, tmol::jax::run<tmol::jax::HBondKernel>, );

TMOL_JAX_ENERGY(
    tmol_lk_ball_weighted,
    (tmol::jax::run<tmol::jax::LKBallWeightedKernel, double>),
    .Attr<double>("reach"));

TMOL_JAX_ENERGY(
    tmol_copy_conformer_dofs,
    (tmol::jax::run<tmol::jax::CopyConformerDofsKernel, int64_t>),
    .Attr<int64_t>("sampler"));

TMOL_JAX_ENERGY(
    tmol_assign_conformer_chi,
    tmol::jax::run<tmol::jax::AssignConformerChiKernel>, );

TMOL_JAX_ENERGY(
    tmol_copy_conformer_rows,
    (tmol::jax::run<tmol::jax::CopyConformerRowsKernel, int64_t, int64_t>),
    .Attr<int64_t>("n_rows").Attr<int64_t>("root_shift"));

TMOL_JAX_ENERGY(
    tmol_correct_conformer_chi,
    tmol::jax::run<tmol::jax::CorrectConformerChiKernel>, );

TMOL_JAX_STAGE(
    tmol_graph_topology,
    tmol::jax::run_graph_stage<tmol::jax::GraphTopologyStage>, );

TMOL_JAX_STAGE(
    tmol_graph_retention,
    tmol::jax::run_graph_stage<tmol::jax::GraphRetentionStage>, );

TMOL_JAX_STAGE(
    tmol_rotamer_dispatch,
    (tmol::jax::run_graph_stage<tmol::jax::RotamerDispatchStage, double>),
    .Attr<double>("reach"));

TMOL_JAX_STAGE(
    tmol_rotamer_elec,
    (tmol::jax::
         run_graph_stage<tmol::jax::RotamerNonbondedStage<true>, double>),
    .Attr<double>("reach"));

TMOL_JAX_STAGE(
    tmol_rotamer_ljlk,
    (tmol::jax::
         run_graph_stage<tmol::jax::RotamerNonbondedStage<false>, double>),
    .Attr<double>("reach"));

TMOL_JAX_ENERGY(
    tmol_graph_prefix, tmol::jax::run<tmol::jax::GraphPrefixKernel>, );

TMOL_JAX_STAGE(
    tmol_rotamer_dunbrack,
    tmol::jax::run_graph_stage<tmol::jax::RotamerDunbrackStage>, );

TMOL_JAX_STAGE(
    tmol_rotamer_lk_ball,
    (tmol::jax::run_graph_stage<tmol::jax::RotamerLKBallStage, double>),
    .Attr<double>("reach"));

TMOL_JAX_STAGE(
    tmol_load_balance,
    tmol::jax::run_graph_stage<tmol::jax::LoadBalanceStage>, );

TMOL_JAX_STAGE(
    tmol_connected_rotamer_dispatch,
    tmol::jax::run_graph_stage<tmol::jax::ConnectedRotamerStage>, );

TMOL_JAX_STAGE(
    tmol_rotamer_genbonded,
    (tmol::jax::run_graph_stage<tmol::jax::RotamerGenBondedStage, double>),
    .Attr<double>("reach"));

TMOL_JAX_STAGE(
    tmol_rotamer_hbond,
    (tmol::jax::run_graph_stage<tmol::jax::RotamerHBondStage, double>),
    .Attr<double>("reach"));

TMOL_JAX_STAGE(
    tmol_rotamer_cartbonded,
    (tmol::jax::run_graph_stage<tmol::jax::RotamerCartBondedStage, double>),
    .Attr<double>("reach"));

TMOL_JAX_STAGE(
    tmol_rotamer_na_torsion,
    (tmol::jax::
         run_graph_stage<tmol::jax::RotamerNaTorsionStage, double, double>),
    .Attr<double>("pucker_temperature").Attr<double>("bin_blend_sdev"));

TMOL_JAX_STAGE(
    tmol_na_rotamer_topology,
    tmol::jax::run_graph_stage<tmol::jax::NaRotamerTopologyStage>, );

TMOL_JAX_STAGE(
    tmol_backbone_rotamer_topology,
    tmol::jax::run_graph_stage<tmol::jax::BackboneRotamerTopologyStage>, );

TMOL_JAX_STAGE(
    tmol_rotamer_backbone,
    (tmol::jax::run_graph_stage<tmol::jax::RotamerBackboneStage, double>),
    .Attr<double>("reach"));

TMOL_JAX_ENERGY(
    tmol_graph_maximum_prefix, tmol::jax::run<tmol::jax::GraphMaximumKernel>, );

TMOL_JAX_STAGE(
    tmol_disulfide_rotamer_topology,
    tmol::jax::run_graph_stage<tmol::jax::ConnectionTopologyStage<false>>, );

TMOL_JAX_STAGE(
    tmol_metal_rotamer_topology,
    tmol::jax::run_graph_stage<tmol::jax::ConnectionTopologyStage<true>>, );

TMOL_JAX_STAGE(
    tmol_rotamer_disulfide,
    (tmol::jax::
         run_graph_stage<tmol::jax::RotamerConnectionStage<false>, double>),
    .Attr<double>("reach"));

TMOL_JAX_STAGE(
    tmol_rotamer_metal,
    (tmol::jax::
         run_graph_stage<tmol::jax::RotamerConnectionStage<true>, double>),
    .Attr<double>("reach"));

TMOL_JAX_STAGE(
    tmol_constraint_rotamer_topology,
    tmol::jax::run_graph_stage<tmol::jax::ConstraintTopologyStage>, );

TMOL_JAX_STAGE(
    tmol_graph_entries,
    tmol::jax::run_graph_stage<tmol::jax::GraphEntryStage>, );

TMOL_JAX_STAGE(
    tmol_na_sampling,
    (tmol::jax::run_graph_stage<
        tmol::jax::NaSamplingStage,
        double,
        int64_t,
        double,
        double>),
    .Attr<double>("temperature")
        .Attr<int64_t>("sample_syn")
        .Attr<double>("max_syn_well")
        .Attr<double>("syn_mean"));

TMOL_JAX_STAGE(
    tmol_pose_imposition,
    tmol::jax::run_graph_stage<tmol::jax::PoseImpositionStage>, );

TMOL_JAX_STAGE(
    tmol_conformer_copy_plan,
    tmol::jax::run_graph_stage<tmol::jax::ConformerCopyPlanStage>, );

TMOL_JAX_STAGE(
    tmol_conformer_indices,
    tmol::jax::run_graph_stage<tmol::jax::ConformerIndexStage>, );

TMOL_JAX_STAGE(
    tmol_opth_sampling,
    tmol::jax::run_graph_stage<tmol::jax::OptHSamplingStage>, );

TMOL_JAX_STAGE(
    tmol_conformer_chi_derivatives,
    tmol::jax::run_graph_stage<tmol::jax::ConformerChiDerivativeStage>, );

TMOL_JAX_STAGE(
    tmol_leaf_atoms, tmol::jax::run_graph_stage<tmol::jax::LeafAtomStage>, );

TMOL_JAX_STAGE(
    tmol_canonical_candidates,
    tmol::jax::run_graph_stage<tmol::jax::CanonicalCandidateStage>, );

TMOL_JAX_STAGE(
    tmol_canonical_topology,
    tmol::jax::run_graph_stage<tmol::jax::CanonicalTopologyStage>, );

TMOL_JAX_STAGE(
    tmol_histidine, tmol::jax::run_graph_stage<tmol::jax::HistidineStage>, );

TMOL_JAX_STAGE(
    tmol_canonical_policy,
    tmol::jax::run_graph_stage<tmol::jax::CanonicalPolicyStage>, );

TMOL_JAX_STAGE(
    tmol_hydrogen_completion,
    tmol::jax::run_graph_stage<tmol::jax::HydrogenStage>, );

TMOL_JAX_STAGE(
    tmol_protonation,
    tmol::jax::run_graph_stage<tmol::jax::ProtonationStage>, );

TMOL_JAX_STAGE(
    tmol_graph_bounds,
    tmol::jax::run_graph_stage<tmol::jax::GraphBoundsStage>, );

TMOL_JAX_STAGE(
    tmol_armijo,
    (tmol::jax::run_graph_stage<tmol::jax::ArmijoStage, double, double>),
    .Attr<double>("first").Attr<double>("second"));

#undef TMOL_JAX_STAGE
#undef TMOL_JAX_ENERGY
#undef TMOL_JAX_TRAITS
#undef TMOL_JAX_CONTEXT
