// One CPU/CUDA ABI for the native LJ/LK/electrostatic execution group.
#pragma once

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
    tmol_ljlk,
    (tmol::jax::run<tmol::jax::IndependentNonbondedKernel<false>, double>),
    .Attr<double>("reach"));

TMOL_JAX_ENERGY(
    tmol_ljlk_precise,
    (tmol::jax::
         run<tmol::jax::IndependentNonbondedKernel<false, true>, double>),
    .Attr<double>("reach"));

TMOL_JAX_ENERGY(
    tmol_elec,
    (tmol::jax::run<tmol::jax::IndependentNonbondedKernel<true>, double>),
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
