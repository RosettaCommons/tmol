"""Mature FastRelax schedule and protocol; frontends supply execution only."""

import time
import warnings
from math import pi
from typing import Protocol, TypeVar
from collections.abc import Sequence

PoseT = TypeVar("PoseT")
ScoreT = TypeVar("ScoreT", contravariant=True)


class RelaxMinimizer(Protocol[PoseT, ScoreT]):
    """Framework-neutral callable contract for a FastRelax minimization stage."""

    def __call__(
        self,
        pose_stack: PoseT,
        sfxn: ScoreT,
        *,
        verbose: bool,
        rtol: float | None = None,
    ) -> PoseT:
        """Minimize and return poses, accepting an optional stage tolerance."""


RelaxScheduleEntry = float | int | dict[str, float]

# Jack Maguire's tuned MonomerRelax2019 schedule. Fractions scale the score
# function's initial fa_rep and constraint weights at each pack-minimize stage;
# min_rtol is the stage's minimizer relative tolerance.
DEFAULT_RELAX_SCHEDULE = [
    {
        "fa_rep_pack_frac": 0.040,
        "fa_rep_min_frac": 0.051,
        "cst_frac": 1.0,
        "min_rtol": 0.01,
    },
    {
        "fa_rep_pack_frac": 0.265,
        "fa_rep_min_frac": 0.280,
        "cst_frac": 0.5,
        "min_rtol": 0.01,
    },
    {
        "fa_rep_pack_frac": 0.559,
        "fa_rep_min_frac": 0.581,
        "cst_frac": 0.0,
        "min_rtol": 0.01,
    },
    {
        "fa_rep_pack_frac": 1.000,
        "fa_rep_min_frac": 1.000,
        "cst_frac": 0.0,
        "min_rtol": 5e-4,
    },
]


def _normalize_schedule(
    schedule: Sequence[RelaxScheduleEntry],
    constrain: bool = False,
    ramp_constraints: bool = False,
) -> list[tuple[float, float, float, float | None]]:
    """Normalize user schedule entries into pack, minimize, and constraint weights.

    Args:
        schedule: Numbers, used for both fa_rep stages, or dictionaries with
            ``fa_rep_pack_frac``, ``fa_rep_min_frac``, and optional
            ``cst_frac`` and ``min_rtol``.
        constrain: Whether the score function has an active constraint term.
        ramp_constraints: Whether to ramp its weight to zero.

    Returns:
        ``(pack_fraction, minimize_fraction, constraint_fraction, min_rtol)``
        per step.

    Raises:
        ValueError: If an entry is neither a float/int nor a dict with the
            required keys.
    """

    def constraint_fraction(step_index: int) -> float:
        """Determine the constraint fraction for a particular step.

        If ramping, ramp constraint frations down from 1.0 to 0.0 over
        the first half of the schedule, then keep at 0.0.
        """
        if not constrain:
            return 0.0
        if not ramp_constraints:
            return 1.0
        n_steps = len(schedule)
        if step_index > n_steps // 2:
            return 0.0
        return 1 - step_index / (n_steps / 2)

    normalized = []
    for i, entry in enumerate(schedule):
        if isinstance(entry, (int, float)):
            constraint = constraint_fraction(i)
            normalized.append((float(entry), float(entry), constraint, None))
        elif isinstance(entry, dict):
            pack_frac = float(entry["fa_rep_pack_frac"])
            min_frac = float(entry["fa_rep_min_frac"])
            constraint = (
                constraint_fraction(i)
                if "cst_frac" not in entry
                else float(entry["cst_frac"])
            )
            min_rtol = entry.get("min_rtol")
            min_rtol = None if min_rtol is None else float(min_rtol)
            normalized.append((pack_frac, min_frac, constraint, min_rtol))
        else:
            raise ValueError(
                f"Schedule entry must be a number or a dict with keys"
                f" 'fa_rep_pack_frac', 'fa_rep_min_frac', and optionally 'cst_frac'"
                f" and 'min_rtol', got {type(entry)}"
            )
    return normalized


KINEMATIC_RELAX_LBFGS_OPTIONS = {
    "rtol": 0.01,
    "atol": None,
    "patience": 1,
    "gradtol": 180.0 / pi,
}


def fast_relax(
    pose_stack,
    sfxn,
    *,
    execution,
    packer_palette=None,
    task_operations=None,
    num_repeats=2,
    ramp_constraints=None,
    schedule=None,
    minimizer=None,
    release_between_stages=True,
    verbose=False,
):
    """Run the shared schedule, releasing stage state and accepting per pose."""
    if schedule is None:
        schedule = DEFAULT_RELAX_SCHEDULE

    constraint_weight_start = sfxn.get_weight(execution.score_type.constraint)
    use_constraints = constraint_weight_start != 0
    if not use_constraints and ramp_constraints:
        print(
            "Warning: ramp_constraints is True but sfxn's 'constraint' weight is 0; no constraints will be used."
        )
    if ramp_constraints is None:
        ramp_constraints = True

    steps = _normalize_schedule(schedule, use_constraints, ramp_constraints)

    if not steps:
        raise ValueError("Relax schedule must contain at least one step.")

    # Warn if the final step doesn't restore fa_rep to its full weight.
    final_min_frac = steps[-1][1]
    if abs(final_min_frac - 1.0) > 1e-6:
        warnings.warn(
            f"Final schedule step has fa_rep min fraction {final_min_frac:.4f},"
            f" not 1.0!",
            stacklevel=2,
        )

    if task_operations is None:
        task_operations = execution.default_task_operations(pose_stack, sfxn)

    fa_rep_start = float(sfxn.get_weight(execution.score_type.fa_ljrep))

    wpsm = sfxn.render_whole_pose_scoring_module(pose_stack)
    best_score = wpsm(pose_stack.coords)
    del wpsm
    best_ps = execution.clone(pose_stack)

    if minimizer is None:
        minimizer = execution.default_minimizer()
    if packer_palette is None:
        packer_palette = execution.default_palette()

    ps = pose_stack
    for _ in range(num_repeats):
        for pack_frac, min_frac, cst_frac, min_rtol in steps:
            ps = relax_pack_min_step(
                execution=execution,
                pose_stack=ps,
                sfxn=sfxn,
                packer_palette=packer_palette,
                fa_rep_pack_weight=pack_frac * fa_rep_start,
                fa_rep_min_weight=min_frac * fa_rep_start,
                cst_weight=cst_frac * constraint_weight_start,
                task_operations=task_operations,
                minimizer=minimizer,
                min_rtol=min_rtol,
                verbose=verbose,
                release_between_stages=release_between_stages,
            )

        best_ps, best_score = accept_best(
            execution, sfxn, best_ps, best_score, ps, verbose
        )
        ps = execution.clone(best_ps)
    if use_constraints:
        # Restore original constraint weight to the score function
        sfxn.set_weight(execution.score_type.constraint, constraint_weight_start)
    return ps


def relax_pack_min_step(
    pose_stack,
    sfxn,
    packer_palette,
    fa_rep_pack_weight,
    fa_rep_min_weight,
    cst_weight,
    task_operations,
    minimizer,
    verbose,
    release_between_stages=True,
    min_rtol=None,
    *,
    execution,
):
    """Pack then minimize with the mature stage weights and lifetime hooks."""
    if verbose:
        execution.synchronize(pose_stack)
    start_time = time.perf_counter()
    task = execution.task(pose_stack, packer_palette)
    for op in task_operations:
        op(task)

    sfxn.set_weight(execution.score_type.fa_ljrep, fa_rep_pack_weight)
    sfxn.set_weight(execution.score_type.constraint, cst_weight)
    if verbose:
        print(
            f"packing with fa_rep of {fa_rep_pack_weight: .2f} and constraint weight of {cst_weight: .2f}"
        )
    if verbose:
        execution.synchronize(pose_stack)
    end_time1 = time.perf_counter()
    packed_pose_stack = execution.pack(pose_stack, sfxn, task, verbose)

    sfxn.set_weight(execution.score_type.fa_ljrep, fa_rep_min_weight)
    if verbose:
        print(
            f"minimizing with fa_rep of {fa_rep_min_weight: .2f} and constraint weight of {cst_weight: .2f}"
        )
    if verbose:
        execution.synchronize(pose_stack)
    end_time2 = time.perf_counter()
    stage_kwargs = {} if min_rtol is None else {"rtol": min_rtol}
    minimized_pose_stack = minimizer(
        packed_pose_stack,
        sfxn,
        verbose=verbose,
        **stage_kwargs,
    )
    # Releasing costs a network construction and, with cuda_graph, a capture at
    # every stage; retaining costs the scorer and L-BFGS history through the
    # next packing phase. Large systems are bound by the second -- it is why
    # this release exists -- and small ones by the first.
    if release_between_stages:
        release_retained_state = getattr(minimizer, "release_retained_state", None)
        if release_retained_state is not None:
            release_retained_state()
    if verbose:
        execution.synchronize(pose_stack)
    end_time3 = time.perf_counter()

    if verbose:
        print(
            f"pack-min {end_time3 - start_time: .2f} task-init {end_time1 - start_time: .2f}"
            + f" packing {end_time2 - end_time1: .2f} min {end_time3 - end_time2: .2f}"
        )

    return minimized_pose_stack


def accept_best(
    execution,
    sfxn,
    best_pose_stack,
    best_pose_score,
    candidate_pose_stack,
    verbose=False,
):
    """Compare the rendered objective and accept independently for each pose."""
    scorer = sfxn.render_whole_pose_scoring_module(candidate_pose_stack)
    candidate_score = scorer(candidate_pose_stack.coords)
    better = candidate_score < best_pose_score
    if not bool(better.any()):
        return best_pose_stack, best_pose_score
    if verbose:
        print("accepting new best scores")
        print(f" old best score: {best_pose_score[better]}")
        print(f" new best score: {candidate_score[better]}")
    return (
        execution.select_pose(best_pose_stack, candidate_pose_stack, better),
        execution.select_scores(best_pose_score, candidate_score, better),
    )


def default_task_operation(pose, samplers, add_group_sampler):
    """Use the same repacking and covalent-group registration policy."""

    def apply(task):
        task.restrict_to_repacking()
        for sampler in samplers:
            task.add_conformer_sampler(sampler)
        add_group_sampler(task, pose)

    return [apply]
