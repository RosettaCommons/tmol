"""Trajectory metadata for the native annealers; no execution dependency."""

from operator import index

DEFAULT_ANNEAL_TRAJECTORIES = 128
MIN_CPU_FUNNEL_THREADS = 8


def validate_anneal_trajectories(value):
    """Return an integer trajectory count accepted by the native int32 ABI."""
    value = index(value)
    if value < 4:
        raise ValueError(f"n_trajectories must be at least 4; got {value}")
    if value * 10 // 4 > 2**31 - 1:
        raise ValueError("annealer trajectory count exceeds int32")
    return value


def annealer_trajectory_counts(high=DEFAULT_ANNEAL_TRAJECTORIES, *, cpu=False):
    """Return high/low/final counts, preserving PR626 CPU/GPU rounding."""
    high = validate_anneal_trajectories(high)
    low = high * 10 // 4 if cpu else (high // 4) * 10
    return high, low, low // 4
