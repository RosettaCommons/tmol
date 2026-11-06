"""Conformer selection policy on boolean arrays supplied by each executor."""


def include_current_mask(original, allowed, sampler_allowed):
    """Include only allowed original identities handled by this sampler."""
    return original & allowed & sampler_allowed


def fallback_mask(original, none_allowed, all_samplers_disabled, covered, allowed):
    """Keep fixed residues and fill uncovered positions with their input state.

    ``covered`` aggregates actual samples over all identities at a position.
    A designed position covered by another identity must not regain its input
    identity just because that original identity produced no samples.
    """
    return original & (none_allowed | all_samplers_disabled | (~covered & allowed))
