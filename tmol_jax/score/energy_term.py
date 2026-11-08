"""Interface for a term's topology setup and differentiable energy evaluation."""

from abc import ABC, abstractmethod

from tmol_core.cache import AnnotationKey, PreparationCache
from tmol_core.scoring.channels import TERM_CHANNELS


class EnergyTerm(ABC):
    """Terms return unweighted channels; ScoreFunction applies their weights."""

    def __init__(self, param_db):
        self.param_db = param_db
        self._preparations = PreparationCache()

    def _prepared(self, name, prepare, *sources, settings=()):
        key = AnnotationKey.from_sources(self.param_db, *sources, settings=settings)
        return self._preparations.get_or_create(name, key, prepare)

    def set_options(self, options):
        """Configure supported term-specific rendering options."""

    @abstractmethod
    def score_types_provided(self):
        """ScoreType channels implemented by this term."""
        raise NotImplementedError

    def setup_for_pose_stack(self, pose_stack, pbt=None):
        """Return a pytree of coordinate-independent arrays for this topology."""
        return {}

    def __call__(self, pose_stack, term_state, coords_override=None):
        """Return {ScoreType: energy[poses]} without mutating state.

        Use coords_override when supplied; otherwise use pose_stack.coords.
        This evaluation must support jax.jit, differentiation and batching.
        """
        coords = pose_stack.coords if coords_override is None else coords_override
        return self.evaluate(term_state, coords)

    @classmethod
    def evaluate(cls, term_state, coords):
        """Execute a stock term using only explicit prepared-state operands."""
        raise NotImplementedError


def prepared_palette(
    pbt, database, name, prepare, *, settings=(), device=None, sources=()
):
    """Bounded palette preparation keyed by live sources and execution device."""
    key = AnnotationKey.from_sources(
        database,
        *pbt._scoring_sources,
        *sources,
        settings=(pbt.max_n_atoms, pbt.max_n_conn, device, *settings),
    )
    # Device copies must not evict the common host records used to prepare them.
    attribute = "_scoring_preparations" if device is None else "_scoring_device_arrays"
    cache = getattr(pbt, attribute, None)
    if cache is None:
        # Whole-pose and sparse preparation share the finite score-family set.
        cache = (
            PreparationCache()
            if device is None
            else PreparationCache(max_entries=len(TERM_CHANNELS))
        )
        setattr(pbt, attribute, cache)
    return cache.get_or_create(name, key, prepare)
