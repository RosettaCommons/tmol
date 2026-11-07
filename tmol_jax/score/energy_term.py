"""Interface for a term's topology setup and differentiable energy evaluation."""

from abc import ABC, abstractmethod

from tmol_core.cache import AnnotationKey, PreparationCache


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


def prepared_palette(pbt, database, name, prepare, *, settings=()):
    """Owner-local immutable host preparation, shared by terms using a palette."""
    key = AnnotationKey.from_sources(
        database,
        *pbt.active_block_types,
        settings=(pbt.max_n_atoms, pbt.max_n_conn, *settings),
    )
    cache = getattr(pbt, "_scoring_preparations", None)
    if cache is None:
        cache = pbt._scoring_preparations = PreparationCache()
    return cache.get_or_create(name, key, prepare)
