"""Numerical boundary of ligand distance-geometry preparation.

Chemistry, distance bounds, random seeds and restart selection remain shared.
Executors return owned NumPy coordinates and retain the mature refinement
objectives, precision and optimization settings. Execution objects are hashable
identities because seeded conformers are cached separately for each executor.
"""

from typing import ContextManager, Protocol

import numpy as np


class ConformerExecution(Protocol):
    def flush_denormals(self) -> ContextManager[None]:
        """Scope the executor's denormal policy to conformer refinement."""
        ...

    def planar_refine(self, coordinates, pair_targets, max_iters: int) -> np.ndarray:
        """Refine float64 planar coordinates against exact pair distances."""
        ...

    def chiral_anneal(
        self,
        coordinates,
        initial_lower,
        initial_upper,
        lower,
        upper,
        components,
        chirals,
        rng_seed: int,
    ) -> np.ndarray:
        """Run the float32 four-dimensional chiral annealing stages."""
        ...

    def stress_refine(
        self,
        coordinates,
        initial_lower,
        initial_upper,
        lower,
        upper,
        components,
        chirals,
    ) -> np.ndarray:
        """Run the float32 final three-dimensional stress refinement."""
        ...
