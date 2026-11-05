"""JAX pytrees for canonical Lennard-Jones/Lazaridis-Karplus parameters.

Atom-type arrays are gathered into pose order during setup. Boolean chemical
flags determine hydrogen-bond sigma overrides; missing/padded types are masked."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.types import as_jnp, pytree_dataclass

# Containers                                                                   #


@pytree_dataclass
class LJTypeParams:
    """Per-atom-type LJ + LK parameters, indexed by atom-type index.

    Shapes: every field is a 1-D JAX array of length ``n_atom_types``.
    The boolean fields are stored as ``bool_`` arrays rather than floats
    so ``jnp.where`` works naturally; they broadcast against float
    arrays exactly as expected.

    Fields
    ------
    lj_radius
        Atom radius (A). Used to form ``sigma`` via ``lj_sigma``.
    lj_wdepth
        Well-depth (kcal/mol). ``epsilon`` is the geometric mean over a
        pair ``sqrt(w_i * w_j)``.
    is_donor
        True for atoms acting as H-bond donors (heavy-atom side).
    is_acceptor
        True for atoms acting as H-bond acceptors.
    is_hydroxyl
        True for hydroxyl donors; selects
        ``global.lj_hbond_OH_donor_dis``.
    is_polarh
        True for polar hydrogens.
    lk_volume
        LK solvent-accessible volume.
    lk_dgfree
        LK desolvation free energy (kcal/mol).
    lk_lambda
        LK Gaussian width parameter.
    """

    lj_radius: jax.Array
    lj_wdepth: jax.Array
    is_donor: jax.Array
    is_acceptor: jax.Array
    is_hydroxyl: jax.Array
    is_polarh: jax.Array
    is_carbon_lk: jax.Array
    lk_volume: jax.Array
    lk_dgfree: jax.Array
    lk_lambda: jax.Array

    # Constructors                                                       #
    @classmethod
    def from_database(
        cls, ljlk_db: Any, chem_db: Any, *, active_types=()
    ) -> "LJTypeParams":
        """Transfer common parameters after validating the active chemistry."""
        from tmol_core.scoring.atom_types import atom_type_parameters
        from tmol_core.scoring.ljlk import ljlk_type_parameters, LJLK_FIELDS

        names, flags = atom_type_parameters(chem_db.atom_types)
        host = ljlk_type_parameters(names[:-1], ljlk_db)
        for bt in active_types:
            host.validate_block_type(bt)
        # Only unused/padding entries may be missing. Safe intermediates keep
        # masked pair derivatives finite without inventing real-atom parameters.
        defaults = np.asarray([1.7, 0, 0, 1, 0], dtype=np.float32)
        values = np.where(np.isfinite(host.values), host.values, defaults)
        return cls(
            **{name: jnp.asarray(values[:, i]) for i, name in enumerate(LJLK_FIELDS)},
            **{
                name: jnp.asarray(flags[name][:-1])
                for name in ("is_donor", "is_acceptor", "is_hydroxyl", "is_polarh")
            },
            is_carbon_lk=jnp.asarray(host.is_carbon_lk),
        )

    # Gather                                                             #
    def gather(self, idx: jax.Array) -> "LJTypeParams":
        """Gather per-atom-type parameters along the 0-th axis.

        Returns a new :class:`LJTypeParams` where every field has been
        indexed with ``idx``. Out-of-range indices (``idx < 0``) are
        clamped to 0; callers should mask the result with
        ``idx >= 0`` if they care about padding.
        """
        safe = jnp.where(idx >= 0, idx, 0)
        return jax.tree.map(lambda field: field[safe], self)


@pytree_dataclass
class LJLKGlobal:
    """LJ / LK global parameters (one scalar each).

    ``max_dis`` is the far cpoly cutoff (``cpoly_dmax``) and
    ``lj_dlin_sigma_factor`` is the multiplier in ``d_lin = sigma *
    factor`` that sets the linear-extrapolation crossover on the LJ
    repulsive wall. Both are threaded through to the potentials so a
    non-default YAML override changes behaviour without code edits.
    The three H-bond sigma overrides select per-pair ``sigma`` in
    :func:`tmol_jax.score.ljlk.potentials.lj_sigma`.
    """

    lj_hbond_dis: jax.Array
    lj_hbond_OH_donor_dis: jax.Array
    lj_hbond_hdis: jax.Array
    max_dis: jax.Array
    lj_dlin_sigma_factor: jax.Array

    @classmethod
    def from_database(cls, ljlk_db: Any) -> "LJLKGlobal":
        gp = ljlk_db.global_parameters
        return cls(
            lj_hbond_dis=as_jnp(gp.lj_hbond_dis, dtype=jnp.float32),
            lj_hbond_OH_donor_dis=as_jnp(gp.lj_hbond_OH_donor_dis, dtype=jnp.float32),
            lj_hbond_hdis=as_jnp(gp.lj_hbond_hdis, dtype=jnp.float32),
            max_dis=as_jnp(gp.max_dis, dtype=jnp.float32),
            lj_dlin_sigma_factor=as_jnp(gp.lj_dlin_sigma_factor, dtype=jnp.float32),
        )


__all__ = ["LJTypeParams", "LJLKGlobal"]
