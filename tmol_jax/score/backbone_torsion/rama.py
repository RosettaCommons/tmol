"""Periodic cubic Ramachandran interpolation and host table preparation.

Packed tables include ordinary and prepro variants; pose connectivity selects
the latter when the next residue is proline. Untabulated residue types receive
zero grids. Required parameters come from Torch-free archives; has_tables
reports whether a custom database supplies Rama tables."""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# Local copy of the canonical 20-AA tuple — see ``_aa.py`` for the
# rationale (the tmol_jax.chemical package currently has a circular
# import chain that fires as soon as ``tmol_jax.chemical.*`` is
# imported from here).
from tmol_jax.numeric.bspline import bspline_coefficients
from tmol_jax.score.backbone_torsion._aa import (
    backbone_table_reference_aliases,
    canonical_aa_3letter,
)
from tmol_jax.database.scoring.rama import RamaDatabase
from tmol_jax.score.backbone_torsion.potentials import rama_potential
from tmol_jax.types import Array

# Default table shape used when no rama.zip is available. 36x36 matches
# upstream tmol tables (10 deg spacing over a full 360 deg window).
_DEFAULT_N_PHI = 36
_DEFAULT_N_PSI = 36


def rama_score(
    phi: Array,
    psi: Array,
    aa_type: Array,
    rama_table: Array,
    bbstart: Optional[Array] = None,
    bbstep: Optional[Array] = None,
) -> Array:
    """Bicubic B-spline Ramachandran energy for a single ``(phi, psi)`` sample.

    Parameters
    ----------
    phi, psi
        Scalar JAX arrays (radians). Dihedral angles are expected in
        ``[-pi, pi]`` but the periodic gather inside the B-spline
        evaluator wraps any convention modulo ``2*pi``.
    aa_type
        Scalar integer index into the leading axis of ``rama_table``
        (typically the residue's position in the canonical 20-AA list).
        Negative values zero the result, matching the "no table for
        this residue" convention.
    rama_table
        ``float[n_aa, n_phi, n_psi]`` packed B-spline coefficient grid.
        This is the output of :func:`bspline_coefficients` applied
        separately to each AA plane of the original energy table —
        :class:`RamaDatabaseAccess` handles the prefiltering for you.
    bbstart, bbstep
        Optional ``float[2]`` start and step for the table axes
        (radians). When omitted, defaults to the canonical
        ``[-pi, -pi]`` start with ``2*pi / n_phi`` step — which is what
        the upstream tmol rama tables use.

    Returns
    -------
    jnp.ndarray
        Scalar energy (kcal/mol). Zero if ``aa_type < 0`` or if the
        packed table has zero AAs (e.g. when ``torch`` is unavailable).
    """
    n_phi = rama_table.shape[1]
    n_psi = rama_table.shape[2]

    if bbstart is None:
        bbstart = jnp.array([-jnp.pi, -jnp.pi], dtype=jnp.float32)
    if bbstep is None:
        bbstep = jnp.array(
            [2.0 * jnp.pi / n_phi, 2.0 * jnp.pi / n_psi],
            dtype=jnp.float32,
        )

    # Empty-table short circuit: if the caller is running without the
    # zip payload, return a traced zero so the downstream reduction
    # still jits cleanly.
    if rama_table.shape[0] == 0:
        return jnp.zeros_like(phi)

    return rama_potential(phi, psi, aa_type, rama_table, bbstart, bbstep)


class RamaDatabaseAccess:
    """Pack a :class:`RamaDatabase` into JAX-ready arrays.

    The class:

    * Builds a ``[n_aa, n_phi, n_psi]`` B-spline coefficient grid
      indexed by the canonical 20-AA axis. Missing residues get an
      all-zero plane. Each per-AA plane is pre-filtered via
      :func:`tmol_jax.numeric.bspline.bspline_coefficients` so the
      evaluator in :func:`rama_score` is a pure cubic-B-spline
      interpolation — matching upstream tmol's
      ``ndspline<2, 3>::interpolate``.
    * Stores per-AA ``bbstart``/``bbstep`` (picked from the first table
      the residue matches) so :func:`rama_score` can consult them.

    Parameters
    ----------
    rama_database
        The loaded :class:`RamaDatabase`. May have
        ``rama_tables = ()`` when the zip payload is unavailable; in
        that case :attr:`has_tables` is ``False`` and :attr:`rama_table`
        is an empty ``[0, 36, 36]`` array.

    Attributes
    ----------
    rama_table
        ``jnp.ndarray[n_aa, n_phi, n_psi]`` packed B-spline
        coefficient grid.
    bbstart
        ``jnp.ndarray[2]`` first-table start (radians).
    bbstep
        ``jnp.ndarray[2]`` first-table step (radians).
    aa_to_table_idx
        ``np.ndarray[int32]`` — per-canonical-AA index into the first
        axis of ``rama_table`` for the non-prepro context. ``-1`` when
        no table was available. Retained for backwards compatibility.
    aa_to_table_idx_2
        ``np.ndarray[int32, shape=(n_canonical_aa, 2)]`` — column 0 is
        the non-prepro table index, column 1 is the prepro (upper
        neighbour == PRO) table index. Mirrors upstream tmol's
        ``block_type_rama_table[bt][upper_nbr_is_pro]`` access pattern.
    has_tables
        ``True`` iff at least one residue ended up with a real grid.
    """

    __slots__ = (
        "rama_table",
        "bbstart",
        "bbstep",
        "aa_to_table_idx",
        "aa_to_table_idx_2",
        "reference_to_table_idx_2",
        "has_tables",
        "_source_uniq_id",
    )

    def __init__(self, rama_database: RamaDatabase):
        # Split lookup rows into a non-prepro and a prepro keyed dict.
        # Upstream indexes ``block_type_rama_table[bt][upper_nbr_is_pro]``
        # so the JAX port needs the same 2-column mapping.
        name_to_tid_nonprepro: dict[str, str] = {}
        name_to_tid_prepro: dict[str, str] = {}
        for row in rama_database.rama_lookup:
            if row.res_upper == "PRO":
                name_to_tid_prepro[row.res_middle] = row.table_id
            else:
                # "_" or any other sentinel is the non-prepro context.
                name_to_tid_nonprepro[row.res_middle] = row.table_id

        table_id_to_idx: dict[str, int] = {}
        tables_np: list[np.ndarray] = []
        bbstart_np: Optional[np.ndarray] = None
        bbstep_np: Optional[np.ndarray] = None

        for t in rama_database.rama_tables:
            if t.table_id in table_id_to_idx:
                continue
            table_id_to_idx[t.table_id] = len(tables_np)
            tables_np.append(np.asarray(t.table, dtype=np.float32))
            if bbstart_np is None:
                bbstart_np = np.asarray(t.bbstart, dtype=np.float32)
                bbstep_np = np.asarray(t.bbstep, dtype=np.float32)

        n_aa = len(canonical_aa_3letter)
        # Column 0: non-prepro table, column 1: prepro table.
        aa_to_table_idx_2 = np.full((n_aa, 2), -1, dtype=np.int32)

        if tables_np:
            stacked = np.stack(tables_np, axis=0).astype(np.float32)
            for i, aa in enumerate(canonical_aa_3letter):
                tid_nonprepro = name_to_tid_nonprepro.get(aa)
                tid_prepro = name_to_tid_prepro.get(aa)
                if tid_nonprepro is not None:
                    j = table_id_to_idx.get(tid_nonprepro)
                    if j is not None:
                        aa_to_table_idx_2[i, 0] = j
                if tid_prepro is not None:
                    j = table_id_to_idx.get(tid_prepro)
                    if j is not None:
                        aa_to_table_idx_2[i, 1] = j
            # Pre-filter each AA plane into B-spline coefficients. The
            # prefilter is a pair of separable 1-D causal / anti-causal
            # IIR filters (Thévenaz-Blu-Unser, periodic boundary). We
            # run the JAX filter vmapped over the leading AA axis so
            # the returned coefficients match upstream bit-for-bit.
            coeffs = jax.vmap(lambda plane: bspline_coefficients(plane, degree=3))(
                jnp.asarray(stacked)
            )
            self.rama_table = coeffs
            self.bbstart = jnp.asarray(bbstart_np)
            self.bbstep = jnp.asarray(bbstep_np)
            self.has_tables = True
        else:
            # Torch unavailable / zip missing. Return an empty table so
            # the energy term can still build and report 0 kcal/mol.
            self.rama_table = jnp.zeros(
                (0, _DEFAULT_N_PHI, _DEFAULT_N_PSI), dtype=jnp.float32
            )
            self.bbstart = jnp.array([-jnp.pi, -jnp.pi], dtype=jnp.float32)
            self.bbstep = jnp.array(
                [
                    2.0 * jnp.pi / _DEFAULT_N_PHI,
                    2.0 * jnp.pi / _DEFAULT_N_PSI,
                ],
                dtype=jnp.float32,
            )
            self.has_tables = False

        # Backwards-compat: a 1-D non-prepro view as the primary
        # ``aa_to_table_idx``. Callers that don't consult the upper-
        # neighbour identity still see the non-prepro mapping.
        references = set(name_to_tid_nonprepro) | set(name_to_tid_prepro)
        self.reference_to_table_idx_2 = {
            reference: np.array(
                [
                    table_id_to_idx.get(name_to_tid_nonprepro.get(reference), -1),
                    table_id_to_idx.get(name_to_tid_prepro.get(reference), -1),
                ],
                dtype=np.int32,
            )
            for reference in references
        }
        for alias, source in backbone_table_reference_aliases.items():
            if source in self.reference_to_table_idx_2:
                self.reference_to_table_idx_2[alias] = self.reference_to_table_idx_2[
                    source
                ].copy()
        self.aa_to_table_idx = aa_to_table_idx_2[:, 0].copy()
        self.aa_to_table_idx_2 = aa_to_table_idx_2
        self._source_uniq_id = rama_database.uniq_id

    # Convenience                                                     #

    @property
    def n_aa(self) -> int:
        """Number of AA-specific planes in :attr:`rama_table`."""
        return int(self.rama_table.shape[0])

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.rama_table.shape)  # type: ignore[return-value]


__all__ = [
    "RamaDatabaseAccess",
    "rama_score",
]
