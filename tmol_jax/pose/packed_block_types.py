"""JAX palette of residue atom, bond, connection and ideal-coordinate arrays.

Array fields are pytree leaves; chemistry and residue identities are host
metadata. Connection axes reserve at least two slots, with -1 for absent atoms.
Torsion-specific annotations are prepared by the scoring terms."""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import attrs
import jax.numpy as jnp
import numpy as np

from tmol_jax.types import pytree_dataclass

# PackedBlockTypes                                                             #


@pytree_dataclass
@attrs.define(slots=False)
class PackedBlockTypes:
    """A flat, array-backed "palette" of residue types.

    Given a list of ``RefinedResidueType``s (the *active* block types
    the caller cares about), this class produces packed 2-D / 3-D
    arrays with the leading dimension being block-type index. Unused
    slots are zero-filled (for int/float arrays) or ``False`` (for
    bool masks); the ``n_atoms`` / ``n_bonds`` / ``n_conn`` arrays
    record the true per-type counts.

    Array fields are JAX arrays and therefore act as pytree leaves;
    Python-typed fields (``chem_db``, ``active_block_types``,
    ``atom_names``, the ``max_*`` / ``n_types`` ints) are pytree aux.
    """

    chem_db: Any
    active_block_types: Tuple[Any, ...]
    atom_names: Tuple[Tuple[str, ...], ...]

    n_types: int
    max_n_atoms: int
    max_n_bonds: int
    max_n_conn: int

    n_atoms: jnp.ndarray  # [n_types] int32
    atom_is_real: jnp.ndarray  # [n_types, max_n_atoms] bool
    atom_elements: jnp.ndarray  # [n_types, max_n_atoms] int32
    atom_types: jnp.ndarray  # [n_types, max_n_atoms] int32
    atom_is_hydrogen: jnp.ndarray  # [n_types, max_n_atoms] bool
    atom_is_backbone: jnp.ndarray  # [n_types, max_n_atoms] bool

    bond_indices: jnp.ndarray  # [n_types, max_n_bonds, 2] int32
    n_bonds: jnp.ndarray  # [n_types] int32

    ideal_coords: jnp.ndarray  # [n_types, max_n_atoms, 3] float32

    conn_atoms: jnp.ndarray  # [n_types, max_n_conn] int32
    n_conn: jnp.ndarray  # [n_types] int32

    intrares_path_distance: jnp.ndarray
    # [n_types, max_n_atoms, max_n_atoms] int32

    # Core builder                                                        #
    @classmethod
    def from_restypes(
        cls,
        chem_db: Any,
        active_types: Sequence[Any],
    ) -> "PackedBlockTypes":
        """Transfer the common host topology onto the current JAX device."""
        from tmol_core.chemical.packed import PackedBlockTypesData

        host = PackedBlockTypesData.from_restypes(chem_db, active_types, min_n_conn=2)
        arrays = {
            name: jnp.asarray(getattr(host, name))
            for name in (
                "n_atoms",
                "atom_is_real",
                "atom_elements",
                "atom_types",
                "atom_is_backbone",
                "n_bonds",
                "ideal_coords",
                "n_conn",
                "intrares_path_distance",
            )
        }
        arrays["atom_is_hydrogen"] = jnp.asarray(host.atom_is_hydrogen, dtype=bool)
        arrays["bond_indices"] = jnp.asarray(
            np.where(host.bond_is_real[..., None], host.bond_indices, 0)
        )
        arrays["conn_atoms"] = jnp.asarray(host.conn_atom)
        return cls(
            chem_db=chem_db,
            active_block_types=host.active_block_types,
            atom_names=host.atom_names,
            n_types=host.n_types,
            max_n_atoms=host.max_n_atoms,
            max_n_bonds=host.max_n_bonds,
            max_n_conn=host.max_n_conn,
            **arrays,
        )

    # Convenience constructors                                            #
    @classmethod
    def default(cls, chem_db: Any | None = None) -> "PackedBlockTypes":
        """Build the default patched protein residue palette.

        JAX's current device context determines array placement. Chemistry and
        residue identities remain host metadata."""
        # Lazy imports (see module docstring for why).
        if chem_db is None:
            from tmol_jax.chemical.patched_chemdb import (
                default_patched_chemdb,
            )

            chem_db = default_patched_chemdb()

        from tmol_jax.chemical.restypes import RefinedResidueType  # noqa: F401

        canonical_aas = (
            "ALA",
            "CYS",
            "ASP",
            "GLU",
            "PHE",
            "GLY",
            "HIS",
            "ILE",
            "LYS",
            "LEU",
            "MET",
            "ASN",
            "PRO",
            "GLN",
            "ARG",
            "SER",
            "THR",
            "VAL",
            "TRP",
            "TYR",
        )

        # Group residues by base ``name3``; we collect every variant
        # whose ``name3`` matches one of the canonical AAs.
        by_name3: dict = {}
        for res in chem_db.residues:
            n3 = getattr(res, "name3", None)
            name = getattr(res, "name", None)
            if n3 is None or name is None:
                continue
            by_name3.setdefault(n3, []).append(res)

        def _variant_sort_key(res: Any) -> tuple:
            """Sort key: base residue first, then by variant-tag tuple.

            We rely on the residue's ``name`` being ``base`` (no colon)
            for the base variant, and ``base:<display_name>[:...]`` for
            patched ones. The key orders:

              - bases first (empty variant tuple),
              - then single variants sorted alphabetically,
              - then double+ variants sorted alphabetically.
            """
            name = getattr(res, "name", "")
            parts = name.split(":")
            # Everything after the base name is a sorted tuple of
            # display names (the chemdb already sorts them).
            variants = tuple(parts[1:])
            return (len(variants), variants, name)

        selected: List[Any] = []
        for aa in canonical_aas:
            variants = sorted(by_name3.get(aa, []), key=_variant_sort_key)
            for res in variants:
                if not isinstance(res, RefinedResidueType):
                    from tmol_jax.chemical.restypes import _refined_from_raw

                    res = _refined_from_raw(res)
                selected.append(res)

        return cls.from_restypes(chem_db, selected)

    # Lookup helpers                                                      #


__all__ = ["PackedBlockTypes"]
