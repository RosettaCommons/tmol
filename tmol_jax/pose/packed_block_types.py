"""JAX palette of residue atom, bond, connection and ideal-coordinate arrays.

Array fields are pytree leaves; chemistry and residue identities are host
metadata. Connection axes reserve at least two slots, with -1 for absent atoms.
Torsion-specific annotations are prepared by the scoring terms."""

from __future__ import annotations

from functools import cached_property
from typing import Any, Sequence, Tuple

import attrs
import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.types import pytree_dataclass

# PackedBlockTypes                                                             #


@attrs.frozen(eq=False)
class _ResidueContext:
    # Residue collections contain NumPy arrays. Their generated structural
    # equality is not a valid JAX static-metadata comparison; retain identity.
    value: Any


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
    _restype_context: Any = None

    @property
    def device(self):
        return self.n_atoms.device

    @property
    def restype_set(self):
        if self._restype_context is None:
            from tmol_jax.chemical.restypes import ResidueTypeSet

            self._restype_context = _ResidueContext(
                ResidueTypeSet.from_database(self.chem_db)
            )
        return self._restype_context.value

    @cached_property
    def restype_index(self):
        import pandas

        return pandas.Index([restype.name for restype in self.active_block_types])

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
        device = jax.config.jax_default_device or jax.devices()[0]
        arrays = {
            name: jax.device_put(getattr(host, name), device=device)
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
        arrays["atom_is_hydrogen"] = jax.device_put(
            np.asarray(host.atom_is_hydrogen, dtype=bool), device=device
        )
        arrays["bond_indices"] = jax.device_put(
            np.where(host.bond_is_real[..., None], host.bond_indices, 0), device=device
        )
        arrays["conn_atoms"] = jax.device_put(host.conn_atom, device=device)
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
    def from_restype_list(cls, chem_db, restype_set, active_block_types, device):
        """Pack the supplied chemistry and ordered types on a concrete device.

        This accepts the mature constructor's arguments. The supplied residue
        set remains the context; active types are never replaced by defaults.
        """
        from tmol_jax.io.structure import _device

        active_block_types = tuple(active_block_types)
        if not active_block_types:
            raise ValueError("Cannot pack an empty residue-type palette")
        with jax.default_device(_device(device)):
            result = cls.from_restypes(chem_db, active_block_types)
        result._restype_context = _ResidueContext(restype_set)
        return result

    @classmethod
    def default(cls, chem_db: Any | None = None) -> "PackedBlockTypes":
        """Build the shared default molecular palette on the current JAX device.

        Standalone metal sites require an explicitly prepared palette; ordinary
        residues retain all chemical variants and their atom identities.
        """
        from tmol_core.io.canonical_ordering import only_coordinates_a_metal
        from tmol_jax.chemical.patched_chemdb import default_patched_chemdb
        from tmol_jax.chemical.restypes import RefinedResidueType, _refined_from_raw

        if chem_db is None:
            chem_db = default_patched_chemdb()
        selected = tuple(
            res if isinstance(res, RefinedResidueType) else _refined_from_raw(res)
            for res in chem_db.residues
            if not only_coordinates_a_metal(res)
        )
        return cls.from_restypes(chem_db, selected)

    # Lookup helpers                                                      #
    def inds_for_restypes(self, res_types):
        """Resolve ordered residue names, returning -1 for absent types."""
        return self.restype_index.get_indexer([restype.name for restype in res_types])

    def cpu(self):
        """Transfer palette arrays while retaining their exact chemistry context."""
        return jax.device_put(self, jax.devices("cpu")[0])


__all__ = ["PackedBlockTypes"]
