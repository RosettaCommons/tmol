"""JAX execution of shared disulfide, metal, histidine and protonation policy."""

import jax
import numpy as np

from tmol_core.io.canonical_ordering import histidine_atom_indices
from tmol_core.io.disulfides import prepare_disulfide_search, resolve_disulfides
from tmol_core.io.metal_detection import (
    declared_metal_sites,
    find_metal_geometries,
    _donor_patches,
    _metal_connection_rows,
)
from tmol_core.io.protonation import (
    prepare_protonation_plan,
    required_donor_rows,
    neutral_terminus_patches,
)

from . import _native
from ._metal import grow_palette


def _protonation_arrays(pbt, co, types):
    plan = prepare_protonation_plan(co, pbt.chem_db, types)
    cached = getattr(pbt, "_protonation_execution", None)
    device = pbt.n_atoms.device
    if cached is None or cached[0] is not plan or cached[1] != device:
        cached = plan, device, jax.device_put(plan.arrays, device)
        object.__setattr__(pbt, "_protonation_execution", cached)
    return cached[2]


def resolve_input_variants(co, pbt, cf, types, coords, present, bonds):
    from ._selection import _rows, _wide

    device = pbt.n_atoms.device

    def run(function, *args):
        return function(*jax.device_put(args, device))

    disulfides = _rows(cf, "disulfides", 3, types.shape, (1, 2))
    plan = prepare_disulfide_search(
        co.cys_inds.cys_co_aa_inds,
        types,
        disulfides if "disulfides" in cf else None,
        cf.get("find_additional_disulfides", True),
        sg_atoms=co.cys_inds.sg_atoms_for_co_cys,
    )
    observed = np.where(present[..., None], coords, np.nan)
    disulfides, variants = resolve_disulfides(
        plan, observed, co.cys_inds.sg_atom_for_co_cys
    )
    disulfides = disulfides.astype(np.int32)
    declared, required = declared_metal_sites(
        cf.get("metal_sites"), cf.get("metal_coordination")
    )
    metal_variants, assignments = find_metal_geometries(
        co,
        pbt.chem_db,
        types,
        observed,
        excluded_donor_residues=variants != 0,
        declared_sites=declared,
        required_donors=required,
        find_additional=cf.get("find_additional_metal_coordination", True),
    )
    variants = np.where(metal_variants != 0, metal_variants, variants)
    his_rows = np.nonzero(np.isin(types, co.his_inds.his_co_aa_inds))
    if len(his_rows[0]):
        atoms = histidine_atom_indices(co.his_inds)
        variants, observed, present = run(
            _native.histidine_coordinates,
            observed,
            types,
            variants,
            _wide(his_rows[0]),
            _wide(his_rows[1]),
            present,
            atoms,
        )
    variants, unresolved = run(
        _native.canonical_protonation,
        types,
        variants,
        present,
        _protonation_arrays(pbt, co, types),
        _wide(bonds),
        _wide(required_donor_rows(assignments)),
    )
    if np.any(np.asarray(unresolved)):
        raise ValueError(
            f"no chemical form can satisfy required metal donors at {np.argwhere(np.asarray(unresolved)).tolist()}"
        )
    explicit = cf.get("protonation_variants")
    if explicit is not None:
        explicit = np.asarray(explicit, np.int32).copy()
        explicit[disulfides[:, 0], disulfides[:, 1]] = -1
        explicit[disulfides[:, 0], disulfides[:, 2]] = -1
        import jax.numpy as jnp

        variants = jnp.where(
            jax.device_put(explicit, device) >= 0,
            jax.device_put(explicit, device),
            variants,
        )
    patches = _donor_patches(
        co, pbt.chem_db, types, assignments
    ) + neutral_terminus_patches(co, pbt.chem_db, types, np.asarray(variants))
    metal_classes = {
        co.restype_io_equiv_classes[types[p, r]] for (p, r), _ in assignments
    }
    pbt = grow_palette(pbt, co, patches, metal_classes)
    return (
        pbt,
        disulfides,
        variants,
        observed,
        present,
        assignments,
        np.asarray(_metal_connection_rows(assignments), np.int32).reshape(-1, 5),
    )
