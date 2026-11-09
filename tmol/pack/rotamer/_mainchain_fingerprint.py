"""Torch device/cache adapter for shared retained-atom copy plans."""

import attr
import numpy
import torch
from typing import Tuple
from tmol.types._array import NDArray
from tmol.types._functional import _validate_args
from tmol.database._patched_chemdb import PatchedChemicalDatabase
from tmol.chemical import RefinedResidueType
from tmol.pose._packed_block_types import PackedBlockTypes
from tmol.score._annotation_cache import _cached_annotation, _store_annotation
from tmol_core.pack.fingerprint import (  # noqa: F401
    AtomFingerprint,  # noqa: F401
    MCFingerprint,  # noqa: F401
    MCFingerprints,  # noqa: F401
    _mc_inds_for_chiral_mc_atom,  # noqa: F401
    _mainchain_region,  # noqa: F401
    create_non_sidechain_fingerprint,
    create_mainchain_fingerprint,
    annotate_residue_type_with_sampler_fingerprints as _annotate_residue_type_with_sampler_fingerprints,  # noqa: E501,F401
    fingerprint_mapping_layout,
    pack_fingerprint_rows,
)


@_validate_args
def _create_non_sidechain_fingerprint(  # noqa: C901
    rt: RefinedResidueType,
    parents: NDArray[numpy.int32][:],
    sc_atoms: NDArray[numpy.int32][:],
    chem_db: PatchedChemicalDatabase,
):
    return create_non_sidechain_fingerprint(rt, parents, sc_atoms, chem_db)


@_validate_args
def _create_mainchain_fingerprint(
    rt: RefinedResidueType, sc_roots: Tuple[str, ...], chem_db: PatchedChemicalDatabase
):
    return create_mainchain_fingerprint(rt, sc_roots, chem_db)


def _find_unique_fingerprints(pbt: PackedBlockTypes):
    rows, sampler_mapping, key = fingerprint_mapping_layout(pbt.active_block_types)
    cached = _cached_annotation(pbt, "_mc_packed_annotation", key)
    if cached is not None:
        if cached.sampler_mapping != sampler_mapping:
            cached = attr.evolve(cached, sampler_mapping=sampler_mapping)
        pbt.mc_fingerprints = cached
        return _store_annotation(pbt, "_mc_packed_annotation", key, cached)

    host = pack_fingerprint_rows(rows, sampler_mapping)

    def to_device(array):
        return torch.tensor(array, dtype=torch.int64, device=pbt.device)

    fingerprints = attr.evolve(
        host,
        atom_mapping=to_device(host.atom_mapping),
        source_atom_mapping=to_device(host.source_atom_mapping),
        source_fingerprint=to_device(host.source_fingerprint),
    )
    pbt.mc_fingerprints = fingerprints
    return _store_annotation(pbt, "_mc_packed_annotation", key, fingerprints)
