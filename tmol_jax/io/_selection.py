"""JAX execution of common canonical chemistry, selection and connection policy."""

import logging

import jax
import numpy as np

from tmol_core.chemical import NEUTRAL_TERMINUS_VAR_BASE
from tmol_core.io.canonical import prepare_conjugations, candidate_score_thresholds
from tmol_core.io.connections import (
    prepare_covalent_connections,
    prepare_metal_connections,
)

from . import _native

logger = logging.getLogger(__name__)


def _wide(values):
    values = np.ascontiguousarray(values, dtype=np.int64)
    return values.view(np.uint8).reshape((*values.shape, 8))


def _rows(cf, name, width, shape, endpoints):
    value = cf.get(name)
    source = np.asarray(() if value is None else value)
    if source.size and (
        source.dtype.kind not in "iu"
        or np.any(source > np.iinfo(np.int32).max)
        or (source.dtype.kind == "i" and np.any(source < np.iinfo(np.int32).min))
    ):
        raise ValueError(f"{name} requires integer indices within the int32 range")
    values = source.astype(np.int32).reshape(-1, width)
    if len(values):
        p = values[:, 0]
        if np.any((p < 0) | (p >= shape[0])):
            raise ValueError(f"{name}: pose index outside input")
        for col in endpoints:
            if np.any((values[:, col] < 0) | (values[:, col] >= shape[1])):
                raise ValueError(f"{name}: residue index outside input")
    return values


def _execution_metadata(pbt, prepared):
    """One device snapshot for the current source-identity-keyed preparation."""
    device = pbt.n_atoms.device
    cached = getattr(pbt, "_canonical_execution_metadata", None)
    if cached is not None and cached[0] is prepared and cached[1] == device:
        return cached[2:]
    types = pbt.active_block_types
    arrays = jax.device_put(
        (
            (
                prepared.var_combo_candidate_bt_index.astype(np.int32),
                prepared.var_combo_is_real_candidate,
                prepared.bt_canonical_atom_is_absent,
                prepared.bt_non_term_patch_added_canonical_atom_is_present,
                prepared.bt_is_non_default_terminus,
            ),
            prepared.co_equiv_class_is_polymeric,
            prepared.co_polymer_connection_atoms.astype(np.int32),
            prepared.co_polymer_connection_atoms[:, 1].astype(np.int32),
            prepared.co_polymer_connection_atoms[:, 0].astype(np.int32),
            np.array([bt.up_connection_ind for bt in types], np.int32),
            np.array([bt.down_connection_ind for bt in types], np.int32),
            np.array([bt.connection_to_cidx.get("dslf", -1) for bt in types], np.int32),
        ),
        device,
    )
    conjugations = prepare_conjugations(types)
    object.__setattr__(
        pbt, "_canonical_execution_metadata", (prepared, device, arrays, conjugations)
    )
    return arrays, conjugations


def select_and_connect(co, pbt, cf, types, chains, coords, present, gaps):
    """Materialize the shared discrete input decisions on the palette's device.

    This preparation boundary intentionally inspects results on the host; its
    individual native stages support jit/vmap, while the complete loader does
    not promise tracing through discrete chemistry or topology decisions.
    """
    device = pbt.n_atoms.device

    def put(values):
        return jax.device_put(values, device)

    from ._variants import resolve_input_variants
    from .pose_stack_construction import _canonical_metadata

    bonds = _rows(cf, "covalent_bonds", 5, types.shape, (1, 3))
    pbt, disulfides, variants, coords, present, assignments, metal_rows = (
        resolve_input_variants(co, pbt, cf, types, coords, present, bonds)
    )
    prepared, _ = _canonical_metadata(pbt, co)
    (
        metadata,
        class_polymeric,
        class_ports,
        class_up,
        class_down,
        up,
        down,
        dslf,
    ), conjugations = _execution_metadata(pbt, prepared)
    real = types >= 0
    cycles = _rows(cf, "cyclic_bonds", 3, types.shape, (1, 2))
    explicit_ports = np.zeros((*types.shape, 2), bool)
    if len(bonds):
        for col in (1, 3):
            p, r, a = bonds[:, 0], bonds[:, col], bonds[:, col + 1]
            if np.any(~real[p, r]) or np.any((a < 0) | (a >= present.shape[2])):
                raise ValueError("covalent_bonds must refer to real canonical atoms")
            ports = prepared.co_polymer_connection_atoms[types[p, r]]
            rows, sides = np.nonzero(a[:, None] == ports)
            explicit_ports[p[rows], r[rows], sides] = True
    dtypes, dchains, dcoords, dpresent, dgaps, dreal, dvariants, dcycles, dexplicit = (
        put(
            (
                types,
                chains,
                coords,
                present,
                gaps,
                real,
                variants,
                cycles,
                explicit_ports,
            )
        )
    )
    if cf.get("find_additional_cyclic_closures", True):
        partners = np.asarray(
            _native.canonical_cycles(
                dtypes,
                dchains,
                class_up,
                class_down,
                dcoords,
                dcycles,
            )
        )
        p, r = np.nonzero(partners >= 0)
        if len(p):
            cycles = np.concatenate(
                (cycles, np.stack((p, r, partners[p, r]), axis=1).astype(np.int32))
            )
            dcycles = put(cycles)
    termini, first, last, polymeric = _native.canonical_chain_endings(
        dchains,
        dtypes,
        dgaps,
        dreal,
        class_polymeric,
        class_ports,
        dcycles,
        dexplicit,
    )
    termini = _native.canonical_atom_termini(
        dpresent,
        dtypes,
        dvariants,
        termini,
        dgaps & polymeric[..., None],
        dreal,
        metadata[:3],
    )
    dvariants = _native.canonical_variants(
        dtypes, dvariants, termini, metadata[1], neutral_base=NEUTRAL_TERMINUS_VAR_BASE
    )
    selected, score, _ = _native.canonical_candidates(
        dpresent, dreal, dtypes, termini, dvariants, metadata
    )
    selected = np.asarray(selected)
    scores = np.asarray(score).copy().view(np.int64).reshape(types.shape)
    warning, failure = candidate_score_thresholds(co.max_n_canonical_atoms)
    failed = real & ((selected < 0) | (scores >= failure))
    if failed.any():
        rows = [
            (int(p), int(r), co.restype_io_equiv_classes[types[p, r]])
            for p, r in zip(*np.nonzero(failed))
        ]
        raise ValueError(
            f"no acceptable chemical variant for canonical residues {rows}"
        )
    if np.any(real & (scores >= warning)):
        logger.warning(
            "canonical input contains atoms absent from the best available chemical variant at %s",
            np.argwhere(real & (scores >= warning)).tolist(),
        )
    covalent = prepare_covalent_connections(
        co, pbt.active_block_types, conjugations, selected, bonds
    )
    metal = prepare_metal_connections(
        co,
        pbt.active_block_types,
        conjugations,
        covalent.block_types,
        np.concatenate(
            (metal_rows, _rows(cf, "metal_connections", 5, types.shape, (1, 3)))
        ),
    )
    if covalent.warnings or metal.warnings:
        raise ValueError(
            "canonical declared connections could not be retained: "
            + "; ".join((*covalent.warnings, *metal.warnings))
        )
    selected = metal.block_types.astype(np.int32)
    edges = np.asarray(covalent.connections + metal.connections, np.int32).reshape(
        -1, 5
    )
    dselected, ddisulfides, dedges = put((selected, disulfides, edges))
    connections, status = _native.canonical_connections(
        dselected,
        polymeric,
        first,
        last,
        dgaps,
        up,
        down,
        dslf,
        ddisulfides,
        dcycles,
        dedges,
        max_connections=pbt.max_n_conn,
    )
    if np.any(np.asarray(status)):
        raise ValueError(
            f"invalid canonical connection topology (per-pose status {np.asarray(status).tolist()})"
        )
    return (
        selected,
        np.asarray(connections),
        np.asarray(coords),
        np.asarray(present),
        pbt,
        assignments,
    )
