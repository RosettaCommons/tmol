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
from tmol_core.io.disulfides import prepare_disulfide_search, resolve_disulfides

from . import _native

logger = logging.getLogger(__name__)


def _wide(values):
    values = np.ascontiguousarray(values, dtype=np.int64)
    return values.view(np.uint8).reshape((*values.shape, 8))


def _rows(cf, name, width, shape, endpoints):
    value = cf.get(name)
    values = np.asarray(() if value is None else value, dtype=np.int32).reshape(
        -1, width
    )
    if len(values):
        p = values[:, 0]
        if np.any((p < 0) | (p >= shape[0])):
            raise ValueError(f"{name}: pose index outside input")
        for col in endpoints:
            if np.any((values[:, col] < 0) | (values[:, col] >= shape[1])):
                raise ValueError(f"{name}: residue index outside input")
    return values


def select_and_connect(co, pbt, prepared, cf, types, chains, coords, present, gaps):
    """Materialize the shared discrete input decisions on the palette's device.

    This preparation boundary intentionally inspects results on the host; its
    individual native stages support jit/vmap, while the complete loader does
    not promise tracing through discrete chemistry or topology decisions.
    """
    device = pbt.n_atoms.device

    def put(values):
        return jax.device_put(values, device)

    def run(function, *values, **options):
        return function(*put(values), **options)

    real = types >= 0
    for name in ("metal_sites", "metal_coordination", "protonation_variants"):
        if cf.get(name) is not None:
            raise NotImplementedError(
                f"canonical {name} resolution is not implemented; use exact residue blocks and named ports"
            )
    disulfides = _rows(cf, "disulfides", 3, types.shape, (1, 2))
    plan = prepare_disulfide_search(
        co.cys_inds.cys_co_aa_inds,
        types,
        disulfides if "disulfides" in cf else None,
        cf.get("find_additional_disulfides", True),
    )
    disulfides, variants = resolve_disulfides(
        plan,
        np.where(present[..., None], coords, np.nan),
        co.cys_inds.sg_atom_for_co_cys,
    )
    disulfides = disulfides.astype(np.int32)
    his_rows = np.nonzero(np.isin(types, co.his_inds.his_co_aa_inds))
    if len(his_rows[0]):
        inds = co.his_inds
        atoms = np.array(
            [
                [
                    getattr(inds, "his_" + name + "_in_co")
                    for name in ("ND1", "NE2", "HD1", "HE2", "HN", "NH", "NN", "CG")
                ]
            ],
            np.int32,
        )
        variant_array, mapping, _ = run(
            _native.histidine_state,
            coords,
            types,
            variants,
            _wide(his_rows[0]),
            _wide(his_rows[1]),
            present,
            atoms,
        )
        variants = np.asarray(variant_array)
        mapping = np.asarray(mapping).copy().view(np.int64).reshape(present.shape)
        coords = np.take_along_axis(coords, mapping[..., None], axis=2)
        present = np.take_along_axis(present, mapping, axis=2)
    cycles = _rows(cf, "cyclic_bonds", 3, types.shape, (1, 2))
    if cf.get("find_additional_cyclic_closures", True):
        ports = co.polymer_conn_inds
        partners = np.asarray(
            run(
                _native.canonical_cycles,
                types,
                chains,
                np.asarray(ports.up_atom_for_co_restype, np.int32),
                np.asarray(ports.down_atom_for_co_restype, np.int32),
                np.where(present[..., None], coords, np.nan),
                cycles,
            )
        )
        p, r = np.nonzero(partners >= 0)
        cycles = np.concatenate(
            (cycles, np.stack((p, r, partners[p, r]), axis=1).astype(np.int32))
        )
    bonds = _rows(cf, "covalent_bonds", 5, types.shape, (1, 3))
    explicit_ports = np.zeros((*types.shape, 2), bool)
    if len(bonds):
        for col in (1, 3):
            p, r, a = bonds[:, 0], bonds[:, col], bonds[:, col + 1]
            if np.any(~real[p, r]) or np.any((a < 0) | (a >= present.shape[2])):
                raise ValueError("covalent_bonds must refer to real canonical atoms")
            ports = prepared.co_polymer_connection_atoms[types[p, r]]
            rows, sides = np.nonzero(a[:, None] == ports)
            explicit_ports[p[rows], r[rows], sides] = True
    metadata = put(
        (
            prepared.var_combo_candidate_bt_index.astype(np.int32),
            prepared.var_combo_is_real_candidate,
            prepared.bt_canonical_atom_is_absent,
            prepared.bt_non_term_patch_added_canonical_atom_is_present,
            prepared.bt_is_non_default_terminus,
        )
    )
    termini, first, last, polymeric = run(
        _native.canonical_chain_endings,
        chains,
        types,
        gaps,
        real,
        prepared.co_equiv_class_is_polymeric,
        prepared.co_polymer_connection_atoms.astype(np.int32),
        cycles,
        explicit_ports,
    )
    dpresent, dtypes, dvariants, dgaps, dreal = put(
        (present, types, variants, gaps, real)
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
    conjugations = prepare_conjugations(pbt.active_block_types)
    covalent = prepare_covalent_connections(
        co, pbt.active_block_types, conjugations, selected, bonds
    )
    # Explicit prepared metal connections use the same structural port policy.
    # Geometry inference and donor patch generation are guarded above.
    metal = prepare_metal_connections(
        co,
        pbt.active_block_types,
        conjugations,
        covalent.block_types,
        _rows(cf, "metal_connections", 5, types.shape, (1, 3)),
    )
    for message in (*covalent.warnings, *metal.warnings):
        logger.warning(message)
    selected = metal.block_types.astype(np.int32)
    edges = np.asarray(covalent.connections + metal.connections, np.int32).reshape(
        -1, 5
    )
    block_types = pbt.active_block_types
    connections, status = run(
        _native.canonical_connections,
        selected,
        polymeric,
        first,
        last,
        gaps,
        np.array([bt.up_connection_ind for bt in block_types], np.int32),
        np.array([bt.down_connection_ind for bt in block_types], np.int32),
        np.array(
            [bt.connection_to_cidx.get("dslf", -1) for bt in block_types], np.int32
        ),
        disulfides,
        cycles,
        edges,
        max_connections=pbt.max_n_conn,
    )
    if np.any(np.asarray(status)):
        raise ValueError(
            f"invalid canonical connection topology (per-pose status {np.asarray(status).tolist()})"
        )
    return selected, np.asarray(connections), coords, present
