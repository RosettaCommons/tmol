"""Context construction preserves observed atoms and differentiable geometry."""

from contextlib import nullcontext

import numpy as np
import pytest

from .molecules import DOMAIN_CASES


@pytest.mark.parametrize(
    "case", ("ligand", "noncanonical", "connected_noncanonical", "modified_dna")
)
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_context_heavy_atom_coordinates_and_derivatives(backend, case, dtype):
    from tmol_core.io.missing_atoms import prepare_leaf_atom_flags

    b = backend.molecule(DOMAIN_CASES[case], terms=("ref",))
    pbt = b.pose.packed_block_types
    types, offsets, connections = [
        b.numpy(a)
        for a in (
            b.pose.block_type_ind,
            b.pose.block_coord_offset,
            b.pose.inter_residue_connections,
        )
    ]
    is_h = b.numpy(pbt.atom_is_hydrogen).astype(bool)
    leaf = prepare_leaf_atom_flags(
        pbt.active_block_types, is_h, pbt.max_n_atoms
    ).is_leaf_atom
    coords = b.coords().astype(dtype)
    missing = np.zeros(coords.shape[:2], bool)
    bonds = []
    for p, r in np.argwhere(types >= 0):
        t, start = int(types[p, r]), int(offsets[p, r])
        residue = pbt.active_block_types[t]
        # Actual non-leaf heavy atoms require context, not hydrogen-only frames.
        candidates = np.flatnonzero(
            ~is_h[t, : residue.n_atoms] & ~leaf[t, : residue.n_atoms]
        )
        assert len(candidates)
        atom = int(candidates[-1])
        missing[p, start + atom] = True
        ideal = residue.ideal_coords[residue.at_to_icoor_ind].astype(np.float64)
        neighbors = [int(j) for i, j in residue.bond_indices if i == atom]
        bonds.append(
            (
                p,
                start + atom,
                [
                    (start + j, np.linalg.norm(ideal[atom] - ideal[j]))
                    for j in neighbors
                ],
            )
        )
    coords[missing] = 17.25
    precision = (
        b.module.enable_x64(dtype == np.float64)
        if b.framework == "jax"
        else nullcontext()
    )
    with precision:

        def array(x):
            return (
                b.module.device_put(x, b.device)
                if b.framework == "jax"
                else b.module.as_tensor(x, device=b.device)
            )

        x = array(coords)
        if b.framework == "jax":
            from tmol_jax.io._context import (
                prepare_context_completion,
                execute_context_plan,
            )

            out, built, plans = prepare_context_completion(
                b.pose.replace(coords=x), missing, missing
            )

            def run(value):
                return execute_context_plan(value, plans)

            mapped = b.module.jit(b.module.vmap(run))(b.module.numpy.stack((x, x)))
            np.testing.assert_allclose(
                b.numpy(mapped[0]),
                b.numpy(out),
                rtol=2e-6 if dtype == np.float32 else 2e-12,
                atol=2e-6 if dtype == np.float32 else 2e-12,
            )
        else:
            from tmol_core.io.context import build_missing_by_context
            from tmol.io.details._build_missing_context import _place_differentiably

            out, plans = x.clone(), []

            def place(build):
                plans.append(build)
                positions = _place_differentiably(
                    build.steps,
                    build.chosen,
                    out,
                    build.pose,
                    build.s,
                    build.known_atoms,
                    build.node_global,
                )
                return {a: b.numpy(v).astype(np.float64) for a, v in positions.items()}

            built = build_missing_by_context(
                pbt.active_block_types,
                {a.name: a for a in pbt.chem_db.atom_types},
                b.numpy(pbt.conn_atom),
                coords.astype(np.float64),
                missing,
                missing,
                offsets,
                types,
                connections,
                place,
            )

            def run(value):
                result = value.clone()
                for build in plans:
                    _place_differentiably(
                        build.steps,
                        build.chosen,
                        result,
                        build.pose,
                        build.s,
                        build.known_atoms,
                        build.node_global,
                    )
                return result

        if dtype == np.float32:
            _check_public_context_input(b, case, missing, out)

        np.testing.assert_array_equal(built, missing)
        result = b.numpy(out)
        assert np.isfinite(result).all() and result.dtype == dtype
        np.testing.assert_array_equal(result[~missing], coords[~missing])
        np.testing.assert_array_equal(b.numpy(x), coords)
        for p, atom, neighbors in bonds:
            errors = [
                abs(np.linalg.norm(result[p, atom] - result[p, other]) - length)
                for other, length in neighbors
            ]
            assert min(errors) < (3e-6 if dtype == np.float32 else 3e-12)
        weights = array(
            np.sin(np.arange(coords.size) * 0.37).reshape(coords.shape).astype(dtype)
        )
        direction = (
            np.cos(np.arange(coords.size) * 0.23).reshape(coords.shape).astype(dtype)
        )
        direction /= np.linalg.norm(direction)
        if b.framework == "jax":
            gradient = b.module.jit(
                b.module.grad(lambda value: (run(value) * weights).sum())
            )
        else:

            def gradient(value):
                value = value.clone().requires_grad_(True)
                return b.module.autograd.grad((run(value) * weights).sum(), value)[0]

        g = b.numpy(gradient(x))
        np.testing.assert_array_equal(g[missing], 0)
        assert np.isfinite(g).all() and np.linalg.norm(g) > 0
        step = 2e-3 if dtype == np.float32 else 1e-5
        plus, minus = [
            b.numpy(run(array(coords + sign * step * direction))).astype(np.float64)
            for sign in (1, -1)
        ]
        finite_difference = (
            (plus - minus) * b.numpy(weights).astype(np.float64)
        ).sum() / (2 * step)
        np.testing.assert_allclose(
            (g * direction).sum(),
            finite_difference,
            rtol=3e-3 if dtype == np.float32 else 2e-6,
            atol=3e-3 if dtype == np.float32 else 2e-6,
        )

    _check_explicit_residue_reconstruction(backend, case, dtype)


def _check_explicit_residue_reconstruction(backend, case, dtype):
    if backend.framework != "jax":
        return
    import dataclasses
    from tmol_jax.io import ResidueBlock, pose_stack_from_residue_blocks
    from tmol_core.io.missing_atoms import prepare_leaf_atom_metadata

    jax = backend.module
    # The explicit residue-block loader is a JAX API; its additional molecular
    # rows reuse this behavior's original float32/float64 parameterization.
    case_names = {
        "ligand": ("ligand", "d_peptide", "mixed_batch"),
        "connected_noncanonical": ("connected_noncanonical",),
        "modified_dna": ("modified_dna",),
    }.get(case, ())
    for case_name in case_names:
        case = DOMAIN_CASES[case_name]
        terms = {
            "ligand": "gen_torsions",
            "connected_noncanonical": "cart_lengths",
            "d_peptide": "fa_elec",
            "modified_dna": "na_torsion",
            "mixed_batch": "gen_torsions",
        }
        b = backend.molecule(case, terms=(terms[case_name],))
        with jax.enable_x64(dtype == np.float64), jax.default_device(b.device):
            complete = [
                [ResidueBlock(*record) for record in row]
                for row in case.blocks(b.parameter_database, dtype)
            ]
            expected, context = pose_stack_from_residue_blocks(
                complete,
                parameter_database=b.parameter_database,
                connections=case.connections(),
                return_context=True,
            )
            pbt = context.packed_block_types
            info = prepare_leaf_atom_metadata(
                pbt.active_block_types,
                np.asarray(pbt.atom_is_hydrogen),
                pbt.max_n_atoms,
            )
            incomplete, observed, targets = (
                [],
                np.asarray(expected.atom_is_real).copy(),
                [],
            )
            for p, row in enumerate(complete):
                prepared = []
                for r, block in enumerate(row):
                    t = int(expected.block_type_ind[p, r])
                    n = len(block.atom_names)
                    eligible = np.flatnonzero(
                        info.is_leaf_atom[t, :n]
                        & np.asarray(pbt.atom_is_hydrogen[t, :n])
                        & np.all(info.icoors.anc_uaids[t, :n, :, 0] >= 0, axis=1)
                    )
                    keep = np.ones(n, bool)
                    if eligible.size:
                        atom = int(eligible[0])
                        keep[atom] = False
                        offset = int(expected.block_coord_offset[p, r])
                        observed[p, offset + atom] = False
                        targets.append(
                            (
                                p,
                                offset + atom,
                                offset + info.icoors.anc_uaids[t, atom, 0, 0],
                                info.icoors.geom[t, atom, 2],
                            )
                        )
                    prepared.append(
                        ResidueBlock(
                            block.residue_type,
                            np.asarray(block.atom_names)[keep][::-1],
                            block.coords[keep][::-1],
                        )
                    )
                incomplete.append(prepared)
            assert targets, f"{case_name} must exercise missing molecular hydrogens"
            with pytest.raises(ValueError, match="atoms must match"):
                pose_stack_from_residue_blocks(
                    incomplete, context=context, connections=case.connections()
                )
            actual, returned = pose_stack_from_residue_blocks(
                incomplete,
                context=context,
                connections=case.connections(),
                return_context=True,
                build_missing_leaf_atoms=True,
            )
            assert returned is context and actual.packed_block_types is pbt
            assert actual.coords.dtype == dtype
            np.testing.assert_array_equal(
                actual.coords[observed], expected.coords[observed]
            )
            np.testing.assert_array_equal(
                actual.block_type_ind, expected.block_type_ind
            )
            np.testing.assert_array_equal(
                actual.inter_residue_connections, expected.inter_residue_connections
            )
            for p, atom, parent, distance in targets:
                np.testing.assert_allclose(
                    np.linalg.norm(
                        np.asarray(actual.coords[p, atom] - actual.coords[p, parent])
                    ),
                    distance,
                    rtol=2e-6 if dtype == np.float32 else 2e-12,
                    atol=2e-6 if dtype == np.float32 else 2e-12,
                )
            b.pose = actual.to(b.device)
            energy, gradient = b.value_and_grad(b.coords())
            assert np.isfinite(energy).all() and np.isfinite(gradient).all()
            assert np.any(energy != 0) and np.linalg.norm(gradient) > 0
            # Opting into leaf reconstruction must not hide a missing core atom.
            first = incomplete[0][0]
            nonleaf = np.flatnonzero(
                ~info.is_leaf_atom[0, : len(complete[0][0].atom_names)]
            )[0]
            atom = complete[0][0].atom_names[nonleaf]
            keep = np.asarray(first.atom_names) != atom
            invalid = [
                [
                    dataclasses.replace(
                        first,
                        atom_names=np.asarray(first.atom_names)[keep],
                        coords=first.coords[keep],
                    ),
                    *incomplete[0][1:],
                ],
                *incomplete[1:],
            ]
            with pytest.raises(ValueError, match="missing non-leaf"):
                pose_stack_from_residue_blocks(
                    invalid,
                    context=context,
                    connections=case.connections(),
                    build_missing_leaf_atoms=True,
                )
            # The separate opt-in uses common context placement before leaf frames.
            recovered = pose_stack_from_residue_blocks(
                invalid,
                context=context,
                connections=case.connections(),
                build_missing_context=True,
            )
            assert np.isfinite(np.asarray(recovered.coords)).all()
            for p, row in enumerate(invalid):
                for r, block in enumerate(row):
                    residue = recovered.block_type(p, r)
                    offset = int(recovered.block_coord_offset[p, r])
                    indices = [
                        offset + residue.atom_to_idx[name] for name in block.atom_names
                    ]
                    np.testing.assert_array_equal(
                        recovered.coords[p, indices], block.coords
                    )
            b.pose = recovered.to(b.device)
            energy, gradient = b.value_and_grad(b.coords())
            assert np.isfinite(energy).all() and np.isfinite(gradient).all()
            assert np.any(energy != 0) and np.linalg.norm(gradient) > 0
            if case_name == "ligand":
                empty = [
                    [
                        ResidueBlock(
                            complete[0][0].residue_type, (), np.empty((0, 3), dtype)
                        )
                    ]
                ]
                with pytest.raises(ValueError, match="could not be reconstructed"):
                    pose_stack_from_residue_blocks(
                        empty, context=context, build_missing_context=True
                    )


def _check_public_context_input(b, case, missing, expected):
    pbt = b.pose.packed_block_types
    types, offsets = map(b.numpy, (b.pose.block_type_ind, b.pose.block_coord_offset))
    is_h = b.numpy(pbt.atom_is_hydrogen).astype(bool)
    from .test_canonical_molecular_input import (
        canonical_input,
        rebuild_canonical,
    )

    co, cf = canonical_input(DOMAIN_CASES[case], b.parameter_database)
    for p, r in np.argwhere(types >= 0):
        residue = pbt.active_block_types[int(types[p, r])]
        start = int(offsets[p, r])
        target_row = int(r + (r >= 1 and types.shape[1] > 1))
        for atom in np.flatnonzero(missing[p, start : start + residue.n_atoms]):
            slot = co.restypes_atom_index_mapping[residue.io_equiv_class][
                residue.atoms[atom].name
            ]
            cf["coords"][p, target_row, slot] = np.nan
    if b.framework == "torch" and case != "ligand":
        # Torch's canonical constructor limits automatic heavy completion
        # to conformer-backed ligands. Its context primitive below still
        # supports these residues; JAX exposes an explicit opt-in loader.
        with pytest.raises(ValueError, match="missing non-leaf atom"):
            rebuild_canonical(b, co, cf, build_missing_context=True)
    else:
        rebuild_canonical(b, co, cf, build_missing_context=True)
        loaded = b.pose
        # The public loader also applies input-H policy: generated H
        # cannot be retained when their heavy parent was absent. The
        # context primitive itself only reconstructs the requested atoms.
        regenerated = np.zeros(missing.shape, bool)
        for p, r in np.argwhere(types >= 0):
            t, start = int(types[p, r]), int(offsets[p, r])
            residue = pbt.active_block_types[t]
            if not residue.hydrogens_regenerated:
                continue
            ideal = residue.ideal_coords[residue.at_to_icoor_ind]
            for h, parent in residue.bond_indices:
                if is_h[t, h] and missing[p, start + parent]:
                    regenerated[p, start + h] = True
                    np.testing.assert_allclose(
                        np.linalg.norm(
                            b.numpy(loaded.coords)[p, start + h]
                            - b.numpy(loaded.coords)[p, start + parent]
                        ),
                        np.linalg.norm(ideal[h] - ideal[parent]),
                        rtol=2e-6,
                        atol=2e-6,
                    )
        np.testing.assert_allclose(
            b.numpy(loaded.coords)[~regenerated],
            b.numpy(expected)[~regenerated],
            rtol=2e-6,
            atol=2e-6,
        )


def _reserved_input(b, co, cf, packer_atoms):
    if b.framework == "jax":
        from tmol_jax.io import pose_stack_from_canonical_form

        return pose_stack_from_canonical_form(
            dict(cf, canonical_ordering=co),
            b.pose.packed_block_types,
            compact_padding=True,
            build_missing_context=True,
            return_block_has_missing_atoms=True,
            packer_atoms=packer_atoms,
            ligand_hydrogens="input",
        )
    from tmol.io._pose_stack_construction import _pose_stack_from_canonical_form
    from tmol.io._canonical_ordering import CanonicalOrdering

    def array(value):
        return b.module.as_tensor(value, device=b.device)

    return _pose_stack_from_canonical_form(
        CanonicalOrdering.from_chemdb(b.parameter_database.chemical),
        b.pose.packed_block_types,
        array(cf["chain_id"]),
        array(cf["res_types"]),
        array(cf["coords"]),
        None,
        None,
        None,
        res_not_connected=array(cf["res_gap"]),
        ligand_hydrogens="input",
        return_block_has_missing_atoms=True,
        packer_atoms=packer_atoms,
        find_additional_cyclic_closures=False,
    )


@pytest.mark.parametrize("case", ("ligand", "noncanonical", "connected_noncanonical"))
def test_input_preserves_missing_atoms_reserved_for_packing(backend, case):
    from tmol_core.io.missing_atoms import prepare_leaf_atom_flags
    from .test_canonical_molecular_input import canonical_input

    b = backend.molecule(DOMAIN_CASES[case], terms=("ref",))
    pbt = b.pose.packed_block_types
    flags = prepare_leaf_atom_flags(
        pbt.active_block_types, b.numpy(pbt.atom_is_hydrogen), pbt.max_n_atoms
    )
    co, cf = canonical_input(DOMAIN_CASES[case], b.parameter_database)
    targets = []

    def reserved(rt):
        mainchain = set(rt.properties.polymer.mainchain_atoms or ())
        return np.array([atom.name not in mainchain for atom in rt.atoms])

    for p, r in np.argwhere(b.numpy(b.pose.block_type_ind) >= 0):
        kind = int(b.numpy(b.pose.block_type_ind)[p, r])
        rt = pbt.active_block_types[kind]
        eligible = (
            reserved(rt)
            & ~flags.is_leaf_atom[kind, : rt.n_atoms]
            & ~b.numpy(pbt.atom_is_hydrogen)[kind, : rt.n_atoms].astype(bool)
        )
        assert eligible.any()
        atom = int(np.flatnonzero(eligible)[-1])
        source_row = r + (r >= 1 and cf["res_types"].shape[1] > 1)
        slot = co.restypes_atom_index_mapping[rt.io_equiv_class][rt.atoms[atom].name]
        cf["coords"][p, source_row, slot] = np.nan
        targets.append((p, r, rt.atoms[atom].name))
    before = cf["coords"].copy()
    pose, metadata = _reserved_input(b, co, cf, reserved)
    missing = b.numpy(metadata["block_has_missing_atoms"])
    assert missing[b.numpy(pose.block_type_ind) >= 0].all()
    for p, r, name in targets:
        rt = pose.block_type(int(p), int(r))
        start = int(b.numpy(pose.block_coord_offset)[p, r])
        assert np.isnan(b.numpy(pose.coords)[p, start + rt.atom_to_idx[name]]).all()
        source_row = r + (r >= 1 and cf["res_types"].shape[1] > 1)
        for atom in rt.atoms:
            slot = co.restypes_atom_index_mapping[rt.io_equiv_class][atom.name]
            observed = before[p, source_row, slot]
            if (
                not b.numpy(pose.packed_block_types.atom_is_hydrogen)[
                    int(b.numpy(pose.block_type_ind)[p, r]), rt.atom_to_idx[atom.name]
                ]
                and np.isfinite(observed).all()
            ):
                np.testing.assert_array_equal(
                    b.numpy(pose.coords)[p, start + rt.atom_to_idx[atom.name]], observed
                )
    np.testing.assert_array_equal(cf["coords"], before)
