"""Sparse conformer values conserve pose energy and arbitrary weighted forces."""

from contextlib import nullcontext

import numpy as np
import pytest

from .molecules import CASES, DOMAIN_CASES, BACKBONE_CASES
from .hbond_molecules import HBOND_CASES
from .na_molecules import NA_CASES
from .disulfide_molecules import DISULFIDE_CASES
from .metal_molecules import METAL_CASES
from .test_cartbonded import CARTESIAN_CASES, TERMS as CART_TERMS
from .rotamer_execution import RotamerExecution


@pytest.mark.parametrize(
    "family,case_name",
    [
        (family, name)
        for family in ("ljlk", "elec", "lk_ball")
        for name in (
            "ligand",
            "connected_noncanonical",
            "d_peptide",
            "modified_dna",
            "metal_mixed",
            "mixed_batch",
        )
    ]
    + [("genbonded", name) for name in CASES]
    + [("hbond", name) for name in HBOND_CASES]
    + [("na_torsion", name) for name in NA_CASES]
    + [("backbone", name) for name in BACKBONE_CASES]
    + [("disulfide", name) for name in DISULFIDE_CASES]
    + [("metal", name) for name in METAL_CASES]
    + [
        ("cartbonded", name)
        for name in (
            "ligand",
            "noncanonical",
            "connected_noncanonical",
            "d_peptide",
            "mixed_batch",
            "hydroxyls",
        )
    ],
)
def test_sparse_nonbonded_values_and_coordinate_derivatives(backend, case_name, family):
    terms = {
        "genbonded": ("gen_torsions",),
        "cartbonded": CART_TERMS,
        "backbone": ("rama", "omega"),
        "disulfide": ("disulfide",),
        "metal": ("metal_coordination",),
        "hbond": ("hbond",),
        "na_torsion": ("na_torsion", "na_torsion_well"),
        "elec": ("fa_elec",),
        "ljlk": ("fa_ljatr", "fa_ljrep", "fa_lk"),
        "lk_ball": ("lk_ball_iso", "lk_ball", "lk_bridge", "lk_bridge_uncpl"),
    }[family]
    cases = HBOND_CASES if family == "hbond" else CARTESIAN_CASES
    if family == "na_torsion":
        cases = NA_CASES
    elif family == "backbone":
        cases = BACKBONE_CASES
    elif family == "disulfide":
        cases = DISULFIDE_CASES
    elif family == "metal":
        cases = METAL_CASES
    backend.molecule(cases[case_name], terms=terms)
    precision = (
        backend.module.enable_x64() if backend.framework == "jax" else nullcontext()
    )
    with precision:
        execution = RotamerExecution(backend, family=family)
        coords = backend.numpy(execution.rotamers.coords).astype(np.float64)
        values, indices = map(backend.numpy, execution.scores(coords))
        assert np.isfinite(values).all() and np.any(np.abs(values) > 1e-3)
        whole = backend.scores(
            backend.coords().astype(np.float64), sum_terms=False, apply_weights=False
        )
        channel_names = backend.channel_names()
        whole = whole[[channel_names.index(name) for name in terms]]
        aggregated = np.zeros_like(whole)
        for entry, pose in enumerate(indices[0]):
            aggregated[:, pose] += values[:, entry]
        np.testing.assert_allclose(aggregated, whole, rtol=2e-12, atol=2e-10)
        weights = np.cos(np.arange(values.size).reshape(values.shape) * 0.21) + 0.4
        scored, gradient, scored_indices = execution.value_and_grad(coords, weights)
        np.testing.assert_array_equal(scored_indices, indices)
        np.testing.assert_allclose(scored, values, rtol=0, atol=0)
        direction = np.random.default_rng(103).normal(size=coords.shape)
        direction /= np.linalg.norm(direction)
        analytic = np.sum(gradient * direction)
        # Degree-10 HBond fits lose subtraction digits below 1e-4 even in
        # float64. These two steps lie in the converged interval, before
        # cancellation dominates; the gradient tolerance is unchanged.
        steps = (1e-3, 3e-4) if family == "hbond" else (1e-5, 2e-6)
        for step in steps:
            plus, plus_indices = map(
                backend.numpy, execution.scores(coords + step * direction)
            )
            minus, minus_indices = map(
                backend.numpy, execution.scores(coords - step * direction)
            )
            np.testing.assert_array_equal(plus_indices, indices)
            np.testing.assert_array_equal(minus_indices, indices)
            numerical = np.sum((plus - minus) * weights) / (2 * step)
            np.testing.assert_allclose(analytic, numerical, rtol=3e-6, atol=3e-6)
        np.testing.assert_allclose(gradient.sum(axis=0), 0, atol=2e-10)
        np.testing.assert_array_equal(backend.numpy(execution.rotamers.coords), coords)


@pytest.mark.parametrize(
    "family",
    (
        "ljlk",
        "lk_ball",
        "genbonded",
        "hbond",
        "cartbonded",
        "na_torsion",
        "backbone",
        "disulfide",
        "metal",
    ),
)
def test_sparse_custom_parameters_preserve_existing_scorer(backend, family):
    import attrs

    case = DOMAIN_CASES["ligand" if family == "ljlk" else "connected_noncanonical"]
    if family == "hbond":
        case = HBOND_CASES["beta_amino_ligand"]
    elif family == "na_torsion":
        case = NA_CASES["methylcytosine"]
    elif family == "backbone":
        case = BACKBONE_CASES["hydroxyproline"]
    elif family == "disulfide":
        case = DISULFIDE_CASES["ld"]
    elif family == "metal":
        case = METAL_CASES["bridging_donor"]
    backend.molecule(case, terms=("ref",))
    precision = (
        backend.module.enable_x64() if backend.framework == "jax" else nullcontext()
    )
    with precision:
        original = RotamerExecution(backend, family=family)
        coords = backend.numpy(original.rotamers.coords).astype(np.float64)
        first, indices = map(backend.numpy, original.scores(coords))
        scaled_channels, factor = {
            "ljlk": (2, 4),
            "lk_ball": (3, 2),
            "genbonded": (1, 2),
            "cartbonded": (1, 2),
            "hbond": (1, 2),
            "na_torsion": (1, 0.25),
            "backbone": (1, 2),
            "disulfide": (1, 2),
            "metal": (1, 0.25),
        }[family]
        weights = np.broadcast_to(
            (np.arange(first.shape[0]) < scaled_channels)[:, None], first.shape
        ).astype(np.float64)
        if family == "lk_ball":
            assert np.all(np.max(np.abs(first), axis=1) > 1e-4)
        _, gradient, _ = original.value_and_grad(coords, weights)
        database = backend.parameter_database
        # Fourfold well depth doubles its float32 square-root table exactly;
        # this keeps the analytic scaling check free of parameter quantization.
        if family == "disulfide":
            from .test_disulfide import scaled_disulfide_database

            modified = scaled_disulfide_database(database)
        elif family == "metal":
            from .test_metal_coordination import scaled_metal_database

            modified = scaled_metal_database(database)
        elif family == "backbone":
            source = database.scoring.rama
            changed = attrs.evolve(
                source,
                rama_tables=tuple(
                    attrs.evolve(row, table=2 * row.table) for row in source.rama_tables
                ),
            )
            changed = attrs.evolve(changed, uniq_id=changed.content_id())
            modified = attrs.evolve(
                database, scoring=attrs.evolve(database.scoring, rama=changed)
            )
        elif family == "na_torsion":
            source = database.scoring.na_torsion
            changed = attrs.evolve(
                source,
                global_parameters={
                    name: attrs.evolve(
                        row,
                        sdev_sugar=2 * row.sdev_sugar,
                        sdev_chi=2 * row.sdev_chi,
                        sdev_backbone=tuple(2 * x for x in row.sdev_backbone),
                    )
                    for name, row in source.global_parameters.items()
                },
            )
            modified = attrs.evolve(
                database, scoring=attrs.evolve(database.scoring, na_torsion=changed)
            )
        elif family == "cartbonded":
            source = database.scoring.cartbonded

            def scale(record):
                return attrs.evolve(
                    record,
                    length_parameters=tuple(
                        attrs.evolve(p, K=2 * p.K) for p in record.length_parameters
                    ),
                )

            changed = type(source).from_cartres_dict(
                {name: scale(record) for name, record in source.residue_params.items()},
                tuple(scale(record) for record in source.connection_params),
            )
            modified = attrs.evolve(
                database, scoring=attrs.evolve(database.scoring, cartbonded=changed)
            )
        elif family == "hbond":
            source = database.scoring.hbond
            changed = attrs.evolve(
                source,
                donor_type_params=tuple(
                    attrs.evolve(p, weight=2 * p.weight)
                    for p in source.donor_type_params
                ),
            )
            modified = attrs.evolve(
                database, scoring=attrs.evolve(database.scoring, hbond=changed)
            )
        elif family == "genbonded":
            source = database.scoring.genbonded
            changed = attrs.evolve(
                source,
                torsions=tuple(
                    attrs.evolve(
                        p,
                        **{
                            k: 2 * getattr(p, k)
                            for k in ("k1", "k2", "k3", "k4", "offset")
                        },
                    )
                    for p in source.torsions
                ),
                impropers=tuple(attrs.evolve(p, k=2 * p.k) for p in source.impropers),
            )
            modified = attrs.evolve(
                database, scoring=attrs.evolve(database.scoring, genbonded=changed)
            )
        else:
            parameters = tuple(
                attrs.evolve(
                    p,
                    **(
                        {"lj_wdepth": 4 * p.lj_wdepth}
                        if family == "ljlk"
                        else {"lk_dgfree": 2 * p.lk_dgfree}
                    ),
                )
                for p in database.scoring.ljlk.atom_type_parameters
            )
            ljlk = attrs.evolve(database.scoring.ljlk, atom_type_parameters=parameters)
            modified = attrs.evolve(
                database, scoring=attrs.evolve(database.scoring, ljlk=ljlk)
            )
        backend.molecule(case, terms=("ref",), parameter_database=modified)
        changed = RotamerExecution(backend, family=family)
        actual, actual_gradient, actual_indices = changed.value_and_grad(
            coords, weights
        )
        np.testing.assert_array_equal(actual_indices, indices)
        np.testing.assert_allclose(
            actual[:scaled_channels],
            factor * first[:scaled_channels],
            rtol=2e-12,
            atol=2e-11,
        )
        np.testing.assert_allclose(
            actual[scaled_channels:], first[scaled_channels:], rtol=2e-12, atol=2e-11
        )
        np.testing.assert_allclose(
            actual_gradient, factor * gradient, rtol=2e-12, atol=2e-11
        )
        np.testing.assert_array_equal(backend.numpy(original.scores(coords)[0]), first)


def test_sparse_dunbrack_preserves_library_free_conformer_identity(backend):
    backend.molecule(DOMAIN_CASES["ligand"], terms=("ref",))
    e = RotamerExecution(backend, family="dunbrack")
    coords = backend.numpy(e.rotamers.coords)
    values, indices = map(backend.numpy, e.scores(coords))
    np.testing.assert_array_equal(values, 0)
    np.testing.assert_array_equal(indices[0], backend.numpy(e.rotamers.pose_for_rot))
    np.testing.assert_array_equal(indices[1], np.arange(e.rotamers.n_rotamers_total))
    np.testing.assert_array_equal(indices[2], indices[1])
    _, gradient, _ = e.value_and_grad(coords, np.ones_like(values))
    np.testing.assert_array_equal(gradient, np.zeros_like(coords))


@pytest.mark.parametrize(
    "family,case_name",
    [
        (family, case)
        for family in (
            "ljlk",
            "elec",
            "dunbrack",
            "lk_ball",
            "genbonded",
            "cartbonded",
            "backbone",
        )
        for case in ("l_peptide", "mixed_batch")
    ]
    + [("hbond", "sampled_ser_ligand")]
    + [("na_torsion", name) for name in NA_CASES]
    + [("disulfide", name) for name in ("ld", "mixed_batch")]
    + [("metal", name) for name in ("zinc_site", "bridging_donor", "mixed_batch")],
)
def test_sampled_ensemble_selection_conserves_energy_and_forces(
    backend, case_name, family
):
    from .dunbrack_molecules import DUNBRACK_CASES

    terms = {
        "genbonded": ("gen_torsions",),
        "cartbonded": CART_TERMS,
        "backbone": ("rama", "omega"),
        "disulfide": ("disulfide",),
        "metal": ("metal_coordination",),
        "hbond": ("hbond",),
        "elec": ("fa_elec",),
        "ljlk": ("fa_ljatr", "fa_ljrep", "fa_lk"),
        "dunbrack": ("dunbrack_rot", "dunbrack_rotdev", "dunbrack_semirot"),
        "lk_ball": ("lk_ball_iso", "lk_ball", "lk_bridge", "lk_bridge_uncpl"),
        "na_torsion": ("na_torsion", "na_torsion_well"),
    }[family]
    cases = HBOND_CASES if family == "hbond" else DUNBRACK_CASES
    if family == "disulfide":
        cases = DISULFIDE_CASES
    elif family == "metal":
        cases = METAL_CASES
    elif family == "na_torsion":
        cases = NA_CASES
    backend.molecule(cases[case_name], terms=terms)
    precision = (
        backend.module.enable_x64() if backend.framework == "jax" else nullcontext()
    )
    with precision:
        # NA choices here are newly sampled conformers. Mixing exact copied
        # input with a float32 reconstructed sugar introduces backbone roundoff
        # above this test's float64 energy bound. Both APIs' current conformers
        # are exercised separately; this also avoids Torch's float32-only builder.
        e = RotamerExecution(
            backend,
            family=family,
            sampled=True,
            include_current=family != "na_torsion",
        )
        r = e.rotamers
        xyz = backend.numpy(r.coords).astype(np.float64)
        counts = backend.numpy(r.n_rots_for_block)
        starts = backend.numpy(r.rot_offset_for_block)
        assert counts.max() > 2
        values, indices = map(backend.numpy, e.scores(xyz))
        if family == "hbond":
            assert np.any(values < -0.1)
        selected = np.zeros(r.n_rotamers_total, bool)
        reconstructed = backend.coords().astype(np.float64)
        offsets = backend.numpy(r.coord_offset_for_rot)
        types = backend.numpy(r.block_type_ind_for_rot)
        for pose, block in np.argwhere(counts > 0):
            rot = starts[pose, block] + (pose + block + 1) % counts[pose, block]
            selected[rot] = True
            n = e.packed_block_types.active_block_types[types[rot]].n_atoms
            dest = backend.numpy(backend.pose.block_coord_offset)[pose, block]
            reconstructed[pose, dest : dest + n] = xyz[offsets[rot] : offsets[rot] + n]
        chosen = selected[indices[1]] & selected[indices[2]]
        aggregated = np.zeros((values.shape[0], len(counts)))
        for entry in np.flatnonzero(chosen):
            aggregated[:, indices[0, entry]] += values[:, entry]
        channel_names = backend.channel_names()
        whole = backend.scores(reconstructed, sum_terms=False, apply_weights=False)
        whole = whole[[channel_names.index(name) for name in terms]]
        np.testing.assert_allclose(
            aggregated,
            whole,
            rtol=2e-12,
            atol=2e-9,
        )
        weights = np.cos(np.arange(values.size).reshape(values.shape) * 0.21) + 0.4
        _, gradient, _ = e.value_and_grad(xyz, weights)
        direction = np.random.default_rng(203).normal(size=xyz.shape)
        direction /= np.linalg.norm(direction)
        steps = (1e-3, 3e-4) if family == "hbond" else (1e-5, 2e-6)
        for step in steps:
            plus = backend.numpy(e.scores(xyz + step * direction)[0])
            minus = backend.numpy(e.scores(xyz - step * direction)[0])
            np.testing.assert_allclose(
                (gradient * direction).sum(),
                ((plus - minus) * weights).sum() / (2 * step),
                rtol=3e-6,
                atol=3e-6,
            )
