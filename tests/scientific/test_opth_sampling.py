"""Real proton and NHQ sampling in ligand, D, noncanonical and NA chemistry."""

import numpy as np
import pytest

from .molecules import MolecularCase, DOMAIN_CASES
from .test_molecular_sampling import block_mask, check_sampler_support

OPTH_CASES = {
    "protein": MolecularCase(
        (), (("ASN", "GLN", "HIS", "SER", "THR", "TYR"),), ((6,),)
    ),
    "d_amino": MolecularCase(
        (), (("DASN", "DGLN", "DHIS", "DSER", "DTHR", "DTYR"),), ((6,),)
    ),
    "noncanonical": DOMAIN_CASES["noncanonical"],
    "ligand": MolecularCase(
        ("ligand_params/opth_propanediol.tmol",), (("OPL",),), ((1,),)
    ),
    "modified_na_internal": DOMAIN_CASES["modified_dna"],
    "modified_na": MolecularCase(
        DOMAIN_CASES["modified_dna"].parameter_files,
        (("DA:na5prime", "5CM", "DG:na3prime"),),
        ((3,),),
    ),
    "mixed": MolecularCase(
        (
            *DOMAIN_CASES["noncanonical"].parameter_files,
            "ligand_params/opth_propanediol.tmol",
        ),
        (("DASN", "OPL", "HYP"), ("SER",)),
        ((1, 1, 1), (1,)),
    ),
}


def opth_task(backend, *, flip=True):
    if backend.framework == "jax":
        from tmol_jax.pack import PackerPalette, PackerTask, SetPackerTask
        from tmol_jax.pack.opth import OptHSampler
    else:
        from tmol import PackerPalette, PackerTask
        from tmol.pack import OptHSampler, SetPackerTask
    sampler = OptHSampler(flip_NHQ=flip)
    task = PackerTask(backend.pose, PackerPalette())
    task.restrict_to_repacking()
    task.add_conformer_sampler(sampler)
    return task, sampler, SetPackerTask.from_packer_task


@pytest.mark.parametrize("name", OPTH_CASES)
@pytest.mark.parametrize("flip", (False, True))
def test_opth_molecular_samples_and_masks(backend, name, flip):
    b = backend.molecule(OPTH_CASES[name], terms=("ref",))
    task, sampler, concrete = opth_task(b, flip=flip)
    before = b.coords().copy()
    counts, owners, payload = sampler.create_samples_for_poses(b.pose, concrete(task))
    counts, owners, atoms, chi = map(
        b.numpy,
        (
            counts,
            owners,
            payload["chi_defining_atom_for_rotamer"],
            payload["chi_for_rotamers"],
        ),
    )
    assert len(owners) == counts.sum()
    assert atoms.shape == chi.shape and np.isfinite(chi).all()
    np.testing.assert_array_equal(owners, np.repeat(np.arange(len(counts)), counts))
    if name == "modified_na_internal":
        assert not counts.any()
    else:
        assert counts.sum() > 0 and np.any(np.abs(chi) > 0.1)
    if name == "ligand":
        assert counts.sum() == 81
        check_sampler_support(b, sampler, concrete(task), counts)
        assert np.all((atoms >= 0).sum(axis=1) == 2)
        assert len(np.unique(chi, axis=0)) == 81
    np.testing.assert_array_equal(b.coords(), before)
    from .test_current_conformers import ensure_fallback_sampler
    from .test_conformer_coordinates import check_sampled_dihedrals

    if b.framework == "jax":
        from tmol_jax.pack import build_rotamers
    else:
        from tmol.pack import build_rotamers
    ensure_fallback_sampler(b, task)
    fixed = concrete(task)
    _, result = build_rotamers(b.pose, fixed, b.parameter_database.chemical)
    checked = check_sampled_dihedrals(
        b, fixed, result, counts, owners, atoms, chi, before
    )
    if name == "ligand":
        assert checked == 162
        # Proton-only torsions must not displace the ligand's heavy atoms.
        residue = b.pose.packed_block_types.active_block_types[
            int(b.numpy(result.block_type_ind_for_rot)[0])
        ]
        heavy = np.asarray(
            [i for i, atom in enumerate(residue.atoms) if not atom.name.startswith("H")]
        )
        for offset in b.numpy(result.coord_offset_for_rot):
            np.testing.assert_allclose(
                b.numpy(result.coords)[offset + heavy],
                before[0, heavy],
                atol=2e-5,
                rtol=0,
            )
    task.disable_packing_by_block_mask(
        block_mask(b, np.ones(task.is_real_block.shape, bool))
    )
    empty = sampler.create_samples_for_poses(b.pose, concrete(task))
    assert not b.numpy(empty[0]).any() and b.numpy(empty[1]).size == 0


def _sampled_coordinate_function(b, task, sampler):
    """Small execution adapters around the same molecular construction task."""
    if b.framework == "jax":
        from tmol_jax.pack.rotamers import build_rotamers, _sample_conformers

        build_rotamers(b.pose, task, b.parameter_database.chemical)
        samplers, samples, *_ = _sample_conformers(b.pose, task)
        payloads = tuple(
            (
                ()
                if s[2].get("copy_input_coordinates", False)
                else (s[2]["chi_for_rotamers"], s[2]["chi_defining_atom_for_rotamer"])
            )
            for s in samples
        )
        selected = samplers.index(sampler)
        execute = task._assembly_execution[2]

        def run(coords, shift):
            args = list(payloads)
            angles, atoms = args[selected]
            args[selected] = (angles + shift, atoms)
            return execute(coords, tuple(args))

    else:
        from unittest.mock import patch
        import attrs
        from tmol.pack import build_rotamers

        create = type(sampler).create_samples_for_poses

        def run(coords, shift):
            def samples(instance, *args, **kwargs):
                counts, owners, payload = create(instance, *args, **kwargs)
                payload = dict(payload)
                payload["chi_for_rotamers"] = payload["chi_for_rotamers"] + shift
                return counts, owners, payload

            with patch.object(type(sampler), "create_samples_for_poses", samples):
                _, result = build_rotamers(
                    attrs.evolve(b.pose, coords=coords),
                    task,
                    b.parameter_database.chemical,
                )
            return result.coords

    return run


@pytest.mark.parametrize("name", ("protein", "d_amino", "ligand", "mixed"))
@pytest.mark.parametrize("flip", (False, True))
def test_opth_constructed_coordinate_parameter_gradients(backend, name, flip):
    from .test_current_conformers import ensure_fallback_sampler

    b = backend.molecule(OPTH_CASES[name], terms=("ref",))
    task, sampler, concrete = opth_task(b, flip=flip)
    ensure_fallback_sampler(b, task)
    run = _sampled_coordinate_function(b, concrete(task), sampler)
    coords = b.pose.coords
    if b.framework == "jax":
        shift = b.module.device_put(np.array(0.037, np.float32), b.device)
    else:
        shift = b.module.tensor(0.037, device=b.device, requires_grad=True)
    example = run(coords, shift)
    weights = (
        np.sin(
            np.arange(example.size if b.framework == "jax" else example.numel()) * 0.17
        )
        .reshape(example.shape)
        .astype(np.float32)
    )
    weights /= np.sqrt(weights.size)
    weights = (
        b.module.device_put(weights, b.device)
        if b.framework == "jax"
        else b.module.as_tensor(weights, device=b.device)
    )

    def loss(value):
        return (weights * run(coords, value)).sum()

    if b.framework == "jax":
        gradient = b.module.jit(b.module.grad(loss))(shift)
    else:
        gradient = b.module.autograd.grad(loss(shift), shift)[0]
    epsilon = 0.002
    # Difference coordinates before the weighted reduction. Subtracting two
    # float32 scalar losses loses the small derivative on the protein fixture;
    # CPU/CUDA float64 and a step sweep independently verify this oracle.
    plus = b.numpy(run(coords, shift + epsilon)).astype(np.float64)
    minus = b.numpy(run(coords, shift - epsilon)).astype(np.float64)
    finite = ((plus - minus) * b.numpy(weights).astype(np.float64)).sum() / (
        2 * epsilon
    )
    assert np.isfinite(b.numpy(gradient))
    np.testing.assert_allclose(b.numpy(gradient), finite, rtol=0.003, atol=0.0001)
    # Do not silently return a partial derivative through inverse initialization.
    with pytest.raises(NotImplementedError, match="inverse.*kinematics"):
        if b.framework == "jax":
            b.module.grad(lambda x: run(x, shift).sum())(coords)
        else:
            run(coords.detach().requires_grad_(), shift)
