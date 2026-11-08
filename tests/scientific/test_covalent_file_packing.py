"""Complete covalent molecular input and packing retain finite repeatable physics."""

from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from .test_complete_preset import CHANNELS
from .test_molecular_sampling import sampling_task
from .test_canonical_molecular_input import physical_edges

FILES = (
    "lys_biotin_1bdo.cif",
    "oglycan_sia_1g1s.cif",
    "nglycan_tree_1ax2.cif",
)


def prepared_covalent_file(backend, filename):
    frontend = import_module("tmol_jax" if backend.framework == "jax" else "tmol")
    path = (
        Path(__file__).resolve().parents[2]
        / "tmol/tests/data/covalent_fixtures"
        / filename
    )
    backend.pose, context = frontend.pose_stack_from_file(
        path,
        backend.device,
        prepare_ligands=True,
        ligand_seed=17,
        packer_seed=17,
        no_optH=False,
        return_context=True,
    )
    backend.parameter_database = context.parameter_database
    return frontend


def repack_covalent_file(backend, frontend):
    task, _, _ = sampling_task(backend)
    pack = import_module(frontend.__name__ + ".pack")
    task.add_conformer_sampler(pack.FixedAAChiSampler())
    if backend.framework == "jax":
        from tmol_jax.pack.conjugated import add_conjugated_group_sampler

        group = add_conjugated_group_sampler(task)
    else:
        from tmol.pack.rotamer._conjugated_groups import _add_conjugated_group_sampler

        group = _add_conjugated_group_sampler(task, backend.pose)
    assert group is not None, "real covalent fixture must exercise conjugated packing"
    group.chi_sample_expanded_limit = 256
    group.chi_sample_limit = 256
    task.or_bump_check(True)
    return frontend.pack_rotamers(backend.pose, backend.score_function, task, seed=17)


@pytest.mark.parametrize("filename", FILES)
def test_covalent_file_complete_packing_energy_gradient_repeats(backend, filename):
    b = backend
    frontend = prepared_covalent_file(b, filename)
    channels, _ = b.use_beta2016()
    assert len(channels) == 25 and set(channels) == CHANNELS
    before = b.coords().copy()
    positions = np.argwhere(b.numpy(b.pose.block_type_ind) >= 0)
    sequence = [b.pose.block_type(int(p), int(r)).io_equiv_class for p, r in positions]
    edges = physical_edges(b)
    assert np.isfinite(before).all()
    assert len(edges) > 1
    packed = repack_covalent_file(b, frontend)
    np.testing.assert_array_equal(b.coords(), before)
    # Repacking may select histidine tautomers; retain chemical sequence and
    # atom-named connectivity rather than imposing identical type indices.
    assert [
        packed.block_type(int(p), int(r)).io_equiv_class for p, r in positions
    ] == sequence
    b.pose = packed
    assert physical_edges(b) == edges
    b.use_beta2016()
    coordinates = b.coords().copy()
    assert np.isfinite(coordinates).all()
    terms, term_gradients = b.channel_values_and_gradients(coordinates)
    assert terms.shape == (25, packed.n_poses)
    assert np.isfinite(terms).all() and np.isfinite(term_gradients).all()
    for name in ("gen_torsions", "hbond", "dunbrack_rot"):
        assert np.max(np.abs(terms[channels.index(name)])) > 1e-3
    reference_energy, reference_gradient = b.value_and_grad(coordinates)
    assert np.linalg.norm(reference_gradient) > 1e-3
    # Retain the complete-workflow float32 bounds. Native CUDA reductions must
    # satisfy these on the actual covalent system, not only on small proteins.
    for _ in range(30):
        energy, gradient = b.value_and_grad(coordinates)
        np.testing.assert_allclose(energy, reference_energy, rtol=3e-6, atol=3e-5)
        np.testing.assert_allclose(gradient, reference_gradient, rtol=3e-6, atol=3e-5)
    np.testing.assert_array_equal(b.coords(), coordinates)
