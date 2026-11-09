"""Chemical choice preparation is distinct from packing execution."""

from contextlib import nullcontext
from importlib import import_module

import numpy as np
import pytest

from .molecules import CASES, MolecularCase


@pytest.mark.parametrize("noncanonical_design", [False, True])
def test_palette_preserves_termini_chirality_and_ligand_identity(
    backend, noncanonical_design
):
    names = (
        "ALA",
        "DALA",
        "AIB",
        "PRO",
        "ALA:nterm",
        "GLY:nterm",
        "PRO:nterm",
        "CYD",
        "CYS",
        "LG1",
    )
    case = MolecularCase(
        CASES["ligand"].parameter_files, (names, ("ALA",)), ((1,) * len(names), (1,))
    )
    backend.molecule(case, terms=("ref",))
    counts, choices, original, repack = backend.palette_choices(
        noncanonical_design=noncanonical_design
    )
    types = backend.pose.packed_block_types.active_block_types

    def offered(block, mask=None):
        row = choices[0, block]
        valid = row >= 0
        if mask is not None:
            valid &= mask[0, block]
        return {types[i].name for i in row[valid]}

    assert ({"AIB", "DALA"} <= offered(0)) == noncanonical_design
    assert ({"AIB", "ALA"} <= offered(1)) == noncanonical_design
    assert {"ALA:nterm", "GLY:nterm", "PRO:nterm"} <= offered(4)
    assert not {"ALA", "GLY", "PRO"} & offered(4)
    assert offered(9) == {"LG1"}
    assert "CYS" not in offered(7)
    for index, name in enumerate(names):
        assert offered(index, original) == {name}
        assert name in offered(index, repack)
    np.testing.assert_array_equal(counts[1, 1:], 0)
    np.testing.assert_array_equal(choices[1, 1:], -1)
    assert not original[1, 1:].any() and not repack[1, 1:].any()
    # Tasks on the same palette may request different design policies. Changing
    # one policy must neither reuse its predecessor nor modify earlier choices.
    _, changed, _, _ = backend.palette_choices(
        noncanonical_design=not noncanonical_design
    )
    alternatives = {types[i].name for i in changed[0, 0] if i >= 0}
    assert ("DALA" in alternatives) == (not noncanonical_design)
    restored = backend.palette_choices(noncanonical_design=noncanonical_design)
    for actual, expected in zip(restored, (counts, choices, original, repack)):
        np.testing.assert_array_equal(actual, expected)
    if not noncanonical_design:
        b, pbt = backend, backend.pose.packed_block_types
        frontend = import_module("tmol_jax" if b.framework == "jax" else "tmol")
        palette = frontend.PackerPalette()
        dtype = np.int32 if b.framework == "jax" else np.int64

        def indices(values):
            values = np.asarray(values, dtype=dtype)
            return (
                b.module.device_put(values, b.device)
                if b.framework == "jax"
                else b.module.as_tensor(values, device=b.device)
            )

        default_device = (
            b.module.default_device(b.module.devices("cpu")[0])
            if b.framework == "jax"
            else nullcontext()
        )
        with default_device:
            for shape in ((2, 2), (0, 2)):
                source = indices(np.full(shape, -1, dtype))
                counts, choices, same = palette.block_types_from_original(pbt, source)
                repack = palette.create_restrict_to_repacking_mask(pbt, source)
                assert all(
                    a.device == b.device for a in (counts, choices, same, repack)
                )
                counts, choices, same, repack = map(
                    b.numpy, (counts, choices, same, repack)
                )
                assert counts.dtype == choices.dtype == dtype
                assert same.dtype == repack.dtype == np.bool_
                assert counts.shape == shape and choices.shape[:2] == shape
                np.testing.assert_array_equal(counts, 0)
                np.testing.assert_array_equal(choices, -1)
                assert not same.any() and not repack.any()
            # JAX validates these indices eagerly; Torch has a different boundary.
            if b.framework == "jax":
                for bad in (-2, pbt.n_types):
                    source = indices([[bad]])
                    for method in (
                        palette.block_types_from_original,
                        palette.create_restrict_to_repacking_mask,
                    ):
                        with pytest.raises(ValueError, match="outside the palette"):
                            method(pbt, source)


def test_task_identity_restrictions_preserve_unselected_chemistry(backend):
    from .test_molecular_sampling import block_mask
    from .test_current_conformers import current_samplers

    b = backend.molecule(
        MolecularCase(
            CASES["ligand"].parameter_files,
            (("ALA", "GLY", "DALA", "LG1"), ("ALA",)),
            ((1, 1, 1, 1), (1,)),
        ),
        terms=("ref",),
    )
    if b.framework == "jax":
        from tmol_jax import PackerTask, PackerPalette
    else:
        from tmol import PackerTask, PackerPalette
    task = PackerTask(b.pose, PackerPalette())
    original = b.numpy(task.per_block_is_block_type_allowed).copy()
    selected = block_mask(
        b, [[True, False, False, False], [False, False, False, False]]
    )
    task.restrict_absent_name3s({"GLY"}, selected)
    allowed = b.numpy(task.per_block_is_block_type_allowed)
    np.testing.assert_array_equal(
        allowed[~b.numpy(selected)], original[~b.numpy(selected)]
    )
    choices = b.numpy(task.per_block_considered_block_types)
    offered = {
        b.pose.packed_block_types.active_block_types[t].name3
        for t in choices[0, 0][allowed[0, 0]]
    }
    assert offered == {"GLY"}
    # A failed explicit restriction must neither coerce a ligand nor alter task state.
    before = allowed.copy()
    ligand = block_mask(b, [[False, False, False, True], [False, False, False, False]])
    with pytest.raises(ValueError, match="no allowed block type"):
        task.restrict_absent_name3s({"ALA"}, ligand)
    np.testing.assert_array_equal(b.numpy(task.per_block_is_block_type_allowed), before)
    independent = PackerTask(b.pose, PackerPalette())
    np.testing.assert_array_equal(
        b.numpy(independent.per_block_is_block_type_allowed), original
    )
    np.testing.assert_array_equal(
        b.numpy(independent.per_block_considered_block_types), choices
    )
    task.restrict_absent_name3s({"ALA", "GLY"}, selected)
    np.testing.assert_array_equal(b.numpy(task.per_block_is_block_type_allowed), before)
    # Framework execution supports the same bundle protocol without requiring
    # a specific sampler implementation or changing its registration identity.
    samplers = current_samplers(b)

    class Bundle:
        def expand_into_task(self, destination):
            for sampler in samplers:
                destination.add_conformer_sampler(sampler)

    task.add_conformer_sampler(Bundle())
    task.add_conformer_sampler(Bundle())
    assert all(
        any(value is sampler for value in task.conformer_samplers)
        for sampler in samplers
    )
    assert len({id(s) for s in task.conformer_samplers}) == len(task.conformer_samplers)
