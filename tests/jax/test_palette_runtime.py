"""Public palette context remains host metadata through JAX transformations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tmol_jax import PackedBlockTypes
from tmol_jax.chemical import ResidueTypeSet
from tests.scientific.execution import Execution
from tests.scientific.molecules import DOMAIN_CASES


def test_explicit_palette_context_survives_jit_and_vmap():
    b = Execution("jax-" + jax.devices()[0].platform).molecule(
        DOMAIN_CASES["mixed_batch"], terms=()
    )
    original = b.pose.packed_block_types
    lookup = jax.jit(lambda palette, indices: palette.n_atoms[indices])
    batched = jax.jit(jax.vmap(lookup, in_axes=(None, 0)))
    indices = jnp.array([[0, original.n_types - 1], [1, 0]])
    expected = np.array([len(t.atoms) for t in original.active_block_types])
    for _ in range(2):
        collection = ResidueTypeSet.from_database(original.chem_db)
        palette = PackedBlockTypes.from_restype_list(
            original.chem_db, collection, original.active_block_types, b.device
        )
        np.testing.assert_array_equal(lookup(palette, indices[0]), expected[[0, -1]])
        np.testing.assert_array_equal(batched(palette, indices), expected[indices])
        restored = jax.jit(lambda p: p)(palette)
        assert restored.restype_set is collection
        assert restored.chem_db is original.chem_db
        assert all(leaf.device == b.device for leaf in jax.tree.leaves(restored))
        torsions = jax.jit(lambda p, i: (p.n_torsions[i], p.torsion_uaids[i]))
        count, atoms = jax.jit(jax.vmap(torsions, in_axes=(None, 0)))(palette, indices)
        np.testing.assert_array_equal(count, np.asarray(palette.n_torsions)[indices])
        np.testing.assert_array_equal(atoms, np.asarray(palette.torsion_uaids)[indices])


def test_palette_padding_and_empty_batches_keep_device_and_validation():
    from tmol_jax import PackerPalette

    platform = "cuda" if jax.devices()[0].platform == "gpu" else "cpu"
    b = Execution("jax-" + platform).molecule(DOMAIN_CASES["mixed_batch"], terms=())
    palette, pbt = PackerPalette(), b.pose.packed_block_types
    with jax.default_device(jax.devices("cpu")[0]):
        for shape in ((2, 2), (0, 2)):
            original = jax.device_put(np.full(shape, -1, np.int32), b.device)
            counts, choices, same = palette.block_types_from_original(pbt, original)
            repack = palette.create_restrict_to_repacking_mask(pbt, original)
            assert all(a.device == b.device for a in (counts, choices, same, repack))
            assert counts.dtype == choices.dtype == np.int32
            assert same.dtype == repack.dtype == np.bool_
            assert counts.shape == shape and choices.shape[:2] == shape
            np.testing.assert_array_equal(counts, 0)
            np.testing.assert_array_equal(choices, -1)
            assert not np.asarray(same).any() and not np.asarray(repack).any()
        for bad in (-2, pbt.n_types):
            original = jax.device_put(np.array([[bad]], np.int32), b.device)
            for method in (
                palette.block_types_from_original,
                palette.create_restrict_to_repacking_mask,
            ):
                with pytest.raises(ValueError, match="outside the palette"):
                    method(pbt, original)
