"""Public palette context remains host metadata through JAX transformations."""

import jax
import jax.numpy as jnp
import numpy as np

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
