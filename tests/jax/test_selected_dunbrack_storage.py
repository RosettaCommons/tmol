"""Library storage follows actual scored identities without narrowing palettes."""

import attrs
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tmol_jax import ParameterDatabase, PackedBlockTypes
from tmol_jax.io import pose_stack_from_canonical_form
from tmol_jax.score.dunbrack.dunbrack_energy_term import DunbrackEnergyTerm
from tests.scientific.molecules import MolecularCase
from tests.scientific.test_canonical_molecular_input import canonical_input


class _FullLibraryTerm(DunbrackEnergyTerm):
    def _packed_parameters(self, pbt, coordinate_dtype, table_indices):
        # The existing common full-library preparation and native scorer form
        # the independent storage-layout reference; no physics is reimplemented.
        return super()._packed_parameters(pbt, coordinate_dtype, None)


@pytest.mark.parametrize("custom", (False, True))
def test_selected_libraries_preserve_full_palette_and_reused_scoring_context(custom):
    original_database = database = ParameterDatabase.get_default()
    if custom:
        library = database.scoring.dunbrack
        lookup = {row.residue_name: row.dun_table_name for row in library.dun_lookup}
        replacements = dict(LEU="ILE", PHE="TYR", DLEU="DILE", DPHE="DTYR")
        library = attrs.evolve(
            library,
            dun_lookup=tuple(
                (
                    attrs.evolve(
                        row, dun_table_name=lookup[replacements[row.residue_name]]
                    )
                    if row.residue_name in replacements
                    else row
                )
                for row in library.dun_lookup
            ),
        )
        database = attrs.evolve(
            database, scoring=attrs.evolve(database.scoring, dunbrack=library)
        )
    device = jax.devices()[0]
    with jax.default_device(device), jax.enable_x64():
        palette = PackedBlockTypes.default(database.chemical)
        names = tuple(bt.name for bt in palette.active_block_types)
        term = DunbrackEnergyTerm(database)
        full_term = _FullLibraryTerm(database)
        retained = []
        for identity in ("LEU", "PHE", "DLEU", "DPHE", "LEU"):
            case = MolecularCase((), ((identity,),), ((1,),))
            ordering, canonical = canonical_input(case, database)
            pose = pose_stack_from_canonical_form(
                dict(canonical, canonical_ordering=ordering),
                palette,
                compact_padding=True,
            )
            pose = pose.replace(coords=pose.coords.astype(jnp.float64))
            selected, full = [t.setup_for_pose_stack(pose) for t in (term, full_term)]
            selected_bytes = sum(a.nbytes for a in selected.parameters)
            full_bytes = sum(a.nbytes for a in full.parameters)
            assert 0 < selected_bytes < full_bytes / 4

            def score(x, state):
                return jnp.stack(tuple(DunbrackEnergyTerm.evaluate(state, x).values()))

            execute = jax.jit(score)
            derivative = jax.jit(jax.jacrev(score))
            expected = execute(pose.coords, full), derivative(pose.coords, full)
            actual = execute(pose.coords, selected), derivative(pose.coords, selected)
            assert np.max(np.abs(expected[0])) > 1e-3
            assert np.linalg.norm(expected[1]) > 1e-3
            if custom and identity == "LEU":
                original = _FullLibraryTerm(original_database).setup_for_pose_stack(
                    pose
                )
                assert (
                    np.max(np.abs(execute(pose.coords, original) - expected[0])) > 1e-3
                )
            for value, reference in zip(actual, expected):
                assert value.dtype == jnp.float64 and value.device == device
                np.testing.assert_allclose(value, reference, rtol=2e-10, atol=2e-10)
            retained.append((pose.coords, selected, actual))
            assert pose.packed_block_types is palette
            assert tuple(bt.name for bt in palette.active_block_types) == names
        # Later table selections must not invalidate previously rendered states.
        for coords, state, reference in retained:
            for value, expected in zip(
                (execute(coords, state), derivative(coords, state)), reference
            ):
                np.testing.assert_array_equal(value, expected)


def test_selected_library_preparation_validates_unselected_lookup_entries():
    from tmol_jax.score.dunbrack.parameters import prepare_dunbrack_parameters

    database = ParameterDatabase.get_default()
    library = database.scoring.dunbrack
    invalid = attrs.evolve(
        library.dun_lookup[0], residue_name="UNUSED", dun_table_name="missing"
    )
    for extra in (invalid, library.dun_lookup[-1]):
        source = attrs.evolve(library, dun_lookup=(*library.dun_lookup, extra))
        changed = attrs.evolve(
            database, scoring=attrs.evolve(database.scoring, dunbrack=source)
        )
        with pytest.raises(ValueError, match="missing tables|must be unique"):
            prepare_dunbrack_parameters(changed, table_indices=(0,))
