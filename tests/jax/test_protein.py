"""Small actual-protein scoring smoke test without skip-on-error fixtures."""

from pathlib import Path

import jax
import numpy as np

from tmol_jax import pose_stack_from_pdb, beta2016_score_function


def _tripeptide_pdb(tmp_path):
    source = Path(__file__).parents[2] / "tmol/tests/data/pdb/1ubq.pdb"
    lines = [
        line
        for line in source.read_text().splitlines()
        if line.startswith("ATOM") and 1 <= int(line[22:26]) <= 3
    ]
    path = tmp_path / "tripeptide.pdb"
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


def test_protein_score_and_gradient_are_finite(tmp_path):
    from tmol_jax.io import default_packed_block_types

    path = _tripeptide_pdb(tmp_path)
    palette = default_packed_block_types()
    palette_names = tuple(residue.name for residue in palette.active_block_types)
    pose = pose_stack_from_pdb(path, packed_block_types=palette)
    from biotite.structure.io.pdb import PDBFile
    from biotite.structure import stack
    from tmol_jax.io import canonical_form_from_pdb
    from tmol_jax.io.pose_stack_from_biotite import canonical_form_from_biotite

    atoms = PDBFile.read(path).get_structure(model=1)
    canonical = canonical_form_from_pdb(path)
    batched = canonical_form_from_biotite(stack([atoms, atoms]))
    for field in ("coords", "chain_id", "res_types", "atom_is_present"):
        np.testing.assert_array_equal(
            batched[field], np.repeat(canonical[field], 2, axis=0)
        )
    score_function = beta2016_score_function()
    energy, gradient = score_function.jitted_value_and_grad(pose)(pose.coords)
    assert tuple(
        residue.name for residue in pose.packed_block_types.active_block_types
    ) == palette_names
    np.testing.assert_array_equal(pose.n_res_per_pose, [3])
    assert [pose.block_type(0, i).name3 for i in range(3)] == ["MET", "GLN", "ILE"]
    used_types = np.asarray(pose.block_type_ind)
    assert len(np.unique(used_types[used_types >= 0])) == 3
    # Cartesian setup must depend on topology, even when a real atom is at zero.
    cart = next(
        term
        for term in score_function.terms
        if type(term).__name__ == "CartBondedEnergyTerm"
    )
    shifted = pose.replace(coords=pose.coords - pose.coords[:, :1])
    first = cart.setup_for_pose_stack(pose)
    second = cart.setup_for_pose_stack(shifted)
    for left, right in zip(jax.tree.leaves(first), jax.tree.leaves(second)):
        np.testing.assert_array_equal(left, right)
    moved = pose.to(jax.devices()[0])
    np.testing.assert_array_equal(moved.coords, pose.coords)
    assert moved.device == pose.device
    assert energy.shape == (1,)
    assert gradient.shape == pose.coords.shape
    assert np.isfinite(energy).all()
    assert np.isfinite(gradient).all()
    assert np.linalg.norm(gradient) > 0


def test_mixed_batch_matches_individual_scoring_and_masks_padding(tmp_path):
    from tmol_jax.io import canonical_form_from_pdb, default_packed_block_types
    from tmol_jax.io import pose_stack_from_canonical_form

    canonical = canonical_form_from_pdb(_tripeptide_pdb(tmp_path))
    batch = {name: np.repeat(value, 2, axis=0) for name, value in canonical.items()}
    batch["res_name"][1, -1] = ""
    batch["res_types"][1, -1] = -1
    batch["chain_id"][1, -1] = -1
    batch["coords"][1, -1] = np.nan
    batch["atom_is_present"][1, -1] = False
    pose = pose_stack_from_canonical_form(batch, default_packed_block_types())
    sf = beta2016_score_function()
    energies, gradient = sf.jitted_value_and_grad(pose)(pose.coords)
    assert np.isfinite(energies).all() and np.isfinite(gradient).all()
    np.testing.assert_array_equal(
        np.asarray(gradient)[~np.asarray(pose.atom_is_real)], 0
    )
    for index in range(2):
        individual = pose[index]
        expected = sf.jitted(individual)(individual.coords)
        np.testing.assert_allclose(energies[index], expected[0], atol=1e-4)


def test_pdb_parser_uses_first_model():
    from tmol_jax.io.pdb_parsing import parse_pdb

    def atom(serial, x):
        return (
            f"ATOM  {serial:5d}  CA  ALA A   1    "
            f"{x:8.3f}{2.0:8.3f}{3.0:8.3f}  1.00 10.00           C  "
        )

    text = "\n".join(
        [
            "MODEL        1",
            atom(1, 1.0),
            "ENDMDL",
            "MODEL        2",
            atom(1, 9.0),
            "ENDMDL",
            "END",
        ]
    )
    records = parse_pdb(text)
    np.testing.assert_array_equal(records["coords"], [[1.0, 2.0, 3.0]])


def test_peptide_bond_impropers_match_independent_geometry(tmp_path):
    """Planarity impropers across each peptide bond follow plane-normal geometry."""
    import jax.numpy as jnp
    from tmol_jax.score import ScoreType

    pose = pose_stack_from_pdb(_tripeptide_pdb(tmp_path))
    term = next(
        t
        for t in beta2016_score_function().terms
        if type(t).__name__ == "CartBondedEnergyTerm"
    )
    state = term.setup_for_pose_stack(pose)
    # Twist the peptide groups off planarity, where energies vanish anyway.
    coords = pose.coords + 0.17 * jnp.sin(jnp.arange(pose.coords.size)).reshape(
        pose.coords.shape
    ).astype(pose.coords.dtype)
    disconnected = pose.replace(
        inter_residue_connections=jnp.full_like(pose.inter_residue_connections, -1)
    )
    intra_only = term.setup_for_pose_stack(disconnected)
    inter = (
        term(pose, state, coords)[ScoreType.cart_impropers]
        - term(disconnected, intra_only, coords)[ScoreType.cart_impropers]
    )

    pbt = pose.packed_block_types
    x = np.asarray(coords, dtype=np.float64)[0]

    def xyz(block, name):
        block_type = pbt.active_block_types[int(pose.block_type_ind[0, block])]
        return x[int(pose.block_coord_offset[0, block]) + block_type.atom_to_idx[name]]

    def improper(k2, a, b, c, d):
        first, second = np.cross(b - a, c - b), np.cross(c - b, d - c)
        cosine = first @ second / np.linalg.norm(first) / np.linalg.norm(second)
        return 2 * k2 * (1 - cosine**2)

    expected = 0.0
    for left in (0, 1):
        c, n = xyz(left, "C"), xyz(left + 1, "N")
        expected += improper(9.667, xyz(left, "CA"), n, c, xyz(left, "O"))
        expected += improper(10.458, xyz(left + 1, "CA"), c, n, xyz(left + 1, "H"))
    assert expected > 0.01
    np.testing.assert_allclose(inter[0], expected, rtol=2e-4)


def test_written_pdb_round_trips_coordinates(tmp_path):
    from tmol_jax.io import write_pose_stack_pdb

    pose = pose_stack_from_pdb(_tripeptide_pdb(tmp_path))
    out = tmp_path / "written.pdb"
    write_pose_stack_pdb(pose, out)
    again = pose_stack_from_pdb(out)
    assert again.coords.shape == pose.coords.shape
    np.testing.assert_allclose(again.coords, pose.coords, atol=1e-3)


def test_coordinate_gap_disconnects_without_terminal_variants(tmp_path):
    lines = _tripeptide_pdb(tmp_path).read_text().splitlines()
    shifted = [
        (
            line[:30] + f"{float(line[30:38]) + 6:8.3f}" + line[38:]
            if line.startswith("ATOM") and int(line[22:26]) == 3
            else line
        )
        for line in lines
    ]
    path = tmp_path / "gap.pdb"
    path.write_text("\n".join(shifted) + "\n")
    pose = pose_stack_from_pdb(path)
    names = [pose.block_type(0, i).name for i in range(3)]
    assert not any("cterm" in name for name in names[:2]), names
    connections = np.asarray(pose.inter_residue_connections)[0]
    assert set(connections[1, :, 0]) == {0, -1}
    assert set(connections[2, :, 0]) == {-1}


def test_disulfide_energy_matches_torch_reference():
    """Unweighted disulfide energy of 3PLC's four bonds, from TMol's Torch scorer."""
    from tmol_jax.score import ScoreType

    pose = pose_stack_from_pdb(
        Path(__file__).parents[2] / "tmol/tests/data/pdb/3plc.pdb"
    )
    score_function = beta2016_score_function()
    score = score_function.render_whole_pose_scoring_module(
        pose, sum_terms=False, apply_weights=False
    )
    channel = score_function.active_score_types().index(ScoreType.disulfide)
    energy = np.asarray(score(pose.coords))[channel, 0]
    np.testing.assert_allclose(energy, -3.25716, atol=2e-3)


def test_custom_canonical_atom_order_survives_public_input_adapters(tmp_path):
    import attrs
    from biotite.structure.io.pdb import PDBFile
    from tmol_jax.io import CanonicalOrdering, default_packed_block_types
    from tmol_jax.io.pose_stack_from_biotite import pose_stack_from_biotite

    path = _tripeptide_pdb(tmp_path)
    expected = pose_stack_from_pdb(path)
    pbt = default_packed_block_types()
    # A legitimate chemical database may order atom records differently without
    # changing identities, connectivity or geometry. Canonical slots then differ.
    reordered = attrs.evolve(
        pbt.chem_db,
        residues=tuple(
            attrs.evolve(residue, atoms=tuple(reversed(residue.atoms)))
            for residue in pbt.chem_db.residues
        ),
    )
    ordering = CanonicalOrdering.from_chemdb(reordered)
    atoms = PDBFile.read(path).get_structure(model=1)
    for source, loader in (
        (path, pose_stack_from_pdb),
        (atoms, pose_stack_from_biotite),
    ):
        actual = loader(source, canonical_ordering=ordering, packed_block_types=pbt)
        assert [r.name for r in actual.packed_block_types.active_block_types] == [
            r.name for r in expected.packed_block_types.active_block_types
        ]
        np.testing.assert_array_equal(actual.coords, expected.coords)
        np.testing.assert_array_equal(
            actual.inter_residue_connections, expected.inter_residue_connections
        )
    np.testing.assert_array_equal(pose_stack_from_pdb(path).coords, expected.coords)


def test_canonical_reconstruction_preserves_float64_observations(tmp_path):
    from tmol_jax.io import (
        canonical_form_from_pdb,
        default_canonical_ordering,
        default_packed_block_types,
        pose_stack_from_canonical_form,
    )

    canonical = canonical_form_from_pdb(_tripeptide_pdb(tmp_path))
    canonical["coords"] = canonical["coords"].astype(np.float64) + 1.23456789e-8
    with jax.enable_x64():
        pose = pose_stack_from_canonical_form(canonical, default_packed_block_types())
    assert pose.coords.dtype == np.float64
    co = default_canonical_ordering()
    for block in range(pose.max_n_blocks):
        residue = pose.block_type(0, block)
        offset = int(pose.block_coord_offset[0, block])
        mapping = co.restypes_atom_index_mapping[residue.io_equiv_class]
        for atom, record in enumerate(residue.atoms):
            source = mapping[record.name]
            if canonical["atom_is_present"][0, block, source]:
                np.testing.assert_array_equal(
                    pose.coords[0, offset + atom], canonical["coords"][0, block, source]
                )
    assert np.isfinite(pose.coords).all()
