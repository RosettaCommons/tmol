"""Public file helpers preserve custom molecular identity and declared bonds."""

import numpy as np
import pytest

from .fragment_molecules import fragment_fixture
from .test_fragment_input import _build, _frontend


@pytest.mark.parametrize("format", ("pdb", "cif"))
def test_public_structure_files_keep_custom_ligand_atoms_and_physics(
    backend, format, tmp_path
):
    frontend = _frontend(backend)
    whole, _, parameters = fragment_fixture(mixed=True)
    # PDB readers normalize blank chain labels; give every source chain an
    # explicit representable identity before requiring an exact round trip.
    whole.chain_id[whole.chain_id == ""] = "A"
    pose, context = _build(backend, whole, parameters)
    path = tmp_path / f"mixed.{format}"
    assert frontend.file_from_pose_stack(pose, path) == path
    atoms = frontend.atom_array_from_file(path)
    text = frontend.file_from_atom_array(atoms, format=format)
    assert isinstance(text, str) and len(text) > 100
    exported = frontend.atom_array_from_pose_stack(pose)
    for field in ("atom_name", "res_name", "res_id", "chain_id", "element"):
        np.testing.assert_array_equal(getattr(atoms, field), getattr(exported, field))
    np.testing.assert_allclose(atoms.coord, exported.coord, rtol=0, atol=5.1e-4)
    rebuilt = frontend.pose_stack_from_file(
        path, backend.device, context=context, no_optH=True
    )
    np.testing.assert_array_equal(
        backend.numpy(rebuilt.block_type_ind), backend.numpy(pose.block_type_ind)
    )
    np.testing.assert_array_equal(
        backend.numpy(rebuilt.inter_residue_connections),
        backend.numpy(pose.inter_residue_connections),
    )
    backend.pose, backend.parameter_database = rebuilt, context.parameter_database
    channels, _ = backend.use_beta2016()
    values, gradients = backend.channel_values_and_gradients(backend.coords())
    assert np.isfinite(values).all() and np.isfinite(gradients).all()
    gen = channels.index("gen_torsions")
    assert np.linalg.norm(values[gen]) > 1e-3
    assert np.linalg.norm(gradients[gen]) > 1e-3

    if backend.framework != "jax" or format != "pdb":
        return
    from tmol_jax.io import canonical_form_from_pdb
    from .test_canonical_molecular_input import CANONICAL_PROTEIN_PDB

    lines = CANONICAL_PROTEIN_PDB.read_text().splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith("ATOM"))
    lines[first] = lines[first][:12] + " ZZ " + lines[first][16:]
    with pytest.raises(ValueError, match="atom ZZ"):
        canonical_form_from_pdb("\n".join(lines))
