"""Shared scalar parameter records preserve custom database extensions."""

import importlib
from importlib.util import find_spec
from pathlib import Path

import attr
import pytest
import yaml

from tmol_core.database.elec import ElecDatabase
from tmol_core.database.ref import RefDatabase


def test_generated_electrostatic_and_reference_rows(tmp_path):
    electrostatics = {
        "global_parameters": {
            "elec_min_dis": 1.5,
            "elec_max_dis": 5.5,
            "elec_sigmoidal_die_D": 80.0,
            "elec_sigmoidal_die_D0": 6.0,
            "elec_sigmoidal_die_S": 0.4,
        },
        "atom_charge_parameters": [{"res": "LIG", "atom": "C1", "charge": -0.2}],
        "atom_cp_reps_parameters": [],
    }
    extension = {
        "atom_charge_parameters": [{"res": "LIG:nterm", "atom": "C1", "charge": 0.3}]
    }
    base = tmp_path / "elec.yaml"
    extra = tmp_path / "extra.yaml"
    base.write_text(yaml.safe_dump(electrostatics))
    extra.write_text(yaml.safe_dump(extension))
    plain = ElecDatabase.from_file(base)
    extended = ElecDatabase.from_file(base, generated=[tmp_path / "absent.yaml", extra])
    assert [(r.res, r.charge) for r in extended.atom_charge_parameters] == [
        ("LIG", -0.2),
        ("LIG:nterm", 0.3),
    ]
    assert len(plain.atom_charge_parameters) == 1
    base.write_text("weights: {LIG: 1.0, ALA: 2.0}\n")
    extra.write_text("weights: {LIG: -3.0}\n")
    assert RefDatabase.from_file(base, generated=[extra]).weights == {
        "LIG": -3.0,
        "ALA": 2.0,
    }
    assert RefDatabase.from_file(base).weights == {"LIG": 1.0, "ALA": 2.0}


@pytest.mark.parametrize("frontend", ["tmol", "tmol_jax"])
@pytest.mark.parametrize(
    "term,cls",
    [
        ("elec", "ElecDatabase"),
        ("ref", "RefDatabase"),
        ("disulfide", "DisulfideDatabase"),
    ],
)
def test_frontends_share_the_canonical_scalar_loader(frontend, term, cls):
    if find_spec("torch" if frontend == "tmol" else "jax") is None:
        pytest.skip("execution frontend is not installed")
    module = importlib.import_module(
        f"{frontend}.database.scoring.{('_' if frontend == 'tmol' else '')}{term}"
    )
    core = importlib.import_module(f"tmol_core.database.{term}")
    assert getattr(module, cls) is getattr(core, cls)


def test_jax_custom_angle_policy_and_recursive_records():
    if find_spec("jax") is None:
        pytest.skip("JAX frontend is not installed")
    import math
    from tmol_jax.database.scoring._structure import Angle, structure
    from tmol_jax.database.scoring.ljlk import LJLKDatabase

    # Existing JAX custom tables allow unit-qualified angles outside one turn.
    assert structure("720 degrees", Angle) == 4 * math.pi
    path = (
        Path(__file__).resolve().parents[2] / "tmol/database/default/scoring/ljlk.yaml"
    )
    if not path.exists():
        from tmol_jax.database import default_directory

        path = default_directory() / "scoring/ljlk.yaml"
    database = LJLKDatabase.from_file(path)
    assert attr.has(type(database.global_parameters))
    assert isinstance(database.atom_type_parameters, tuple)
    assert all(isinstance(p.lj_radius, float) for p in database.atom_type_parameters)


def test_generated_cartesian_records_and_connection_identity(tmp_path):
    from tmol_core.database.cartbonded import CartBondedDatabase

    empty = dict(
        length_parameters=[],
        angle_parameters=[],
        torsion_parameters=[],
        improper_parameters=[],
        hxltorsion_parameters=[],
    )
    base = tmp_path / "cart.yaml"
    extra = tmp_path / "extra.yaml"
    base.write_text(yaml.safe_dump({"residue_params": {"CUSTOM": empty}}))
    changed = {**empty, "length_parameters": [dict(atm1="A", atm2="B", x0=1.5, K=200)]}
    connection = dict(
        block_type1="CUSTOM",
        connection1="link",
        block_type2="CUSTOM",
        connection2="other",
        length_parameters=[],
        angle_parameters=[],
    )
    extra.write_text(
        yaml.safe_dump(
            {
                "residue_params": {"CUSTOM:variant": changed},
                "connection_params": [connection],
            }
        )
    )
    plain = CartBondedDatabase.from_file(base)
    extended = CartBondedDatabase.from_file(base, [extra])
    assert extended.residue_params["CUSTOM:variant"].length_parameters[0].K == 200
    assert extended.connection_params[0].connection2 == "other"
    assert plain.hash != extended.hash
    assert "CUSTOM:variant" not in plain.residue_params


def test_scoring_root_preserves_custom_overlays_and_required_files(tmp_path):
    import shutil
    from tmol_core.database.scoring import load_scoring_parameters

    source = Path(__file__).resolve().parents[2] / "tmol/database/default/scoring"
    for filename in source.glob("*.yaml"):
        shutil.copyfile(filename, tmp_path / filename.name)
    # Custom generated parameters must survive assembly without changing the
    # original database or requiring a particular binary table representation.
    (tmp_path / "ref_d_amino_acids.yaml").write_text("weights: {CUSTOM: -3.5}\n")
    (tmp_path / "elec_metal_ions.yaml").write_text(
        "atom_charge_parameters: [{res: CUSTOM, atom: ZN, charge: 1.75}]\n"
    )
    tables = [object(), object(), object()]

    def provider(table):
        def load(root):
            assert root == tmp_path
            return table

        return load

    providers = dict(
        zip(("dun_loader", "rama_loader", "omega_loader"), map(provider, tables))
    )
    scoring = load_scoring_parameters(tmp_path, **providers)
    assert [scoring[k] for k in ("dun", "rama", "omega_bbdep")] == tables
    assert scoring["ref"].weights["CUSTOM"] == -3.5
    assert any(
        (r.res, r.atom, r.charge) == ("CUSTOM", "ZN", 1.75)
        for r in scoring["elec"].atom_charge_parameters
    )
    assert "CUSTOM" not in RefDatabase.from_file(source / "ref.yaml").weights
    (tmp_path / "hbond.yaml").unlink()
    with pytest.raises(FileNotFoundError, match="hbond.yaml"):
        load_scoring_parameters(tmp_path, **providers)
