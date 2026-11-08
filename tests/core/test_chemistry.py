"""One chemistry contract consumed by the core and optional frontends."""

import importlib
from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
import sys

import attr
import cattr
import pytest

from tmol_core import chemical
from tmol_core.chemical.patching import PatchedChemicalDatabase
from tmol_core.data import chemical_directory
from tmol_core.extern.pysmiles import read_smiles, write_smiles


@pytest.fixture(scope="module")
def raw_database():
    return chemical.ChemicalDatabase.from_file(chemical_directory())


@pytest.mark.parametrize("frontend", ["tmol_core", "tmol", "tmol_jax"])
def test_shared_records_and_patching(frontend, raw_database):
    if frontend == "tmol_core":
        schema = chemical
        patching = importlib.import_module("tmol_core.chemical.patching")
    else:
        framework = "torch" if frontend == "tmol" else "jax"
        if find_spec(framework) is None:
            pytest.skip(f"{framework} execution frontend is not installed")
        schema = importlib.import_module(f"{frontend}.database.chemical")
        patching = importlib.import_module(
            "tmol.database._patched_chemdb"
            if frontend == "tmol"
            else "tmol_jax.chemical.patched_chemdb"
        )
    assert schema.RawResidueType is chemical.RawResidueType
    # Termini, disulfide-ready cysteine, histidine tautomers and D chemistry
    # all pass through the same patch engine, with frontend refinement last.
    names = {"CYS", "CYD", "HIS", "HIS_D", "HIS_POS", "DALA"}
    subset = attr.evolve(
        raw_database,
        residues=tuple(r for r in raw_database.residues if r.name in names),
    )
    expected = PatchedChemicalDatabase.from_chem_db(subset)
    actual = patching.PatchedChemicalDatabase.from_chem_db(subset)
    assert [r.name for r in actual.residues] == [r.name for r in expected.residues]
    assert any(":nterm" in r.name for r in actual.residues)
    assert any(":cterm" in r.name for r in actual.residues)
    for a, b in zip(actual.residues, expected.residues):
        for field in attr.fields(chemical.RawResidueType):
            assert getattr(a, field.name) == getattr(b, field.name)

    if frontend != "tmol_jax":
        return
    from tmol_jax.database.chemical import Icoor, IcoorVariant
    from tmol_jax.chemical.patched_chemdb import update_icoor

    source = Icoor("A", 1.5, 0.8, 1.2, "B", "C", "D")
    mapping = {"<A>": "A"}
    inherited = update_icoor((source,), (IcoorVariant("A", source="<A>"),), (), mapping)
    zeroed = update_icoor(
        (source,), (IcoorVariant("A", source="<A>", phi=0.0),), (), mapping
    )
    assert inherited[0].phi == 1.5
    assert zeroed[0].phi == 0.0
    assert zeroed[0].theta == 0.8

    import attrs
    import numpy as np
    from tmol_jax.database import ParameterDatabase
    from tmol_jax.chemical.restypes import _refined_from_raw
    from tmol_jax.pose.packed_block_types import PackedBlockTypes

    database = ParameterDatabase.get_default().chemical
    raw = next(residue for residue in database.residues if residue.name == "ALA")
    residue = _refined_from_raw(attrs.evolve(raw, atoms=raw.atoms[::-1]))
    packed = PackedBlockTypes.from_restypes(database, [residue])
    nitrogen = packed.ideal_coords[0, residue.atom_to_idx["N"]]
    alpha_carbon = packed.ideal_coords[0, residue.atom_to_idx["CA"]]
    np.testing.assert_allclose(nitrogen, 0, atol=1e-7)
    np.testing.assert_allclose(
        np.linalg.norm(alpha_carbon - nitrogen),
        residue.icoors[residue.icoors_index["CA"]].d,
        atol=1e-6,
    )

    import attrs
    from tmol_jax.database.chemical import ChemicalDatabase
    from tmol_jax.chemical.patched_chemdb import PatchedChemicalDatabase
    from tmol_jax.pose.packed_block_types import PackedBlockTypes

    raw = ChemicalDatabase.get_default()
    original = PatchedChemicalDatabase.from_chem_db(raw)
    alanine = next(residue for residue in raw.residues if residue.name == "ALA")
    custom = attrs.evolve(
        alanine, name="UXA", base_name="UXA", name3="UXA", io_equiv_class="UXA"
    )
    extended = original.with_added_residues((custom,))
    assert isinstance(extended, PatchedChemicalDatabase)
    assert not any(r.name == "UXA" for r in original.residues)
    palette = PackedBlockTypes.default(extended)
    names = {r.name for r in palette.active_block_types}
    assert {"UXA", "UXA:nterm", "UXA:cterm"} <= names
    base = next(r for r in palette.active_block_types if r.name == "UXA")
    assert tuple(a.name for a in base.atoms) == tuple(a.name for a in alanine.atoms)


@pytest.mark.parametrize("geometry", ["linear", "tetrahedral", "octahedral"])
def test_metal_site_capacity(geometry):
    fields = {f.name: f for f in attr.fields(chemical.MetalSite)}
    assert "internal_satisfiers" in fields
    site = chemical.MetalSite(metal_atom="ZN", geometry=geometry)
    assert site.n_free_sites == chemical.GEOMETRY_SITE_COUNT[geometry]
    assert (
        attr.evolve(site, internal_satisfiers=("S1",)).n_free_sites
        == site.n_free_sites - 1
    )


def test_custom_schema_validation(raw_database):
    residue = raw_database.residues[0]
    with pytest.raises(ValueError, match="contains ':'"):
        attr.evolve(residue, base_name="invalid:base")
    with pytest.raises(ValueError, match="CoordinationGeometry"):
        cattr.structure("not-a-geometry", chemical.CoordinationGeometry)
    with pytest.raises(ValueError, match="AcceptorHybridization"):
        cattr.structure("not-a-hybridization", chemical.AcceptorHybridization)
    bonds = [{"bonds": [["N", "CA"], ["CA", "C", "DOUBLE"]]}]
    assert chemical._normalize_bond_tuples(bonds)[0]["bonds"] == [
        ["N", "CA", "SINGLE", False],
        ["CA", "C", "DOUBLE", False],
    ]


@pytest.mark.parametrize("smiles", ["CCO", "c1ccccc1", "[NH4+]", "C1CCCCC1"])
def test_smiles_graph_roundtrip(smiles):
    graph = read_smiles(smiles)
    restored = read_smiles(write_smiles(graph))
    import networkx as nx

    assert nx.is_isomorphic(
        graph,
        restored,
        node_match=lambda a, b: all(
            a.get(k) == b.get(k) for k in ("element", "charge", "hcount", "aromatic")
        ),
        edge_match=lambda a, b: a["order"] == b["order"],
    )


def test_core_import_and_default_data_need_no_framework(tmp_path):
    source = Path(__file__).resolve().parents[2]
    script = """
import importlib.abc, json, sys
class NoFrontend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "jax", "jaxlib", "tmol", "tmol_jax"}:
            raise AssertionError("core imported " + fullname)
sys.meta_path.insert(0, NoFrontend())
from tmol_core.chemical import ChemicalDatabase
from tmol_core.chemical.patching import PatchedChemicalDatabase
from tmol_core.chemical.restypes import refined_from_raw
from tmol_core.chemical.packed import PackedBlockTypesData
import tmol_core.scoring.elec
import tmol_core.scoring.genbonded, tmol_core.scoring.count_pair
import tmol_core.scoring.atom_types, tmol_core.scoring.ljlk, tmol_core.scoring.hbond
from tmol_core.chemical.metal_donors import _metal_donor_patch
patch = _metal_donor_patch("HIS", "NE2", 2)
assert [c.name for c in patch.add_connections] == ["metal_NE2", "metal2_NE2"]
assert all(not c.kinematic for c in patch.add_connections)
import tmol_core.scoring.na_torsion_params, tmol_core.scoring.na_torsion_topology
import tmol_core.database.na_torsion
import tmol_core.scoring.metal, tmol_core.database.metal_coordination
import tmol_core.scoring.backbone_torsion, tmol_core.scoring.dunbrack
import tmol_core.scoring.dunbrack_topology
import tmol_core.pose.bond_separation
import tmol_core.scoring.cartbonded, tmol_core.scoring.cartbonded_connections, tmol_core.scoring.atom_ids
import tmol_core.database.context, tmol_core.database.mirrored_dunbrack
import tmol_core.database.ljlk, tmol_core.database.hbond
import tmol_core.database.cartbonded, tmol_core.database.generated
import tmol_core.database.elec, tmol_core.database.ref, tmol_core.database.disulfide
raw = ChemicalDatabase.get_default()
PackedBlockTypesData.from_restypes(raw, [refined_from_raw(residue) for residue in raw.residues])
print(json.dumps({"raw": len(raw.residues), "patched": len(PatchedChemicalDatabase.from_chem_db(raw).residues)}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(source)},
        check=True,
        capture_output=True,
        text=True,
    )
    counts = json.loads(result.stdout)
    assert counts["raw"] > 100
    assert counts["patched"] > counts["raw"]
