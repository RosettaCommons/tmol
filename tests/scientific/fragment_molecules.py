"""A real protein-ligand structure with a named chemical bridge cut."""

from pathlib import Path
import io

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
from scipy.sparse.csgraph import connected_components
import zstandard

from tmol_core.ligand import load_params_file, FRAGMENT_ID_ANNOTATION


def fragment_fixture(*, repeated=False, mixed=True):
    data = Path(__file__).parents[2] / "tmol/tests/data/protein_ligand_test"
    path = data / "ace.tmol.nomin.cif.zst"
    if not path.exists():
        path = path.with_suffix("")
    raw = path.read_bytes()
    if path.suffix == ".zst":
        raw = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw)).read()
    atoms = pdbx.get_structure(
        pdbx.CIFFile.read(io.StringIO(raw.decode())), model=1, include_bonds=True
    )
    params = data / "ace.xtal-lig.mmff94.tmol"
    residue = load_params_file(params)[0].residue_type
    by_name = {atom.name: i for i, atom in enumerate(residue.atoms)}
    graph = np.zeros((len(by_name), len(by_name)), bool)
    cut = frozenset(("C1", "C2"))
    assert any(frozenset(bond[:2]) == cut for bond in residue.bonds)
    for left, right, *_ in residue.bonds:
        if frozenset((left, right)) != cut:
            graph[by_name[left], by_name[right]] = graph[
                by_name[right], by_name[left]
            ] = True
    count, components = connected_components(graph, directed=False)
    assert count == 2
    if not mixed:
        atoms = atoms[atoms.res_name == "LG1"]
    if repeated:
        copy = atoms[atoms.res_name == "LG1"].copy()
        copy.chain_id[:] = "Z"
        copy.coord += [15, 3, 2]
        atoms = struc.concatenate([atoms, copy])
    annotated = atoms.copy()
    labels = np.zeros(len(atoms), np.int32)
    for i in np.flatnonzero(atoms.res_name == "LG1"):
        labels[i] = 1 + components[by_name[atoms.atom_name[i]]]
    annotated.set_annotation(FRAGMENT_ID_ANNOTATION, labels)
    return atoms, annotated, params
