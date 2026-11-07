"""The common renderer accepts frontend conversion without loading frameworks."""

import builtins
import sys

import biotite.structure as struc
import numpy
import pytest

from tmol_core.io import visualize


def test_pose_conversion_is_injected_and_host_inputs_do_not_use_it(monkeypatch):
    initial_modules = set(sys.modules)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"tmol", "torch", "jax", "tmol_jax", "atomworks"}:
            raise AssertionError(f"framework imported by shared rendering: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    atoms = struc.AtomArray(1)
    atoms.coord = numpy.array([[1.25, 2.5, 3.75]])
    atoms.chain_id = numpy.array(["B"])
    atoms.res_id = numpy.array([17])
    atoms.res_name = numpy.array(["ZN"])
    atoms.atom_name = numpy.array(["ZN"])
    atoms.element = numpy.array(["Zn"])
    atoms.hetero = numpy.array([True])
    token = object()
    calls = []

    def convert(model):
        calls.append(model)
        return atoms if model is token else NotImplemented

    text = visualize._pdb_text(token, _pose_to_atom_array=convert)
    assert calls == [token]
    assert "HETATM" in text and "  1.250   2.500   3.750" in text
    assert visualize._first_model_atom_serials(text) == [1]
    assert visualize._pdb_text(atoms, _pose_to_atom_array=convert) == text
    assert visualize._pdb_text(text, _pose_to_atom_array=convert) == text
    assert calls == [token]
    with pytest.raises(TypeError, match="received"):
        visualize._pdb_text(object(), _pose_to_atom_array=convert)

    assert not {"torch", "jax", "tmol", "tmol_jax", "atomworks"}.intersection(
        set(sys.modules) - initial_modules
    )
