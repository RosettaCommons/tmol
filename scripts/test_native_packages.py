"""The optional backends share a release and leave Torch minors unconstrained."""

import tomllib
from pathlib import Path
from packaging.requirements import Requirement


def test_native_package_metadata():
    root = Path(__file__).resolve().parents[1]
    base = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    for major in (12, 13):
        project = tomllib.loads(
            (root / f"packages/cu{major}/pyproject.toml").read_text()
        )["project"]
        assert project["name"] == f"tmol-ops-cu{major}"
        assert project["version"] == base["version"]
        assert f"tmol=={base['version']}" in project["dependencies"]
        assert base["optional-dependencies"][f"cu{major}"] == [
            f"{project['name']}=={base['version']}"
        ]
        for dependencies in (base["dependencies"], project["dependencies"]):
            torch = next(
                Requirement(r) for r in dependencies if Requirement(r).name == "torch"
            )
            assert str(torch.specifier) == ">=2.11"


def test_core_dependencies_leave_optional_and_transitive_packages_to_their_owners():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    core = {Requirement(r).name for r in project["dependencies"]}
    assert core.isdisjoint(
        {
            "openbabel-wheel",
            "attrs_strict",
            "pyarrow",
            "sparse",
            "typing_extensions",
            "llvmlite",
        }
    )
    extras = project["optional-dependencies"]
    assert any(Requirement(r).name == "openbabel-wheel" for r in extras["ligand"])
    for extra in ("dev", "docs"):
        assert any(
            Requirement(r).name == "tmol" and "ligand" in Requirement(r).extras
            for r in extras[extra]
        )
