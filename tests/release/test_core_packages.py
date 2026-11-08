"""Core dependencies must be explicit, versioned and independently installable."""

import io
from pathlib import Path
import tarfile
import tomllib
from zipfile import ZipFile

import pytest

from package_requirements import requires_core

ROOT = Path(__file__).resolve().parents[2]


def test_frontends_require_matching_core_and_own_separate_namespaces():
    torch = tomllib.loads((ROOT / "pyproject.toml").read_text())
    jax = tomllib.loads((ROOT / "packages/jax/pyproject.toml").read_text())
    core = tomllib.loads((ROOT / "packages/core/pyproject.toml").read_text())
    dependency = f'tmol-core=={torch["project"]["version"]}'
    assert dependency in torch["project"]["dependencies"]
    assert dependency in jax["project"]["dependencies"]
    assert torch["tool"]["scikit-build"]["wheel"]["packages"] == ["tmol"]
    assert jax["tool"]["setuptools"]["packages"]["find"]["include"] == ["tmol_jax*"]
    assert core["tool"]["setuptools"]["packages"]["find"]["include"] == ["tmol_core*"]
    assert all(
        "torch" not in r and "jax" not in r for r in core["project"]["dependencies"]
    )


@pytest.mark.parametrize(
    "requirement,expected",
    [
        ("tmol-core==0.2.0", True),
        ("tmol_core >= 0.2", True),
        ("tmol-core-other==0.2.0", False),
        ("torch>=2.11", False),
    ],
)
@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_candidate_metadata_selects_core_without_imports(
    tmp_path, requirement, expected, kind
):
    metadata = (
        f"Metadata-Version: 2.4\nName: tmol\nRequires-Dist: {requirement}\n".encode()
    )
    if kind == "wheel":
        path = tmp_path / "tmol-0.2.0-py3-none-any.whl"
        with ZipFile(path, "w") as archive:
            archive.writestr("tmol-0.2.0.dist-info/METADATA", metadata)
    else:
        path = tmp_path / "tmol-0.2.0.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            member = tarfile.TarInfo("tmol-0.2.0/PKG-INFO")
            member.size = len(metadata)
            archive.addfile(member, io.BytesIO(metadata))
    assert requires_core(path) == expected
