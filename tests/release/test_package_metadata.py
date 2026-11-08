"""Package dependency, namespace, optional-backend and license contracts."""

import io
from pathlib import Path
import tarfile
import tomllib
from zipfile import ZipFile

import pytest
from packaging.requirements import Requirement

from package_requirements import requires_core
from build_jax_native import stage_native

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


def test_native_package_metadata():
    base = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    for major in (12, 13):
        project = tomllib.loads(
            (ROOT / f"packages/cu{major}/pyproject.toml").read_text()
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


def test_native_license_files_and_metadata(tmp_path):
    stage_native(ROOT, tmp_path)
    project = tomllib.loads((ROOT / "packages/jax-native/pyproject.toml").read_text())[
        "project"
    ]
    assert set(project["license"].split(" AND ")) == {
        "BSD-3-Clause",
        "Apache-2.0",
        "MPL-2.0",
        "BSD-2-Clause",
        "0BSD",
    }
    assert project["urls"]["Repository"] == "https://github.com/RosettaCommons/tmol"
    declared = {
        path.relative_to(tmp_path)
        for pattern in project["license-files"]
        for path in tmp_path.glob(pattern)
        if path.is_file()
    }
    staged = {
        path.relative_to(tmp_path)
        for path in (tmp_path / "licenses").rglob("*")
        if path.is_file()
    } | {Path("LICENSE"), Path("NOTICE")}
    assert declared == staged
    for source in (ROOT / "tmol/extern/Eigen").glob("COPYING.*"):
        assert (
            tmp_path / "licenses/Eigen" / source.name
        ).read_bytes() == source.read_bytes()
    assert (tmp_path / "LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()
    assert (tmp_path / "licenses/moderngpu-LICENSE.md").read_bytes() == (
        ROOT / "tmol/extern/moderngpu/LICENSE.md"
    ).read_bytes()
    notice = (tmp_path / "NOTICE").read_text()
    assert "Copyright 2023 The OpenXLA Authors" in notice
    assert "Copyright (c) Fabian Giesen, 2016" in notice
