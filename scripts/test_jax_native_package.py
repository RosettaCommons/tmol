"""The shared native package must carry every compiled dependency's license."""

from pathlib import Path
import tomllib

from build_jax_native import stage_native

ROOT = Path(__file__).resolve().parents[1]


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
