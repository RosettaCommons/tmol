"""Same-version artifacts must not substitute for the recorded release."""

import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


def test_release_rejects_swapped_artifact_and_changed_native_binary(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/jax_release_manifest.py"
    spec = importlib.util.spec_from_file_location("release_manifest", script)
    release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release)
    revision = "a" * 40

    def wheel(package, source_revision=revision, binary=b"native", recorded=b"native"):
        path = tmp_path / f"{package}-0.2.0-py3-none-any.whl"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                f"{package}/source.json",
                json.dumps(dict(revision=source_revision, dirty=False, sha256={})),
            )
            if package == "tmol_jax_native":
                archive.writestr(f"{package}/_tmol_cpu.so", binary)
                archive.writestr(
                    f"{package}/build-cpu.json",
                    json.dumps(dict(library_sha256=release.digest(recorded))),
                )
        return path

    wheels = {package: wheel(package) for package in release.PACKAGES}
    manifest = dict(
        revision=revision,
        wheels={
            package: release.wheel_record(path, package, revision)
            for package, path in wheels.items()
        },
    )
    release.verify_artifacts(manifest, wheels)
    wheel("tmol_core", "b" * 40)
    with pytest.raises(ValueError, match="source identity"):
        release.verify_artifacts(manifest, wheels)
    wheel("tmol_core")
    wheel("tmol_jax_native", binary=b"different")
    with pytest.raises(ValueError, match="binary digest mismatch"):
        release.verify_artifacts(manifest, wheels)
