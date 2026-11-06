"""Bind the three JAX release wheels and installed native libraries to one source."""

import argparse
import hashlib
from importlib.util import find_spec
import json
from pathlib import Path
import zipfile

PACKAGES = ("tmol_core", "tmol_jax", "tmol_jax_native")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def wheel_record(path, package, revision):
    with zipfile.ZipFile(path) as archive:
        source = archive.read(f"{package}/source.json")
        identity = json.loads(source)
        if identity["revision"] != revision or identity["dirty"] is True:
            raise ValueError(f"{path}: source identity does not match clean {revision}")
        builds = {}
        if package == "tmol_jax_native":
            for name in archive.namelist():
                if name.startswith(package + "/build-") and name.endswith(".json"):
                    platform = Path(name).stem.removeprefix("build-")
                    build = json.loads(archive.read(name))
                    libraries = [
                        p
                        for p in archive.namelist()
                        if p.startswith(f"{package}/_tmol_{platform}")
                        and p.endswith(".so")
                    ]
                    if (
                        len(libraries) != 1
                        or digest(archive.read(libraries[0])) != build["library_sha256"]
                    ):
                        raise ValueError(f"{path}: native binary digest mismatch")
                    builds[Path(libraries[0]).name] = build["library_sha256"]
            if not any(name.startswith("_tmol_cpu") for name in builds):
                raise ValueError("native wheel has no recorded CPU library")
    return dict(
        filename=path.name,
        sha256=digest(path.read_bytes()),
        source_sha256=digest(source),
        libraries=builds,
    )


def verify_artifacts(manifest, wheels):
    if set(wheels) != set(PACKAGES):
        raise ValueError("release requires core, JAX and native wheels")
    for package, wheel in wheels.items():
        if (
            wheel_record(wheel, package, manifest["revision"])
            != manifest["wheels"][package]
        ):
            raise ValueError(f"{package}: artifact differs from release manifest")


def verify_installed(manifest):
    for package in PACKAGES:
        root = Path(find_spec(package).origin).parent
        record = manifest["wheels"][package]
        source = (root / "source.json").read_bytes()
        if digest(source) != record["source_sha256"]:
            raise ValueError(f"{package}: installed source identity mismatch")
        identity = json.loads(source)
        files = (
            record["libraries"] if package == "tmol_jax_native" else identity["sha256"]
        )
        for relative, expected in files.items():
            if digest((root / relative).read_bytes()) != expected:
                raise ValueError(f"{package}/{relative}: installed content mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--core-wheel", required=True, type=Path)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--native-wheel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    wheels = dict(zip(PACKAGES, (args.core_wheel, args.wheel, args.native_wheel)))
    manifest = dict(
        revision=args.revision,
        wheels={
            package: wheel_record(path, package, args.revision)
            for package, path in wheels.items()
        },
    )
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
