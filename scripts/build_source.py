"""Record immutable source identity in separately built release artifacts."""

import hashlib
import json
from pathlib import Path
import re
import subprocess


def source_identity(root, revision=None):
    root = Path(root).resolve()
    if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("source revision must be the verified full Git commit SHA")
    probe = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0 and Path(probe.stdout.strip()).resolve() == root:
        actual = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision is not None and revision != actual:
            raise ValueError("source revision disagrees with the checkout")
        dirty = bool(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(root),
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                ],
                text=True,
            ).strip()
        )
        # Wheels, virtual environments and downloaded release candidates live
        # outside source packages. New files inside build inputs still count.
        dirty |= bool(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(root),
                    "ls-files",
                    "--others",
                    "--exclude-standard",
                    "--",
                    "tmol",
                    "tmol_core",
                    "tmol_jax",
                    "scripts",
                    "packages",
                ],
                text=True,
            ).strip()
        )
        return dict(revision=actual, dirty=dirty)
    if revision is None:
        raise ValueError("Source archives require --source-revision with verified SHA")
    return dict(revision=revision, dirty=None)


def stamp_package(root, stage, package, revision=None):
    record = source_identity(root, revision)
    directory = stage / package
    record["sha256"] = {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "source.json"
    }
    (directory / "source.json").write_text(json.dumps(record, indent=2) + "\n")
    (stage / "MANIFEST.in").write_text(
        f"include VERSION\ninclude {package}/source.json\n"
    )
