"""Opt-in CI capture of actual operands for failed numerical contracts.

GitHub Actions retains bounded ZIP/base64 payloads in the existing JUnit artifact.
TMOL_SCIENCE_FAILURE_DIR also enables optional local files. Passing/skipped tests
produce nothing; diagnostic probes never alter the original test verdict.
"""

import base64
import contextlib
import hashlib
import io
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

import pytest

_TARGETS = {
    "test_sparse_nonbonded_values_and_coordinate_derivatives",
    "test_molecular_nonbonded_channels",
}
_ARRAYS = {
    "x",
    "coords",
    "values",
    "value",
    "scored",
    "gradient",
    "indices",
    "scored_indices",
    "weights",
    "whole",
    "aggregated",
    "direction",
    "numeric",
    "analytic",
}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment(root):
    versions = {}
    for name in ("numpy", "torch", "pytest", "biotite", "scipy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    native = sys.modules.get("tmol._C")
    native_path = Path(native.__file__) if native is not None else None
    torch = sys.modules.get("torch")
    import numpy as np

    numpy_config = io.StringIO()
    with contextlib.redirect_stdout(numpy_config):
        np.show_config()
    return {
        "numpy_config": numpy_config.getvalue(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "versions": versions,
        "source": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "native": (
            None
            if native_path is None
            else {"path": str(native_path), "sha256": _sha256(native_path)}
        ),
        "torch_config": None if torch is None else torch.__config__.show(),
        "torch_threads": None if torch is None else torch.get_num_threads(),
        "torch_default_dtype": (
            None if torch is None else str(torch.get_default_dtype())
        ),
        "thread_environment": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "CPU_PYTEST_WORKERS",
                "PYTEST_XDIST_WORKER",
                "TMOL_USE_JIT",
                "TMOL_TEST_BACKENDS",
                "TMOL_MPS_OFFLOAD",
            )
        },
    }


def _capture(item, call):
    import numpy as np

    frame = next(
        entry.frame.f_locals
        for entry in call.excinfo.traceback
        if entry.name in _TARGETS
    )
    arrays = {
        name: value
        for name, value in frame.items()
        if name in _ARRAYS
        and isinstance(value, np.ndarray)
        and not value.dtype.hasobject
    }
    backend = frame.get("backend")
    if backend is not None and hasattr(backend, "pose"):
        pose = backend.pose
        for name in (
            "block_type_ind",
            "block_coord_offset",
            "inter_residue_connections",
        ):
            arrays["pose_" + name] = backend.numpy(getattr(pose, name))
    probes = {}
    try:
        # A failed path-equality assertion precedes the independent derivative
        # check. Record its two steps without turning the failed test green.
        cpu_torch = (
            getattr(backend, "framework", None) == "torch"
            and getattr(backend, "requested_device", None) == "cpu"
        )
        execution = frame.get("execution") if cpu_torch else None
        if execution is not None and {"coords", "weights", "gradient"} <= arrays.keys():
            coords, weights = arrays["coords"], arrays["weights"]
            direction = np.random.default_rng(103).normal(size=coords.shape)
            direction /= np.linalg.norm(direction)
            arrays["probe_direction"] = direction
            probes["analytic_directional"] = float(
                np.sum(arrays["gradient"] * direction)
            )
            probes["finite_differences"] = []
            steps = (1e-3, 3e-4) if frame.get("family") == "hbond" else (1e-5, 2e-6)
            for step in steps:
                plus = backend.numpy(execution.scores(coords + step * direction)[0])
                minus = backend.numpy(execution.scores(coords - step * direction)[0])
                probes["finite_differences"].append(
                    {
                        "step": step,
                        "value": float(np.sum((plus - minus) * weights) / (2 * step)),
                    }
                )
        elif cpu_torch and "x" in arrays:
            value64, gradient64 = backend.channel_values_and_gradients(
                arrays["x"].astype(np.float64)
            )
            arrays["probe_value64"], arrays["probe_gradient64"] = value64, gradient64
            probes["force_sum_float64_execution"] = gradient64.sum(
                axis=2, dtype=np.float64
            ).tolist()
    except Exception as error:
        probes["error"] = repr(error)
    metadata = {
        "nodeid": item.nodeid,
        "diagnostic_probes": probes,
        "failure": str(call.excinfo.value),
        "case": {k: frame[k] for k in ("family", "case_name", "name") if k in frame},
        "arrays": {
            k: {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "sha256": hashlib.sha256(v.tobytes()).hexdigest(),
            }
            for k, v in arrays.items()
        },
    }
    if "gradient" in arrays:
        gradient = arrays["gradient"]
        axis = 2 if frame.get("name") is not None else 0
        metadata["gradient_sum_original_dtype"] = gradient.sum(axis=axis).tolist()
        metadata["gradient_sum_float64"] = gradient.sum(
            axis=axis, dtype=np.float64
        ).tolist()
    arrays_buffer = io.BytesIO()
    np.savez_compressed(arrays_buffer, **arrays)
    root = Path(item.config.rootpath)
    files = {
        "arrays.npz": arrays_buffer.getvalue(),
        "metadata.json": (json.dumps(metadata, indent=2) + "\n").encode(),
        "environment.json": (json.dumps(_environment(root), indent=2) + "\n").encode(),
    }
    # Discovered compiler records supplement, not replace, the native hash.
    for pattern in (
        "**/CMakeCache.txt",
        "**/compile_commands.json",
        "**/CMakeCXXCompiler.cmake",
    ):
        for path in (root / "build").glob(pattern):
            files[str(path.relative_to(root))] = path.read_bytes()
    return files


def _payload(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    data = buffer.getvalue()
    if len(data) > 2 * 1024 * 1024:
        raise ValueError(f"science diagnostic ZIP exceeds 2 MiB ({len(data)} bytes)")
    return base64.b64encode(data).decode("ascii")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    destination = os.environ.get("TMOL_SCIENCE_FAILURE_DIR")
    enabled = destination or os.environ.get("GITHUB_ACTIONS") == "true"
    if (
        not enabled
        or call.when != "call"
        or not report.failed
        or getattr(item, "originalname", None) not in _TARGETS
        or call.excinfo is None
    ):
        return
    try:
        files = _capture(item, call)
        payload = _payload(files)
        property = ("tmol_science_failure_zip_base64", payload)
        # JUnit finalization reads the teardown report, created after this hook.
        item.user_properties.append(property)
        if report.user_properties is not item.user_properties:
            report.user_properties.append(property)
        report.sections.append(
            ("science failure operands", "ZIP/base64 attached to JUnit properties")
        )
        if destination:
            prefix = (
                os.environ.get("PYTEST_XDIST_WORKER", "main")
                + "-"
                + hashlib.sha256(item.nodeid.encode()).hexdigest()[:16]
            )
            for name, content in files.items():
                target = Path(destination) / prefix / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
    except Exception as error:
        # Preserve the original failed assertion even when diagnostics fail.
        report.sections.append(("science capture error", repr(error)))
