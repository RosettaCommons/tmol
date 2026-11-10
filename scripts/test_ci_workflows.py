"""Keep hosted PR checks independent of maintainer-requested GPU execution."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def workflow(name):
    # BaseLoader preserves GitHub's `on` key instead of interpreting YAML 1.1
    # booleans; all values here are workflow configuration strings.
    return yaml.load(
        (ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader
    )


def test_cpu_ci_runs_on_hosted_runners_without_requesting_gpu():
    ci = workflow("ci.yml")
    assert {"push", "pull_request", "workflow_dispatch"} <= ci["on"].keys()
    assert ci["on"]["workflow_dispatch"]["inputs"]["gpu"]["type"] == "boolean"
    jobs = ci["jobs"]
    assert jobs["gpu"]["if"] == (
        "github.event_name == 'workflow_dispatch' && inputs.gpu"
    )
    assert all(
        not row["runs-on"].startswith("self-hosted")
        for row in jobs["cpu"]["strategy"]["matrix"]["include"]
    )
    assert "gpu" not in jobs["cpu"].get("needs", [])
    # The aggregate must accept a deliberately skipped GPU lane, while failing
    # a requested run that did not succeed.
    check = jobs["ci"]["steps"][0]["run"]
    assert "needs.gpu.result" in check and "= skipped" in check
    assert "inputs.gpu" in check and "= success" in check


def test_pr_wheel_checks_do_not_request_gpu_but_releases_require_it():
    smoke = workflow("wheel-smoke.yml")
    assert smoke["jobs"]["test-gpu"]["if"] == (
        "github.event_name == 'workflow_dispatch'"
    )
    release = workflow("publish.yml")
    assert "test_gpu_runtime" in release["jobs"]["upload"]["needs"]
    assert "needs.test_gpu_runtime.result == 'success'" in (
        release["jobs"]["upload"]["if"]
    )
