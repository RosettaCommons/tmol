"""Explicit device lanes for common scientific contracts."""

import os

import pytest


def pytest_configure(config):
    if os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get(
        "TMOL_SCIENCE_FAILURE_DIR"
    ):
        from . import failure_artifacts

        if not config.pluginmanager.is_registered(failure_artifacts):
            config.pluginmanager.register(failure_artifacts)


@pytest.fixture(
    params=os.environ.get(
        "TMOL_TEST_BACKENDS", "torch-cpu,torch-cuda,torch-mps,jax-cpu,jax-cuda"
    ).split(",")
)
def backend(request):
    from .execution import Execution

    return Execution(request.param)
