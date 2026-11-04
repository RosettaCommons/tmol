"""Explicit device lanes for common scientific contracts."""

import os

import pytest


@pytest.fixture(
    params=os.environ.get(
        "TMOL_TEST_BACKENDS", "torch-cpu,torch-cuda,torch-mps,jax-cpu,jax-cuda"
    ).split(",")
)
def backend(request):
    from .execution import Execution

    return Execution(request.param)
