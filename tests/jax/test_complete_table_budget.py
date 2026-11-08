"""Small fused tables fit an explicit budget; forced pages still use replay."""

import os

import pytest

from tmol_core.pack.streaming import complete_table_fits_window
from tmol_jax.pack import streaming
from tests.scientific.execution import Execution
from tests.scientific.rotamer_execution import RotamerExecution
from tests.scientific.test_complete_preset import CASES
from tests.scientific.test_canonical_molecular_input import (
    canonical_input,
    rebuild_canonical,
)
from tests.scientific.test_packing_execution import check_complete_packing


def _backend():
    return Execution(
        "jax-cuda"
        if "jax-cuda" in os.environ.get("TMOL_TEST_BACKENDS", "")
        else "jax-cpu"
    )


@pytest.mark.parametrize("name", ("ligand", "mixed_batch", "d_peptide"))
def test_complete_buffer_bound_covers_compiled_scoring(name):
    backend = _backend()
    backend.molecule(CASES[name], terms=())
    # Public canonical input retains the complete chemistry palette, including
    # parameter tables for residues absent from this particular pose.
    co, canonical = canonical_input(CASES[name], backend.parameter_database)
    rebuild_canonical(backend, co, canonical)
    rotamers = RotamerExecution(backend, family="genbonded", sampled=True).rotamers
    backend.use_beta2016()
    scorer = backend.score_function.render_rotamer_scoring_module(
        backend.pose, rotamers
    )
    bound = streaming.complete_table_bound(scorer, rotamers.coords)
    assert bound is not None
    values, indices = scorer(rotamers.coords)
    assert values.size <= bound[1]
    # Independently measure the production call with device tables as operands.
    # An extra enclosing jit embeds those tables as constants instead, changing
    # the storage contract. Native scratch is an explicit FFI result.
    evaluate = scorer.prepare(rotamers.coords)
    executable = evaluate.func.lower(rotamers.coords, **evaluate.keywords).compile()
    memory = executable.memory_analysis()
    assert memory.output_size_in_bytes + memory.temp_size_in_bytes <= bound[2]
    assert values.nbytes + indices.nbytes <= bound[2]
    assert streaming.use_complete_table(scorer, rotamers.coords)
    assert not streaming.use_complete_table(scorer, rotamers.coords, score_window=1)
    assert not complete_table_fits_window(
        None, bound[1], bound[2], values.dtype.itemsize
    )


@pytest.mark.parametrize("force_pages", (False, True))
def test_public_packing_routes_and_preserves_shared_science(monkeypatch, force_pages):
    backend = _backend()
    # CPU can exercise the same native replay lifecycle for route validation;
    # production eligibility remains CUDA-only and is separately asserted.
    import tmol_core.pack.streaming as policy

    assert not policy.should_stream_interaction_graph("cpu", (), False)
    assert policy.should_stream_interaction_graph("cuda", (), False)
    monkeypatch.setattr(policy, "should_stream_interaction_graph", lambda *args: True)
    original_fit = streaming.use_complete_table
    monkeypatch.setattr(
        streaming,
        "use_complete_table",
        lambda scorer, coords: original_fit(
            scorer, coords, **({"score_window": 1} if force_pages else {})
        ),
    )
    calls = []
    driver = streaming.build_streaming_interaction_graph

    def observe(*args, **kwargs):
        calls.append(True)
        return driver(*args, **kwargs)

    monkeypatch.setattr(streaming, "build_streaming_interaction_graph", observe)
    check_complete_packing(backend, "mixed_batch", bump=False)
    assert len(calls) == int(force_pages)
