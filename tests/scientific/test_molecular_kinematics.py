"""Atom identities and per-term derivatives through real molecular kinematics."""

from contextlib import nullcontext

import numpy as np
import pytest

from .kinematics import MolecularKinematicExecution, MOLECULAR_KINEMATIC_TERMS
from .molecules import BACKBONE_CASES, DOMAIN_CASES
from .dunbrack_molecules import DUNBRACK_CASES
from .na_molecules import NA_CASES
from .metal_molecules import METAL_CASES

KINEMATIC_CASES = {
    name: (case, MOLECULAR_KINEMATIC_TERMS) for name, case in DOMAIN_CASES.items()
}
KINEMATIC_CASES.update(
    backbone_variants=(BACKBONE_CASES["variants"], ("omega", "rama")),
    dunbrack_d_peptide=(
        DUNBRACK_CASES["d_peptide"],
        ("dunbrack_rot", "dunbrack_rotdev", "dunbrack_semirot"),
    ),
    dunbrack_mixed=(
        DUNBRACK_CASES["mixed_batch"],
        ("dunbrack_rot", "dunbrack_rotdev", "dunbrack_semirot"),
    ),
    nucleic_modified=(NA_CASES["methylcytosine"], ("na_torsion", "na_torsion_well")),
    coordination_bridge=(METAL_CASES["bridging_donor"], ("metal_coordination",)),
)


@pytest.mark.parametrize("case_name", KINEMATIC_CASES)
def test_molecular_coordinate_mapping_and_term_gradients(backend, case_name):
    context = (
        backend.module.enable_x64() if backend.framework == "jax" else nullcontext()
    )
    with context:
        case, terms = KINEMATIC_CASES[case_name]
        backend.molecule(case, terms=terms)
        coordinates = backend.coords().astype(np.float64)
        # Padding deliberately contains a recognizable value and must survive.
        pbt = backend.pose.packed_block_types
        kinds = backend.numpy(backend.pose.block_type_ind)
        counts = backend.numpy(pbt.n_atoms)
        offsets = backend.numpy(backend.pose.block_coord_offset)
        physical = np.zeros(coordinates.shape[:2], bool)
        for p, b in zip(*np.nonzero(kinds >= 0)):
            begin = offsets[p, b]
            physical[p, begin : begin + counts[kinds[p, b]]] = True
        coordinates[~physical] = 7.125
        execution = MolecularKinematicExecution(backend, coordinates)
        ids = execution.atom_ids
        assert ids[0] == -1
        np.testing.assert_array_equal(np.sort(ids[1:]), np.flatnonzero(physical))
        initial = execution.initial_dofs
        rebuilt = execution.evaluate(initial)
        np.testing.assert_allclose(
            rebuilt[physical], coordinates[physical], rtol=2e-5, atol=2e-5
        )
        np.testing.assert_array_equal(rebuilt[~physical], coordinates[~physical])
        values, gradient = execution.score(initial, gradient=True)
        assert np.isfinite(values).all() and np.isfinite(gradient).all()
        assert np.any(np.abs(values) > 1e-3) and np.linalg.norm(gradient) > 1e-3
        direction = np.random.default_rng(177).normal(size=initial.shape)
        direction /= np.linalg.norm(direction)
        step = 1e-5
        numeric = (
            execution.score(initial + step * direction)
            - execution.score(initial - step * direction)
        ) / (2 * step)
        analytic = np.sum(gradient * direction, axis=(1, 2))
        np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-5)
