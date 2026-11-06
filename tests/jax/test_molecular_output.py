"""Output atom identity follows chemical types even for multi-letter metals."""

from io import StringIO

import jax
import numpy as np
import pytest

from tests.scientific.execution import Execution
from tests.scientific.molecules import DOMAIN_CASES
from tmol_jax.io.write_pose_stack_pdb import (
    atom_records_from_pose_stack,
    write_pose_stack_pdb,
)


@pytest.mark.parametrize(
    "case_name,metal", (("metal_mixed", "Zn"), ("metal_cluster", "Fe"))
)
def test_pdb_elements_preserve_multi_letter_chemical_identity(case_name, metal):
    backend = Execution("jax-cpu").molecule(DOMAIN_CASES[case_name], terms=("ref",))
    with jax.enable_x64():
        coords = np.asarray(backend.pose.coords, dtype=np.float64).copy()
        coords[0, 0, 0] = 1.123456789123
        backend.pose = backend.pose.replace(
            coords=jax.device_put(coords, backend.device)
        )
        records = atom_records_from_pose_stack(backend.pose)
        assert records[0]["x"] == coords[0, 0, 0]
    metal_rows = [
        row for row in records if row["element"].strip().lower() == metal.lower()
    ]
    assert metal_rows
    stream = StringIO()
    write_pose_stack_pdb(backend.pose, stream)
    text = stream.getvalue()
    lines = [
        line for line in text.splitlines() if line.startswith(("ATOM  ", "HETATM"))
    ]
    assert len(lines) == len(records)
    for line, record in zip(lines, records):
        assert line[12:16].strip() == record["atomn"]
        assert line[76:78].strip().lower() == record["element"].strip().lower()
