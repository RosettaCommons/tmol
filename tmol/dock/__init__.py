"""Docking of a small molecule into a rigid or flexible-sidechain receptor.

Dock one ligand of a protein-ligand complex ``pose_stack`` (a single-pose
``PoseStack`` holding the receptor and the ligand in any placement)::

    from tmol.dock import GridBackend, GASchedule, run_docking

    # The GA runs on the pose's device; build the pose where you want to search.
    backend = GridBackend.from_pose_stack(pose_stack, "LG1", param_db)
    result = run_docking(pose_stack, backend, schedule=GASchedule())
    result.best_pose_stack, result.best_energies

``"LG1"`` is the ligand's residue name; a block index, or a boolean mask over the
pose's atoms selecting exactly the ligand's atoms, work too. To dock many ligands, build one complex and
one backend per ligand; to let pocket sidechains move, pass
``flexible_residues=select_pocket_residues(pose_stack, "LG1")``.

The docked ligand's coordinates come back in its block type's atom order; a
ligand prepared from a MOL2/SDF names its atoms by element plus the input file's
1-based atom index (``Cl8``).
"""

from .ga import (
    DockingResult,
    GASchedule,
    RandomPlacement,
    run_docking,
    select_pocket_residues,
)
from .ligand_aligner import LigandAligner
from .scoring_backend import ContinuousBackend, GridBackend

__all__ = [
    "ContinuousBackend",
    "DockingResult",
    "GASchedule",
    "GridBackend",
    "LigandAligner",
    "RandomPlacement",
    "run_docking",
    "select_pocket_residues",
]
