# Ligand docking

Search for a ligand's placement in a binding site with a genetic algorithm over
its rigid-body position, its torsions, and optionally the pocket sidechains.

Docking needs a single-pose `PoseStack` holding the receptor and the ligand, the
parameter database that prepared the ligand, and the ligand's identity: a residue
name, a block index, or a boolean atom mask. The input placement only defines the
search box; the population starts from random placements inside it.

```python
import torch
from tmol import ParameterDatabase, atom_array_from_file, pose_stack_from_atom_array
from tmol.dock import GASchedule, GridBackend, run_docking

complex_pose, context = pose_stack_from_atom_array(
    atom_array_from_file("complex.cif"),
    torch.device("cuda"),
    prepare_ligands=True,
    param_db=ParameterDatabase.get_default(),
    return_context=True,
)

backend = GridBackend.from_pose_stack(complex_pose, "BTN", context.parameter_database)
result = run_docking(complex_pose, backend, schedule=GASchedule(), finalize=True)
```

`result.best_pose_stack` is the top-ranked complex and `result.best_energies` its
score; `result.ranked_energies` scores every candidate, best first. Build the pose
on the device the search should run on; the genetic algorithm runs there.

## Scoring backends

`GridBackend` precomputes the rigid receptor's contribution on a grid around the
ligand, so each of the thousands of pose evaluations costs a cubic B-spline
interpolation per ligand atom instead of a receptor-wide pair calculation.
Hydrogen bonds and ligand-internal energies are evaluated live. LK-ball is
gridded too, as Rosetta's weighted approximation; `lk_ball_grid=False` scores it
live and exactly instead.

Grids depend on the receptor, the box, and the probe atom types, not on the
ligand. For a screen, build one set with `GridBackend.pocket_grids()` for the
union of the ligands' atom types and pass it to every `from_pose_stack()`.

`ContinuousBackend` scores the whole complex with beta2016 instead, so no grid
is built and no receptor interaction is approximated, at a **much higher cost
per evaluation**. It is the reference the grid is checked against, and the backend to
use when a receptor moves in ways a fixed grid cannot carry.

The two are interchangeable: both are built with `from_pose_stack()` and take
the same `flexible_residues`, and `run_docking()` drives either one with the
same arguments. Only grid-specific options differ --- `pocket_grids()`,
`lk_ball_grid`, and `initialization="aligner"`, which places the ligand in the
grid box.

```python
from tmol.dock import ContinuousBackend

backend = ContinuousBackend.from_pose_stack(
    complex_pose, "BTN", context.parameter_database
)
```

## Flexible sidechains

Pass `flexible_residues` to let pocket sidechains move. Their chis become part of
the genetic algorithm's search alongside the ligand's own degrees of freedom, and
the refinement stage repacks and minimizes them.

```python
from tmol.dock import select_pocket_residues

flexible = select_pocket_residues(complex_pose, "BTN")
backend = GridBackend.from_pose_stack(
    complex_pose, "BTN", context.parameter_database, flexible_residues=flexible
)
```

`select_pocket_residues()` returns the polymer residues whose heavy atoms come
within a cutoff (default 4.0 A) of a ligand heavy atom. It leaves out residues
whose sidechain cannot move independently: those with no chis, those whose
sidechain closes a ring onto the backbone, and those carrying a non-polymer
covalent connection. Any sequence of block indices works in its place.

## Controlling the search

`GASchedule` holds the search parameters. The defaults run `generations=10`
rounds over a population of `npool=800`, each round minimizing under a
repulsive-weight ramp:

```python
schedule = GASchedule(generations=20, npool=1600)
```

A larger pool and more generations find more; runtime grows with both. How much
is needed depends on the site: an enclosed pocket converges with a small
population, an open or shallow one does not. Check by docking the same complex
under several `seed` values and seeing whether they agree. The
`sidechain_sampling` option chooses how flexible chis are searched, whole
Dunbrack rotamers by default.

`run_docking()` takes the other search controls: `seed`, and `initialization`,
which selects the starting population. `"random"` places the ligand randomly in
the box, `"jitter"` perturbs the input placement, which makes the run a local
refinement, `"conformer"` seeds torsions from a conformer library, and
`"aligner"` places it with `ligand_aligner`.

## Refinement and ranking

Without `finalize`, the result holds the grid-ranked pool: fast, but ranked by
the search's approximate objective. With `finalize=True` the top `top_k`
candidates are rebuilt as full complexes, refined against the exact score
function, and ranked by exact complex energy.

The refinement protocol follows the backend's mode, `FLEX_REFINE` when it
declares flexible sidechains and `RIGID_REFINE` otherwise. Pass a
`RefineProtocol` to override it.

`result.finalization` carries every refined candidate: `pose_stack` holds them
all, ranked, alongside the interaction energy `dH`, the entropy estimate `TdS`,
and the `dG` estimate they are ranked by. Those fields read as `None` when
finalization did not run.

## Reading the result

The docked ligand's coordinates come back in its block type's atom order, which
is the order the ligand was prepared in. A MOL2 ligand keeps the atom names its
file carries; an SDF ligand, which has none, is named by element plus the file's
1-based atom index (`O2`, `H53`). Compare against an external reference by name
rather than position.

## Examples and reference

{ref}`Docking example </tutorial/01_tmol_tutorial.ipynb#8.-advanced-topics>` · {doc}`Docking API </api/dock>` · {doc}`Ligand preparation </user_guide/ligands>`
