---
name: tmol-pack-relax
description: >
  Repack or design side chains, minimize coordinates, and run TMol FastRelax on
  CPU or CUDA. Use for PackerTask masks and samplers, Cartesian or kinematic
  movement, constraints, large-batch packing memory, and CUDA graph selection.
  Do not use for score-only analysis.
allowed-tools: Bash, Read, Write, AskUserQuestion
license: BSD-3-Clause
---

# TMol packing and relax

Keep chemistry, task masks, movement controls, and score-function parameters
explicit. Packing can change block identities and atom layout; render a new
scorer for the returned pose.

This workflow requires a working TMol installation, a prepared `PoseStack`,
and a score function built from the same `ParameterDatabase`.

## Fixed-sequence repacking

```python
from tmol import PackerPalette, PackerTask, pack_rotamers
from tmol.pack import (
    FixedAAChiSampler,
    IncludeCurrentSampler,
    create_dunbrack_sampler_from_database,
)

task = PackerTask(pose, PackerPalette())
task.restrict_to_repacking()
task.add_conformer_sampler(
    create_dunbrack_sampler_from_database(database, pose.device)
)
task.add_conformer_sampler(FixedAAChiSampler())
task.add_conformer_sampler(IncludeCurrentSampler())
packed = pack_rotamers(pose, sfxn, task)
```

Use `disable_packing_by_block_mask()` to freeze selected blocks.
`restrict_to_repacking()` preserves the original identities; mutation or design
requires an explicit allowed-identity task rather than silently relaxing this
restriction. Protein, nucleic-acid, and ligand cases may need different
samplers; follow `docs/workflows/packing.md` and the chemistry-specific guide.

## Minimization

Use `run_cart_min()` for coordinate optimization and pass a boolean coordinate
mask when only selected atoms may move. Use `run_kin_min()` with a configured
`FoldForest` and `MoveMap` when torsions or rigid-body jumps are the actual
degrees of freedom. A `MoveMap` does not enable movement merely by existing;
set the intended flags or masks.

Constraints must be attached to the pose and have a nonzero score-function
weight. Preserve that pair of requirements when diagnosing a constraint that
appears inactive.

## FastRelax

```python
from tmol import KinematicRelaxMinimizer, fast_relax

relaxed = fast_relax(pose, sfxn)  # Cartesian, every atom free
# torsion space, every mainchain/sidechain torsion and jump free
torsion_relaxed = fast_relax(pose, sfxn, minimizer=KinematicRelaxMinimizer())
```

The `minimizer` carries the mode and its settings:
`CartesianRelaxMinimizer(cartesian_move_map)` or
`KinematicRelaxMinimizer(move_map, fold_forest)`; each rejects the other's move
map type. Without a move map the kinematic minimizer frees every torsion and
jump; a `MoveMap` built with `from_pose_stack` frees nothing until enabled. The
fold forest defaults to `FoldForest.reasonable_fold_forest`. On
CUDA both use graph replay; CPU runs eagerly. Pass `cuda_graph=False` to the
minimizer to force eager execution. Custom minimizers follow
`tmol.relax.RelaxMinimizer` and manage their own execution mode.

## Large batches and memory

TMol automatically chunks large pose stacks before native packing, using a
conservative topology and free-memory policy. The compact interaction-graph path
does not require CUDA graph capture. Let the default select a chunk size first.

For diagnosis or an application-specific memory ceiling, lower the bound with:

```bash
TMOL_PACK_MAX_POSES_PER_CHUNK=10 python workflow.py
```

Smaller chunks reduce peak memory but can reduce throughput. A larger override
does not bypass native signed-32-bit safety checks and may fail cleanly if the
work unit is unrepresentable. Record the pose count, largest residue count,
device, chunk override, runtime, and peak allocation when tuning.

## Validate

Confirm that output poses are finite, the expected blocks were allowed to
change, fixed blocks remained fixed, and constraints/movement masks were
honored. FastRelax is stochastic: seed reproducibly for regression tests and
compare outcomes as well as elapsed time.

Primary references: `docs/workflows/packing.md`,
`docs/user_guide/optimization.md`, `docs/user_guide/cpu_threading.md`, and
`docs/api/pack.rst`.
