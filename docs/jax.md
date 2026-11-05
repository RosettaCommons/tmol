# JAX frontend (experimental)

`tmol-jax` provides Torch-free canonical-protein scoring, coordinate gradients
and Cartesian minimization, plus shared native generic-bonded, Cartesian-bonded,
backbone, Dunbrack, nucleic-acid torsion and metal coordination scoring for
explicitly parameterized noncanonical molecules. LJ/LK and electrostatic terms also accept
explicit ligand, D-residue, nucleic-acid and metal inputs; complete scoring for
those domains remains unsupported. It shares immutable parameter contexts, chemical records, topology,
parameter preparation and whole-pose native physics with the Torch frontend through [`tmol-core`](shared_core.md).
Existing `tmol` imports remain available.

## Install

From a TMol development environment with `build` installed:

```sh
python scripts/build_jax.py --outdir dist/jax
python scripts/build_jax_native.py --outdir dist/jax
pip install dist/jax/tmol_core-*.whl dist/jax/tmol_jax-*.whl dist/jax/tmol_jax_native-*.whl
```

The builder exports the canonical Torch-serialized scoring tables as NumPy
archives. Installed core/JAX wheels and standalone sdists require no Torch.
For source development, export tables with
`python scripts/build_jax.py --export-data DIR` and set `TMOL_JAX_DATABASE=DIR`.
The frontend requires Python 3.12+ and JAX 0.11.2+.

Build shared CPU/CUDA physics with
`python scripts/build_jax_native.py --cuda --cuda-architectures "80;90" --outdir dist/jax`.
The shared adapter supports float32/64; install a matching JAX CUDA runtime,
such as `jax[cuda13]`. An older optional native
adapter accelerates float32 hydrogen bonds and LK-ball
using the canonical TMol scalar potentials:

```sh
python scripts/build_jax_cuda.py --outdir dist/jax-cuda --cuda-architectures "80;90"
pip install dist/jax-cuda/tmol_jax_cuda-*.whl
```

Build it with the CUDA toolkit appropriate to the deployment. Its ABI marker is
checked before registration. CPU and other portable routes remain available;
dispatch follows the compilation platform, including CPU arrays in a GPU process.
The native adapter recomputes conservative pair bounds from current coordinates.

## Score and minimize

```python
import jax
from tmol_jax import beta2016_score_function, pose_stack_from_pdb, run_cart_min

pose = pose_stack_from_pdb("input.pdb")
sfxn = beta2016_score_function()
score = sfxn.render_whole_pose_scoring_module(pose)
energies = jax.jit(score)(pose.coords)
gradient = jax.jit(jax.grad(lambda x: score(x).sum()))(pose.coords)
minimized, result = run_cart_min(
    pose, sfxn, pose.atom_is_real, max_iter=100, return_result=True
)
```

The beta2016 preset retains every declared weight. NA torsion and metal
coordination use shared native kernels; complete noncanonical scoring remains
limited by other terms, which reject unvalidated chemistry before tracing.

Rendered scorers snapshot weights, term options and topology. Reuse them for
coordinate changes; render again after changing weights, parameters, types,
connectivity or batch layout. `sum_terms=False` returns weighted channels in
`ScoreType` order; additionally set `apply_weights=False` for unweighted values.
`set_options({"ref_weights": weights})` replaces reference weights; explicit D
entries take precedence over the corresponding L residue. Other score options
currently reject explicitly. Minimization accepts atom or xyz movement masks, freezes padding and returns a
new pose. Check `result["converged"]` and `result["status"]`.

Attach general geometric constraints with `tmol_jax.ConstraintSet.create_empty(
pose.device, pose.n_poses).add_constraints(callback, atom_indices, parameters)`
and `pose.replace(constraint_set=constraints)`, then enable `ScoreType.constraint`.
Indices identify `(pose, block, local_atom)`; `(block, local_atom)` rows repeat
across poses. `ConstraintEnergyTerm` provides shared harmonic distance/coordinate,
bounded distance and circular harmonic formulas. JAX callbacks can also supply
one energy per gathered four-atom row. Parameter arrays remain differentiable;
pose selection preserves their atom identities. Topology preparation is on the
host, and constraints spanning more than two block runs reject explicitly.
Harmonic/bounded arithmetic supports higher derivatives; the shared native
torsion primitive supports first coordinate derivatives. The earlier coordinate
restraint container in `tmol_jax.score.constraint` remains available.

`FoldForest.reasonable_fold_forest(pose)` shares chemical spanning and cycle
handling with Torch, preserving chain labels and non-kinematic closures.
For manually prepared poses without chain metadata, supply `chain_id=` with
shape `[poses, blocks]`. These residue-level edges do not yet construct an
atom-level kinematic forest.

`tmol_jax.kinematics.KinematicForest(parent, frame_x, frame_y, frame_z,
doftype, device=...)` prepares validated indices in kinematic coordinate order,
including virtual root zero. `jax.jit(forest.inverse)(coordinates)` returns
float32/64 DOFs with shape `[nodes, 9]`; `vmap` accepts coordinate samples for
the same forest. Initialization does not propagate coordinate derivatives.
Bind initialized DOFs with `coordinates = forest.bind(dofs)`, then evaluate
`jax.jit(coordinates)(coordinates.initial_dofs)`. This map accepts `[nodes, 6]`
movable DOFs: six columns for jumps and four for bonds. Binding snapshots the
three jump reference angles as fixed data. Native forward scans support first
reverse derivatives and `vmap`; forward-mode and higher derivatives are not
implemented. Automatic pose-to-forest preparation, MoveMap and molecular
internal-coordinate minimization remain absent.

`run_cart_relax` performs a no-pack repulsive-weight ramp with optional mainchain
coordinate constraints. It retains the best full-weight structure. It does not
implement FastRelax. Complete nucleic-acid, metal and ligand/noncanonical
scoring, repacking and kinematic minimization remain unsupported.

`PackerPalette(noncanonical_design=True)` exposes the shared chemical residue-choice
policy through `block_types_from_original(pose.packed_block_types, pose.block_type_ind)`
and `create_restrict_to_repacking_mask(...)`. These prepare choices; JAX rotamer
generation, packing tasks and packing execution remain unavailable.

For already parameterized chemistry, `pose_stack_from_residue_blocks()` accepts
a batch of `ResidueBlock(residue_type, atom_names, coords)` records and explicit
named connection ports. Full residue names include variants; every atom must be
present exactly once. The loader preserves atom identity and floating precision,
and does not infer bonds or change chemistry. `return_context=True` returns the
pose and a context whose `parameter_database` must be used for scoring. Molecular
input support does not imply complete scoring support: see the current
[capability matrix](capability-matrix.json).

Generic, Cartesian bonded, backbone, Dunbrack, nucleic-acid torsion, metal
coordination and combined LJ/LK/electrostatic scoring require the shared native package
built above. Set `ScoreType.gen_torsions` on a `ScoreFunction` constructed with the input's
parameter database. These terms use the same whole-pose native physics and
analytic derivatives as Torch; explicit noncanonical input does not enable
unvalidated terms. See `scripts/build_jax_native.py --help` for CUDA builds.

## Validation and numerical limits

Run `pytest tests/jax` for JAX transformations and runtime contracts, and
`TMOL_TEST_BACKENDS=jax-cpu pytest tests/scientific` for shared scientific cases.
The same cases accept `torch-cpu`, `torch-cuda`, `torch-mps` and `jax-cuda`; missing
devices are explicitly skipped. Core chemistry tests live in `tests/core`.
Native FFI calls support reverse gradients, `vmap` and `jacrev`; batching is
sequential initially. Shared native terms support first-order JVP and reverse gradients in float32/64;
their derivative kernels do not provide higher derivatives. Combined LJ/LK and
electrostatics use live native spatial neighbors and one gradient buffer for a
weighted total. Standalone LJ/LK and electrostatics use their shared native
kernels with the same first-order transformation contract and independent
parameter prerequisites.

Before comparing frontends, verify residue/atom identities and use identical
coordinates and parameters: loaders can idealize missing atoms and hydrogens
differently. Canonical residue and atom metadata come from the common chemical database.
Unknown input atom names reject instead of being dropped. PDB loading selects
the first model/alternate location; represent
chain breaks explicitly. Float32 reductions and torsion formulas differ between
execution routes, so energies and gradients need scientifically justified
term-specific tolerances. The separate L-BFGS implementation need not follow the
same trajectory or stopping point as Torch.

Record setup, compilation and steady execution separately; synchronize with
`block_until_ready()` for timing. Portable pair tables and the dense topology compatibility array still have
quadratic storage. The common sparse graph algorithm supplies their chemistry.
The fused native nonbonded layout currently supports at most 12 connection ports
per residue; larger port counts reject explicitly. Connection atom indices use
the full native integer range. The frontend does not change global JAX precision settings.
The initial implementation derives from
[tudorcotet/jtmol](https://github.com/tudorcotet/jtmol); see `tmol_jax/NOTICE`.
