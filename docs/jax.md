# JAX frontend (experimental)

`tmol-jax` provides Torch-free molecular scoring, coordinate gradients and
Cartesian/internal-coordinate minimization. Chemistry, parameter preparation,
topology and native CPU/CUDA scoring physics are shared with Torch through
[`tmol-core`](shared_core.md). Explicitly parameterized ligands, noncanonical and
D residues, variants, nucleic acids, metals and mixed systems retain their input
identities and parameter context. Existing `tmol` imports remain available.

This is experimental: full packing/design execution is unfinished, and new
hydrogen-bond, LK-ball and disulfide adapters still require final CUDA, package
and Metal validation. See the [capability matrix](capability-matrix.json) for
implemented behavior and evidence rather than assuming full TMol parity.

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
such as `jax[cuda13]`. Build with the CUDA toolkit appropriate to the deployment.
Its ABI marker is checked before registration. Dispatch follows the compilation
platform, including CPU arrays in a GPU process. Native adapters recompute
spatial neighbors from current coordinates. The former `tmol-jax-cuda` prototype
is superseded by `tmol-jax-native`.

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

The beta2016 preset retains all 25 declared channels and uses shared native
physics for coordinate-dependent terms. Required missing parameters raise
during preparation. See the capability matrix for tested domains and open gaps.

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
host, and constraints spanning more than two distinct blocks reject explicitly.
Harmonic/bounded arithmetic supports higher derivatives; the shared native
torsion primitive supports first coordinate derivatives. The earlier coordinate
restraint container in `tmol_jax.score.constraint` remains available.

`FoldForest.reasonable_fold_forest(pose)` shares chemical spanning and cycle
handling with Torch, preserving chain labels and non-kinematic closures.
For manually prepared poses without chain metadata, supply `chain_id=` with
shape `[poses, blocks]`.

`PoseStackKinematics(pose, fold_forest=None)` uses the common chemical stencils
and native pose assembler without Torch. Preparation runs once on CPU;
coordinate execution uses the pose's device and preserves atom IDs and padding.

```python
from tmol_jax import PoseStackKinematics

kinematics = PoseStackKinematics(pose)
coordinates = kinematics.bind()
xyz = jax.jit(coordinates)(coordinates.initial_dofs)
```

The bound map accepts six movable DOF columns per forest node and returns the
original `[poses, atoms, 3]` layout. `kinematics.inverse(xyz)` initializes the
full nine-column layout. Ligands, noncanonical/D residues, nucleic acids, metals
and mixed batches use the same preparation. Empty poses reject explicitly.

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
implemented.

`MoveMap.from_pose_stack(pose)` prepares the shared hierarchical movement policy
on the host, with all DOFs initially disabled. Enable named torsions, individual
bonds or jumps using the same flags and setters as Torch. For example:

```python
from tmol_jax import MoveMap, run_kin_min

movement = MoveMap.from_pose_stack(pose)
movement.move_all_named_torsions = True
minimized, diagnostics = run_kin_min(
    pose, score_function, movement, max_iter=50, return_result=True
)
```

The native resolver maps these choices to chemical DOFs. Internal and Cartesian
minimization share the JAX L-BFGS implementation; the input pose stays unchanged.
Fixed DOFs remain exact, while atoms downstream of movable bonds or jumps move.
Internal-minimization diagnostics report DOFs and their masked gradients.
Float32 internal derivatives can lose accuracy through cancellation; float64 is
recommended when comparing small rotational gradients of strongly repulsive systems.

`run_cart_relax` performs a no-pack repulsive-weight ramp with optional mainchain
coordinate constraints. It retains the best full-weight structure. It does not
implement FastRelax or repacking. The full 25-channel preset is tested on
ligand, noncanonical, D-residue, nucleic-acid, metal and mixed-system fixtures;
this does not establish exhaustive coverage of every custom chemistry.

`PackerPalette(noncanonical_design=True)` exposes the shared chemical residue-choice
policy through `block_types_from_original(pose.packed_block_types, pose.block_type_ind)`
and `create_restrict_to_repacking_mask(...)`. These prepare choices. `tmol_jax.pack.prepare_rotamer_kinematics(packed_block_types)`
shares residue-local trees, ideal DOFs and scan planning with Torch;
`prepare_block_group_kinematics(block_types, links, anchor=0)` prepares connected
chemical groups. Initialization calls the common native CPU inverse directly,
without compiling each residue shape. Samplers own immutable parameter snapshots;
create a new sampler after changing the database. Reusing a sampler retains its
native table buffers while reading fresh coordinates and task requests.
`DunbrackChiSampler.from_database(db,
expanded_limit=1000, limit=1000).prepare_packed_block_types(pbt)` prepares shared
library/explicit-chi metadata on the palette's device. `prepare_atom_copy_plan(pbt,
samplers)` preserves sampler-owned retained atom identities.

Public chi sampling uses the same chemical requests and native CPU/CUDA sampler
as Torch. Register samplers explicitly; the JAX task currently has no default
fallback sampler for uncovered residues:

```python
from tmol_jax.pack import PackerTask, PackerPalette, DunbrackChiSampler

task = PackerTask(pose, PackerPalette())
task.restrict_to_repacking()
sampler = DunbrackChiSampler.from_database(context.parameter_database)
task.add_conformer_sampler(sampler)
counts, choices, defining_atoms, chi = sampler.sample_chi_for_poses(pose, task)
```

Counts and choices refer to concrete task rows; defining atoms use residue-local
indices. Ligands outside the sampler's amino-acid policy receive zero samples.
Task masks and `set_chi_sample_budget(expanded_limit, limit)` control sampling;
budget-frozen torsions retain measured input geometry. Dynamic output sizes
require an eager host allocation barrier, so sampling is not differentiable,
jittable or vmappable. Coordinate assembly is available separately below; full
packing, FastRelax and design execution remain unfinished.

`NaChiRotamerSampler.from_database(db, chi_sample_level=1, sample_syn=True)`
samples DNA/RNA glycosidic and proton chi from the same shared chemical metadata,
scoring tables and native kernels as Torch, including modified nucleotides.
Register it with the task like the Dunbrack sampler. Levels 0–4 expand the
glycosidic distribution; task masks and sample budgets still apply. Public NA
angles retain Torch's float32 table precision. Coordinate assembly preserves the
pose dtype. `plan = sampler.prepare_chi_for_poses(pose, task)` fixes the discrete
sugar-pucker and sample choices eagerly; `plan.chi_values(proton, means, sdev)`
accepts optional float32 table overrides and supports `jit`, `vmap`, JVPs and
higher parameter derivatives. It does not differentiate the discrete preparation.

`OptHSampler(flip_NHQ=True)` samples proton chi in supported protein,
noncanonical, ligand and nucleic-acid chemistry, and optionally ASN/GLN/HIS
flips. Its common native assembly copies input DOFs, resets NHQ downstream
atoms, and applies sampled torsions. Enable OptH and Dunbrack on different
blocks; assigning both to one block raises an error. `sampler.prepare(pose, task)`
fixes discrete samples eagerly. The plan's `chi_values(proton_samples, flip_values)`
supports optional float32 parameter overrides, `jit`, `vmap`, JVPs and higher
parameter derivatives. `measure_flip_values(coords)` supports first coordinate
derivatives; higher coordinate derivatives through native torsions are absent.
CPU molecular sampling and built-coordinate checks pass; combined CUDA validation
is pending.

`IncludeCurrentSampler` and `FallbackSampler` expose
`create_samples_for_poses(pose, concrete_task)`, returning counts, choice indices
and an input-coordinate copy marker. Register either sampler explicitly.
Fallback accepts `built_rotamer_counts=counts` aggregated from earlier samplers;
it retains fixed residues and fills uncovered positions. Coverage includes all
identities at a position, so a sampled design choice does not reintroduce the
original identity. `build_rotamers(pose, task, parameter_database.chemical)` assembles their coordinates
and sampled Dunbrack, explicit amino-acid, NA and OptH chi using common native kernels. It returns
`(pose, rotamer_set)` with atom, residue and pose identities. Register
`FallbackSampler()` explicitly to preserve fixed and uncovered residues. Input
conformers are copied exactly; sampled coordinates preserve retained atoms and
requested adjustable torsions. Custom chemical databases remain attached to the
pose's packed types.

Assembly is eager because sampling determines output sizes; it does not support
`jit`, `vmap` or coordinate differentiation through sampling/initialization. The
native DOF operations support `jit` and `vmap`. Row copying and chi assignment
also support JVPs, gradients and higher derivatives using common native index
plans. Mainchain copying also preserves ideal DOFs where source values are
non-finite and supports these derivatives with fresh native selection indices.
Nonlinear measured-chi correction still lacks JAX derivative rules. Current-only prepared execution preserves input coordinate
derivatives, including fixed residues and padded batches. A concrete `SetPackerTask` reuses
validated topology plans while recomputing coordinates. Internal native stages
now provide sparse graph topology/retention, device load balancing with XLA-owned
scratch, sphere-based candidate enumeration, and the mature CPU annealing schedule.
Sparse LJ/LK, electrostatic, LK-ball, hydrogen-bond, Dunbrack, Cartesian,
generic-bonded, backbone, nucleic-acid, disulfide and metal values and coordinate VJPs use common native
physics with `jit`/`vmap` support;
forward-mode and higher derivatives are not implemented for these sparse entries.
Backbone and connection candidate scoring currently reject correlated groups;
backbone also rejects two-residue cycles. These common sparse dispatch paths
need additional support.
Sparse constraints use the same native candidate expansion and preserve physical
atom order, including A/B/A/B torsions across two blocks. Their JAX callbacks
support coordinate and parameter derivatives, JVPs and `jit`/`vmap`; parameter
Hessians remain available with native torsions. Coordinate derivatives beyond
first order depend on the callback. Constraint lockstep expansion remains absent.
`score_function.render_rotamer_scoring_module(pose, rotamer_set)` composes every
enabled term, including reference energies and constraints. Calling it eagerly
returns `(values, indices)` with uncoalesced scalar entries and `[3, entries]`
pose/rotamer identities. `sum_terms=False` returns active named channels in
`ScoreType` order; `apply_weights=False` exposes their unweighted values. Weights
and options are snapshotted at render time, as for whole-pose scoring.

For coordinate gradients or compiled repeated evaluation, first call
`fixed = scorer.prepare(rotamer_set.coords)` outside tracing, then use
`jax.jit(fixed)` or differentiate its values. This fixes the discrete neighbor
list; refresh it after moving conformers. Calling the eager scorer itself under
JAX transformations rejects explicitly. Native interaction-graph construction
consumes these entries and shares background, one-body and chunked two-body
accumulation with Torch. Bump pruning and correlated-group collapse remain absent
from this JAX graph frontend.

`tmol_jax.pack.pack_rotamers(pose, score_function, task, seed=123)` runs the
complete sparse score, common interaction graph, native annealer and native
pose imposition. It requires float32 coordinates and explicit task samplers,
including fallback for uncovered residues. Ligand, noncanonical, D-peptide,
mixed and DNA repacking fixtures check selected coordinates, immutable chemistry
and the resulting whole-pose energy. CPU and CUDA retain their different mature
search schedules. CUDA phase seeds come from a local NumPy PCG64 generator;
packing leaves global random state unchanged. Bump pruning, correlated groups,
conjugated sampling and FastRelax remain unfinished.

`prepare_pose_imposition(pose, rotamers, graph["retained"], assignments)` prepares
discrete selection eagerly. Its `coordinates(rotamer_coords)` supports `jit`,
`vmap`, coordinate gradients, JVPs and higher derivatives; `apply` returns a new
pose. Identity changes compact atom storage and rebuild bond separation. The
frontend rejects changed connection-port layouts and identity changes with
attached constraints. Design selection covers L/D ARG-to-LYS and fixed-amino-acid
alternatives. Register `FixedAAChiSampler()` to generate Gly, Ala, D variants and
custom amino acids whose only chi are proton chi. It uses the common ideal
sidechain ownership policy, including both Gly alpha hydrogens. Dunbrack alone
does not generate these alternatives. Complete task/design support remains unfinished.

Float64 JAX assembly preserves dtype; this does not establish
float64 support for Torch's complete packing pipeline. Float32 reconstructed NA
backbones can differ slightly across conformers and from exact input copies;
sparse NA neighbor references use the first conformer. Exact whole-pose
conservation therefore requires identical fixed backbone and sugar coordinates;
raw float32 ensembles carry reconstruction error. Independent native Torch/JAX
sampled values and coordinate gradients agree on those raw ensembles.

For already parameterized chemistry, `pose_stack_from_residue_blocks()` accepts
a batch of `ResidueBlock(residue_type, atom_names, coords)` records and explicit
named connection ports. Full residue names include variants; every atom must be
present exactly once. The loader preserves atom identity and floating precision,
and does not infer bonds or change chemistry. `return_context=True` returns the
pose and a context whose `parameter_database` must be used for scoring. Molecular
input support does not imply complete scoring support: see the current
[capability matrix](capability-matrix.json).

All molecular interaction terms use the shared native package built above.
Use a `ScoreFunction` constructed with the input's parameter database; missing
required parameters raise an error. Hydrogen-bond polynomial coefficients retain
the native float64 representation even when JAX x64 mode is disabled. Disulfide
connections retain each residue's L/D stereochemistry. Reference and constraint
terms use their common host records and formulas.

## Validation and numerical limits

Run `pytest tests/jax` for JAX transformations and runtime contracts, and
`TMOL_TEST_BACKENDS=jax-cpu pytest tests/scientific` for shared scientific cases.
The same cases accept `torch-cpu`, `torch-cuda`, `torch-mps` and `jax-cuda`; missing
devices are explicitly skipped. Sparse float64 scientific checks require CPU or
CUDA; they do not validate Metal execution. Core chemistry tests live in `tests/core`.
Whole-pose native FFI calls support reverse gradients, `vmap` and `jacrev`; batching is
sequential initially. Shared whole-pose native terms support first-order JVP and reverse gradients in float32/64;
their derivative kernels do not provide higher derivatives. Coincident or
collinear atoms do not define a torsion; its native derivative can be non-finite. Combined LJ/LK and
electrostatics use live native spatial neighbors and one gradient buffer for a
weighted total. LK-ball likewise composes weighted derivatives in one native
reverse pass. Standalone LJ/LK and electrostatics use their shared native
kernels with the same first-order transformation contract and independent
parameter prerequisites.

Before comparing frontends, verify residue/atom identities and use identical
coordinates, parameters and weight precision: loaders can idealize missing atoms
and hydrogens differently. Match exposed score channels by name; Torch groups
channels by term while JAX orders them by `ScoreType`. Canonical residue and atom metadata come from the common chemical database.
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
