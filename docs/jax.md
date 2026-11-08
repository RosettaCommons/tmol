# JAX frontend (experimental)

`tmol-jax` provides Torch-free molecular scoring, coordinate gradients and
Cartesian/internal-coordinate minimization. Chemistry, parameter preparation,
topology and native CPU/CUDA scoring physics are shared with Torch through
[`tmol-core`](shared_core.md). Explicitly parameterized ligands, noncanonical and
D residues, variants, nucleic acids, metals and mixed systems retain their input
identities and parameter context. Existing `tmol` imports remain available.

This is experimental: complete public API/design compatibility, optimizer
compatibility and end-to-end runtime/memory parity remain unfinished.
Native molecular scoring and selected packing paths have CPU/CUDA and Torch-free
package validation; final Metal validation is tracked separately. See the [capability matrix](capability-matrix.json) for
implemented behavior and evidence rather than assuming full TMol parity.

The standalone periodic spline utilities retain a separate JAX implementation.
Molecular table preparation and scoring use shared native splines; complete
source consolidation remains unfinished.

`tmol_jax.ligand` exposes the common ligand preparation and parameter-file APIs.
SMILES generation uses the shared distance-geometry bounds, chirality/planarity
objectives, stage schedules and restart policy with JAX differentiation and
optimization. MOL2
`mode="keep"` retains supplied geometry and charges. The returned database keeps
the caller's subtype and must be used for subsequent pose construction and scoring.
Optional ligand preparation dependencies are available as `tmol-jax[ligand]`
(or `tmol-core[ligand]` when installing the component packages separately).
Use the [pinned development constraint](shared_core.md) for the tested Torch-free
structure-preparation dependency. Published AtomWorks 3.0.4 imports Torch during
preparation when it is installed; the ordinary PyPI extra does not apply this fix.
CIF preparation and molecular file input use the common framework-free readers.
Preparation runs eagerly on the host; its compiled JAX numerical refinement
uses the CPU backend so seeded chemical parameters are independent of the
scoring device. Prepared poses execute scoring and minimization on CPU or CUDA.
Preparation is not a `jit`/`vmap` frontend. Generated conformers need not follow
Torch's optimizer trajectory, and matched cold/warm preparation and peak-memory
parity are pending.

## Install

From a TMol development environment with `build` installed:

```sh
python scripts/build_jax.py --outdir dist/jax
python scripts/build_jax_native.py --outdir dist/jax
pip install dist/jax/tmol_core-*.whl dist/jax/tmol_jax-*.whl dist/jax/tmol_jax_native-*.whl
```

Source-archive builds require `--source-revision` with the verified full commit
SHA. Release validation binds core, JAX and native wheel hashes to that revision
with `scripts/jax_release_manifest.py`; installed checks also verify package
contents and native library hashes. Same-version artifacts from another revision
are rejected.

The builder exports the canonical Torch-serialized scoring tables as NumPy
archives. Installed core/JAX wheels and standalone sdists require no Torch.
For source development, export tables with
`python scripts/build_jax.py --export-data DIR` and set `TMOL_JAX_DATABASE=DIR`.
The frontend requires Python 3.12+ and JAX 0.11.2+.

Build shared CPU/CUDA physics with
`python scripts/build_jax_native.py --cuda --cuda-architectures "80;90" --outdir dist/jax`.
The shared adapter supports float32/64; install a matching JAX CUDA runtime,
such as `jax[cuda13]`. Build with the CUDA toolkit appropriate to the deployment.
`tmol_jax.has_cuda()` reports native CUDA build support independently of
available JAX devices. Its ABI marker is checked before registration. Dispatch follows the compilation
platform, including CPU arrays in a GPU process. Native adapters recompute
spatial neighbors from current coordinates. The former `tmol-jax-cuda` prototype
is superseded by `tmol-jax-native`.

CUDA LJ/LK and electrostatic execution use the mature Torch compilation policy
in a separate translation unit. Bonded, torsion and kinematic execution retain
standard CUDA math. Native build manifests record every compile and link command.

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

Whole-pose hydrogen-bond, Dunbrack and backbone-torsion CUDA energy reductions
use wider accumulators before returning the caller dtype. Both current frontends
pass the retained float32 branched N-glycan repeat-energy and gradient checks;
the mature baseline still fails that repeat gate. Generated ligand geometry,
historical trajectories, complete runtime and peak-memory parity remain open.

Standalone LJ/LK float32 CUDA gradients now use the shared native wider
accumulator, then convert once to the coordinate dtype. The original lifecycle
repeat test and independent per-term molecular gradients pass with this path;
fixed-coordinate controls retain the original failed implementation's evidence.
These checks do not establish complete workflow or peak-memory parity.

Score weights default to float32 storage, matching mature Torch defaults even
when coordinates use float64. `ScoreFunction(db, weight_dtype=np.float64)` selects
explicit double-precision weights; clones retain that choice.

Immutable chemical parameter arrays are cached by live database/residue context and
execution device. Pose connectivity and coordinate-dependent work remain fresh.

Rendered scorers snapshot weights, term options and topology. Reuse them for
coordinate changes; render again after changing weights, parameters, types,
connectivity or batch layout. `sum_terms=False` returns weighted channels in
`all_score_types()` order, including zero-weight channels of retained terms;
additionally set `apply_weights=False` for unweighted values.
These two options can be set when rendering or overridden on each scorer call;
`scorer.unweighted_scores(coords)` returns all retained unweighted channels.
Per-call options are static under JAX transformations and reuse prepared operands.
`all_terms()`, body-count groups and `weights_tensor()` use the common lifecycle.
Setting a term's final nonzero weight to zero removes it; direct removal updates
coverage immediately while existing rendered scorers retain their snapshots.
`set_options({"ref_weights": weights})` replaces reference weights; explicit D
entries take precedence over the corresponding L residue.
`set_options({"soft_rep": True})` selects the shared soft-repulsive LJ parameters
for whole-pose, block-pair and packing scoring. Unknown options reject explicitly.
`ScoreFunction.from_sfxn_file(path, param_db)` reads versioned weights and options
through the same format parser as Torch; incompatible major versions reject. Minimization accepts atom or xyz movement masks, freezes padding and returns a
new pose. Check `result["converged"]` and `result["status"]`.

Attach general geometric constraints with `tmol_jax.ConstraintSet.create_empty(
pose.device, pose.n_poses).add_constraints(callback, atom_indices, parameters)`
and `pose.replace(constraint_set=constraints)`, then enable `ScoreType.constraint`.
Indices identify `(pose, block, local_atom)`; `(block, local_atom)` rows repeat
across poses. `ConstraintEnergyTerm` provides shared harmonic distance/coordinate,
bounded distance and circular harmonic formulas. JAX callbacks can also supply
one energy per gathered four-atom row. Parameter arrays remain differentiable;
pose selection preserves their atom identities. `clone()` and `to(device)`
retain parameter differentiation; `replicate_constraints()` and
`count_unique_blocks()` use the shared host topology policy. Topology preparation is on the
host. Whole-pose callbacks may span up to four blocks, using one physical atom
from each block. Sparse packing supports at most two distinct blocks and rejects
larger constraints explicitly.
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

`run_kin_min(..., kin_dtype=jax.numpy.float64)` and
`PoseStackKinematics.bind(kin_dtype=...)` select internal-coordinate precision
without changing stored coordinate dtype. Enable JAX x64 for float64 DOFs.
Mixed-precision objectives are promoted to DOF precision for native Armijo;
objectives exceeding that precision reject instead of being truncated.

`optimizer="armijo"` selects the common per-pose optimizer in `run_cart_min`
and `run_kin_min`; `optimizer_kwargs` uses mature TMol setting names.
`tmol_jax.optimization.armijo_minimize` runs the same policy on arrays and returns a
`checkpoint` containing coordinates, movement/segment identities, settings and
history. Resume with `armijo_minimize(objective, checkpoint["x"],
checkpoint=checkpoint)`. NumPy copies of checkpoint arrays can be restored to the
input device. Keep the same objective and parameter context; incompatible
coordinates, masks, segment identities, dtype or history capacity reject.
Checkpoint continuation is distinct from differentiating optimizer trajectories.

Both public minimizers also accept `optimizer=callable`. The function receives
`(objective, initial_variables, *, segment_ids, **settings)`, using compact movable
variables and one objective value per active pose. `optimizer_kwargs` supplies
settings; compiled `value_only` and `value_and_grad` callbacks are passed when
available. The derivative callback returns `(pose_values, gradient_of_their_sum)`.
Return a mapping with `x`, `grad`, scalar `initial_fun` and scalar `fun`; extra
diagnostics are retained. Coordinates and gradients must keep the input JAX
shape, dtype and device and remain finite. Frozen variables are restored by the
existing adapter. This functional contract replaces Torch's optimizer-class
injection; Torch `Optimizer`, `nn.Module` and `state_dict` interfaces stay
framework-specific. Default Torch/JAX minimization and FastRelax can reach
different endpoints. Float32 CUDA Cartesian minimization of ubiquitin also
failed repeated-endpoint checks; the cause remains unresolved. End-to-end
runtime and peak-memory parity are not established.

`CartesianMoveMap(coord_mask=mask)` carries an atom movement mask for Cartesian
relax callbacks. `run_cart_min(..., coord_mask=mask)` accepts the mature keyword;
`move_mask` remains available. Supply only one non-None mask.

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
`DunbrackChiSampler.from_database` snapshots the source immediately and fits
tables at the first supported sampling request or explicit `sampler.tables` access.
Requests with no chemically eligible choices return empty device arrays without
fitting tables, including ligand-only and nucleic-acid tasks with a full palette.
`DunbrackChiSampler.from_database(db,
expanded_limit=1000, limit=1000).prepare_packed_block_types(pbt)` prepares shared
library/explicit-chi metadata on the palette's device. `prepare_atom_copy_plan(pbt,
samplers)` preserves sampler-owned retained atom identities.

Dunbrack scoring prepares native tables only for libraries used by the input and
sampled design types. The common preparation validates the complete source lookup
and preserves custom mappings; the full palette and parameter context stay intact.
Inactive objectives retain their zero channels without uploading scoring tables.
For a coordinate-independent objective, JAX can prune unused inputs. Specify
`jax.jit(..., out_shardings=coords.sharding)` to retain their device when overriding
JAX's default device.

Public chi sampling uses the same chemical requests and native CPU/CUDA sampler
as Torch. Tasks inherit their palette's default fallback sampler. Register
additional sampling policies explicitly:

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
`sampler.defines_rotamers_for_bts(packed_block_types, indices)` queries residue
support before task restrictions, returning a boolean array on the palette's
device with the index shape. This preparation query uses the same chemical and
parameter policy as sampling and runs eagerly.
Task masks and `set_chi_sample_budget(expanded_limit, limit)` control sampling.
`restrict_absent_name3s(names, block_mask=mask)` intersects choices at selected positions and
rejects an empty identity set without partially changing the task; it never
re-enables a previously excluded choice. Sampler bundles use `expand_into_task`.
Budget-frozen torsions retain measured input geometry. Dynamic output sizes
require an eager host allocation barrier, so sampling is not differentiable,
jittable or vmappable. Coordinate assembly, native repacking and the shared FastRelax protocol are
available below; complete design/API compatibility remains unfinished.

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
Shared CPU/CUDA molecular sampling, built-coordinate and sampled-angle gradient
checks pass. This does not provide gradients through discrete sample selection.

`IncludeCurrentSampler` and `FallbackSampler` expose
`create_samples_for_poses(pose, concrete_task)`, returning counts, choice indices
and an input-coordinate copy marker. Register `IncludeCurrentSampler` explicitly;
the default palette supplies `FallbackSampler`.
Fallback accepts `built_rotamer_counts=counts` aggregated from earlier samplers;
it retains fixed residues and fills uncovered positions. Coverage includes all
identities at a position, so a sampled design choice does not reintroduce the
original identity. `build_rotamers(pose, task, parameter_database.chemical)` assembles their coordinates
and sampled Dunbrack, explicit amino-acid, NA and OptH chi using common native kernels. It returns
`(pose, rotamer_set)` with atom, residue and pose identities. The default
fallback preserves fixed and uncovered residues. Input
conformers are copied exactly; sampled coordinates preserve retained atoms and
requested adjustable torsions. Custom chemical databases remain attached to the
pose's packed types.

`rotamer_set.to_atom_array(packed_block_types)` exports a visualization AtomArray
through the shared host projection. Each rotamer is a numbered `model`, with
named atoms, packed coordinates and one-based original block residue numbers.
Element labels follow the mature atom-name policy.

Assembly is eager because sampling determines output sizes; it does not support
`jit`, `vmap` or coordinate differentiation through sampling/initialization. The
native DOF operations support `jit` and `vmap`. Row copying and chi assignment
also support JVPs, gradients and higher derivatives using common native index
plans. Mainchain copying also preserves ideal DOFs where source values are
non-finite and supports these derivatives with fresh native selection indices.
Measured-chi correction supports reverse gradients and reverse Hessian products
through the common native geometry; forward-mode and third derivatives remain
unavailable. Fixed-plan assembly supports first derivatives of sampled angles.
Input-coordinate derivatives through inverse kinematics and higher derivatives
through forward kinematics remain unavailable. Current-only prepared execution
preserves input coordinate derivatives, including fixed residues and padded
batches. A concrete `SetPackerTask` reuses
validated topology plans while recomputing coordinates. Internal native stages
now provide sparse graph topology/retention, device load balancing with XLA-owned
scratch, sphere-based candidate enumeration, and the mature CPU annealing schedule.
Sparse LJ/LK, electrostatic, LK-ball, hydrogen-bond, Dunbrack, Cartesian,
generic-bonded, backbone, nucleic-acid, disulfide and metal values and coordinate VJPs use common native
physics with `jit`/`vmap` support;
forward-mode and higher derivatives are not implemented for these sparse entries.
Backbone candidate scoring supports declared lockstep groups, including a third
backbone member. Independently varying third-member context remains unsupported. Two-residue cycles preserve both directed
bonds. Disulfide, metal and constraint candidates use native joint-state filtering.
Sparse constraints use the same native candidate expansion and preserve physical
atom order, including A/B/A/B torsions across two blocks. Their JAX callbacks
support coordinate and parameter derivatives, JVPs and `jit`/`vmap`; parameter
Hessians remain available with native torsions. Coordinate derivatives beyond
first order depend on the callback.
Sparse constraints spanning three or four blocks are unsupported in both
frontends: mature and current Torch omit those contributions during conformer
expansion, while JAX rejects them. Whole-pose multi-block callbacks are supported.
`score_function.render_rotamer_scoring_module(pose, rotamer_set)` composes every
enabled term, including reference energies and constraints. Calling it eagerly
returns `(values, indices)` with uncoalesced scalar entries and `[3, entries]`
pose/rotamer identities. `sum_terms=False` returns retained named channels in
`all_score_types()` order; `apply_weights=False` exposes their unweighted values. Weights
and options are snapshotted at render time, as for whole-pose scoring.

For coordinate gradients or compiled repeated evaluation, first call
`fixed = scorer.prepare(rotamer_set.coords)` outside tracing, then use
`jax.jit(fixed)` or differentiate its values. This fixes the discrete neighbor
list; refresh it after moving conformers. Calling the eager scorer itself under
JAX transformations rejects explicitly. Native interaction-graph construction
consumes these entries and shares background, one-body and chunked two-body
accumulation with Torch. Producer-declared correlated groups use the common
collapse plan: incompatible joint states are removed and compatible member
energies accumulate into one choice. `task.or_bump_check()` enables the shared
native lower-bound pruning rule. Retained indices preserve original choices,
including fixed survivors and holes left by removed conformers.

`tmol_jax.pack.pack_rotamers(pose, score_function, task, seed=123)` runs the
complete sparse score, common interaction graph, native annealer and native
pose imposition. It requires float32 coordinates; tasks inherit palette defaults
and accept additional samplers. Ligand, noncanonical, D-peptide,
mixed and DNA repacking fixtures check selected coordinates, immutable chemistry
and the resulting whole-pose energy. CPU and CUDA retain their different mature
search schedules. Set `anneal_device="cpu"`, `"cuda:0"`, or a JAX device to
run the search separately from scoring; the result stays on the pose device.
`verbose=True` reports synchronized total packing time.
JAX annealing supports native CPU and CUDA. Explicit seeds use the mature
MT19937 CPU and shared Philox CUDA seed streams while leaving global random state
unchanged. Integers in `[-2**63, 2**64)` are accepted; negative seeds use their
unsigned 64-bit representation, as in Torch. This changes seeded choices from
earlier JAX releases; matching seeds
alone does not establish complete numerical parity. Independent CUDA tasks without bump
pruning replay bounded score pages through the shared three-pass graph driver.
Small tables use complete fused execution when a conservative bound on all
explicit score/candidate/scratch buffers fits one common page; unknown bounds
retain replay. This bounds explicit buffers, not compiler or allocator overhead.
CPU, correlated tasks and bump pruning retain complete score tables. Native
graph tile widths default to 16 on CPU and 32 on CUDA; `chunk_size` overrides
the width. Automatic pose batching uses the common memory policy and preserves
task choices, constraints, fragment metadata and shared palette identity.
`TMOL_PACK_MAX_POSES_PER_CHUNK` overrides the pose limit; it is separate from
`chunk_size`. Pose chunks advance the same local seed stream, so changing
chunk boundaries can change discrete packing choices. Large-workflow runtime
and peak-memory parity remain separately gated.
For already parameterized covalent
groups, `ConjugatedChiSampler` and `add_conjugated_group_sampler` share task masks,
chi budgets and correlation declarations with Torch; native joint kinematics
preserve attachment geometry. Biotin and O/N-glycan prepared groups have full
score/gradient and packing checks; arbitrary custom groups remain unvalidated.
Structure-file ligand preparation uses the shared input pipeline. FastRelax shares the mature
schedule, task callbacks and per-pose acceptance; its JAX optimizer retains the
minimization limitations described above.

`prepare_pose_imposition(pose, rotamers, graph["retained"], assignments)` prepares
discrete selection eagerly. Its `coordinates(rotamer_coords)` supports `jit`,
`vmap`, coordinate gradients, JVPs and higher derivatives; `apply` returns a new
pose. Correlated graph choices expand back to every original member before
identity and atom-layout reconstruction. Identity changes compact atom storage
and rebuild bond separation. The
frontend preserves constraints when their atom names and indices are unchanged.
Changed connection ports or constrained atom indices reject explicitly. Design selection covers L/D ARG-to-LYS and fixed-amino-acid
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

`pose_stack_from_canonical_form()` accepts a supplied chemical ordering and
palette for ligands, noncanonical/D polymers, nucleic acids and mixed batches.
It uses common native candidate, terminus, histidine, protonation, cycle and
connection policies, plus shared disulfide, metal and covalent-attachment preparation. Observed
coordinates retain their dtype and atom identities; missing leaf atoms use the
common native frames. File and atom37 completion leave sampler-owned missing
atoms and dependent leaves for native packing to build; observed atoms remain
unchanged. Explicit bonds refer to canonical residue/atom indices,
including when the input contains padding. Interior padding breaks chains by
default; `compact_padding=True` removes virtual slots while retaining adjacency,
matching the Torch canonical-input contract. Input preparation runs on the host;
the individual native decision stages support `jit` and `vmap`.

The default PDB/Biotite palette includes all ordinary residues and variants in
the chemical database, including D residues and nucleic acids. Custom ligands
use the supplied chemical ordering and palette. Metal geometry and donor inference
use the common chemical policy and grow donor variants in that palette. PDB
CONECT and Biotite bonds use the shared atom-identity mapper for covalent bonds,
disulfides and metal coordination; polymer gap detection uses chemical ports.
Explicit
`metal_sites` and `metal_coordination` rows retain declared geometry and connectivity;
missing virtual sites use shared placement plans. `protonation_variants` supplies
explicit states (`-1` means infer), while observed hydrogen evidence and required
donors select chemical forms through the common native kernel. Disulfide states
remain authoritative. These decisions use the supplied database, including custom
L/D histidine and cysteine atom orderings. Public PDB/CIF/MOL2/SDF loading uses
shared structure preparation. `ligand_hydrogens="input"` (the JAX default) retains generated named
hydrogens only when their observed heavy parent and bond distance are consistent
with the common policy. `"names"` and `"rebuild"` reconstruct generated hydrogens.
Common native geometry reorients eligible rebuilt hydrogens from heavy neighbors,
including connected residues. Its fixed-topology primitive supports `jit`, `vmap`
and first/second reverse derivatives in coordinates and ideal bond lengths;
forward-mode and third derivatives are not implemented. Comparisons to Torch
must request matching input policies. Missing non-leaf atoms raise unless `build_missing_context=True` enables
the shared context planner. Supplied protonation evidence takes precedence over
provisional metal requests, matching Torch. Set `require_all_metal_donors=True`
to reject requests that cannot all be satisfied.
`res_not_connected` retains the JAX terminated-break convention; `res_gap`
suppresses bonds without forcing terminal variants.

For already parameterized chemistry, `pose_stack_from_residue_blocks()` accepts
a batch of `ResidueBlock(residue_type, atom_names, coords)` records and explicit
named connection ports. Full residue names include variants; by default every atom must be
present exactly once. Set `build_missing_leaf_atoms=True` to reconstruct omitted
leaf atoms from common chemical frames and native geometry. Observed atoms remain
exact; missing non-leaf atoms and unavailable frames raise. This option does not
generate ligands or infer bonds. `build_missing_context=True` additionally builds
anchored missing heavy atoms using common geometry and clash-selected rotations,
then completes leaf atoms. Observed atoms stay exact; unanchored components raise.
Context choices run on the host and receive the actual executed coordinates
between build waves. The fixed placement plan supports JAX `jit`, `vmap` and
first coordinate derivatives; this does not make discrete input planning differentiable.
The loader preserves atom identity and floating precision,
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

Run `TMOL_TEST_BACKENDS=jax-cpu pytest tests/scientific` for shared scientific
cases, including applicable JAX transformations and runtime contracts.
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
and hydrogens differently. Both frontends group score channels by retained term
using the common lifecycle, with native channel order within each term. Canonical
residue and atom metadata come from the common chemical database.
Unknown input atom names reject instead of being dropped. PDB loading selects
the first model/alternate location; represent
chain breaks explicitly. Float32 reductions and torsion formulas differ between
execution routes, so energies and gradients need scientifically justified
term-specific tolerances. Both minimizers use common optimizer policies with framework-specific array
execution. Floating reductions can still change trajectories and stopping points.

Record setup, compilation and steady execution separately; synchronize with
`block_until_ready()` for timing. Portable pair tables and the dense topology compatibility array still have
quadratic storage. The common sparse graph algorithm supplies their chemistry.
The fused native nonbonded layout currently supports at most 12 connection ports
per residue; larger port counts reject explicitly. Connection atom indices use
the full native integer range. The frontend does not change global JAX precision settings.
The initial implementation derives from
[tudorcotet/jtmol](https://github.com/tudorcotet/jtmol); see `tmol_jax/NOTICE`.

Public `get_sequence`, ligand/polymer masks and residue/atom label queries use
common identity policy, including author insertion codes and named constraint
atoms. Unknown one-letter identities remain `X`. `mutate_residue` and
`mutate_to_sequence` use shared task restrictions and native packing; pass the
custom parameter score function and an explicit seed for reproducibility. Other
blocks and poses stay frozen. One-letter mutation retains its mature amino-acid
contract; use exact residue identities for other supported types.

`PackedBlockTypes.from_restype_list(chem_db, restype_set, active_block_types,
device)` uses common topology preparation while retaining the supplied chemistry,
residue collection and ordered active types. `inds_for_restypes` returns `-1`
for absent names; `.cpu()` transfers palette arrays and retains that context.
The connection, bond and named-torsion metadata arrays support `jit` and `vmap`.
The public `count_*`, `determine_*` and `join_*` preparation methods use the same
host topology source as Torch and transfer their results to the requested device.
Preparation is eager; array execution and transport retain the supplied context.

`extended_pose_stack_from_sequences(seqs, device=..., param_db=...,
return_context=True)` consumes the common sequence grammar and ideal geometry:
uppercase protein, lowercase DNA, exact `X[NAME]` variants, `X(SMILES)` ligands,
and `:` chain breaks. Use the returned context's parameter database for scoring.
Context reuse retains prepared ligand identities; `termini=False` preserves
uncapped types. Canonical construction retains the supplied residue palette,
including unused tautomers and legal design identities; compacting coordinate
padding does not remove chemistry choices. `PoseStackBuilder.from_block_type_names(palette, names,
chain_lengths)` creates zero coordinates with the same named-connection parser.
A `--port-label` bond requires paired labels; unmatched labels have the mature
parser's ignored-label behavior. Invalid ports and multiply used labels raise.

`pose.chain_id` holds logical chain indices separately from author chain labels.
Sequence/canonical input, splitting, concatenation and fragment reconstruction
carry those indices. Poses constructed without explicit logical IDs infer them
eagerly from the common physical polymer-connection policy.

`tmol_jax.pose.select_blocks(pose, blocks)` preserves ordered atom identities,
coordinate gradients, metadata and the original bond distances between retained
blocks. It requires a homogeneous selected layout, drops constraints and rejects
split-block metadata, matching the mature API. `repeat_pose_stack(pose, n_poses)`
repeats one pose and its constraints with independent metadata copies. These eager
topology operations support coordinate differentiation; render a new scorer
after changing the layout.

`set_named_torsions(pose, poses, blocks, names, values, degrees=True)` uses
shared selection/validation and native inverse/forward kinematics. It preserves
coordinate dtype and parameter context and returns a new pose. Scalar or batched
requests follow the selected fold forest; duplicate requests add their deltas.
Missing-neighbor and non-bonded forest degrees of freedom raise. This eager
initialization is outside autodiff; bound kinematic maps provide differentiable
DOF execution. Mixed protein/ligand, D and beta peptides, DNA and RNA target and
bond-geometry checks pass on both CPU frontends and the matching installed/CUDA gates.

Named geometry queries are available as `get_torsion_names(pose, pose_index,
block_index)` and `get_named_torsions(pose, poses=None, blocks=None, names=None,
degrees=True)`. They share selection and result conventions with Torch, including
paired selections and NaN for absent neighboring atoms. These eager queries
return host values; coordinate differentiation uses scoring primitives.

FastRelax uses the shared mature schedule, packing/minimization stages, release
hooks, and per-pose acceptance policy. Import `fast_relax` from `tmol_jax` and
pass an explicit `seed`. Keyword-only `minimizer=CartesianRelaxMinimizer(...)`
or `KinematicRelaxMinimizer(...)` owns the movement map and optional fold forest;
the default frees every real atom in Cartesian space. Custom minimizers receive
`verbose` and optional per-stage `rtol`. Kinematic relaxation uses the shared
per-radian convergence policy; Cartesian relaxation retains L-BFGS defaults.
`packer_palette` is optional. Explicit CUDA graph capture is not implemented
by JAX (`cuda_graph=True` raises); ordinary native CUDA execution is supported.
Existing packing
capability limits apply. The public minimizers and these callbacks default to
shared per-pose L-BFGS/Armijo, including mature convergence, iteration and history
settings. Packing stages and pose chunks advance one local seed stream per device.
Explicit `optimizer="strong_wolfe"` minimizes the summed objective with
the common strong-Wolfe search and L-BFGS recurrence. JAX supplies fixed-capacity
device history storage and compiled array operations; movement masks and public
relative-loss stopping settings remain supported. The shared search can choose
different trials from earlier JAX releases. End-to-end runtime and peak-memory
parity remain unresolved.

`tmol_jax.build_context_from_atom_array()` prepares reusable chemistry and
parameters. `tmol_jax.pose_stack_from_atom_array(..., return_context=True)`
accepts prepared ligands, noncanonical/D residues and metal coordination using
the complete common canonical host plan for normalization, atom identity,
metadata, connectivity and terminal chemistry. Supplied neutral termini close
polymer gaps even when residue labels are consecutive and atoms are close. Use
the returned context for scoring. Missing sidechains and default OptH use the mature shared sampler
routing and native packing; `no_optH=True` disables OptH and `packer_seed` makes
packing reproducible. Fragment imposition uses common original-atom plans.
`pose_stack_from_file()` reads PDB, CIF, MOL2 and SDF through the common reader;
optional residue bounds select a half-open range. `atom_array_from_file`,
`file_from_atom_array`, and `file_from_pose_stack` expose the common PDB/CIF
read/write contract; writers return text when no path is supplied. Shared AtomArray export preserves
molecular metadata. Shared CPU workflows check
physical atom identity, neutral-terminal hydrogen names, missing HYP/D
sidechains, file input, and all 25 score-channel derivatives. This does not
establish complete input or end-to-end speed/memory parity.


`prepare_atom37_pose_builder(atoms, context)` consumes the shared canonical input
plan, including supplied terminal chemistry and protonation variants from the
context's chemical database. Supplied terminal hydrogens retain their source
coordinates and derivatives. Call `bind = builder.prepare(example_coords)` eagerly,
then use `bind(coords)` inside `jit`, `vmap` or reverse differentiation. Inputs
are float32 `[poses, tokens, 37, 3]`; mapped all-NaN triplets mean missing atoms.
Preparation fixes chemistry, nonleaf presence and inferred metal-site geometry.
Missing leaves use the common native reconstruction kernels. Prepare again when
those discrete choices change. Invalid mapped triplets raise during preparation;
invalid triplets or changed nonleaf presence during traced execution produce
NaN coordinates. `prepare(..., opt_h=True, seed=...)` uses the shared input
sampler policy and native packing. Sampled coordinates are values of a fixed
discrete choice; supplied atoms and rebuilt leaves in frozen blocks retain
coordinate derivatives. Missing sampled sidechains follow the same boundary.
Prepare again to reselect packing choices after coordinate motion.
For repeated source-gradient scoring, render a scorer from `bind(coords)` and
call `scorer.value_and_grad(coordinate_map=bind)`. This returns per-pose energies
and gradients with respect to the Atom37 input. Reused programs receive fresh
coordinate and parameter operands; each callable retains its prepared snapshot.
`coordinate_dtype=jnp.float64` evaluates mapped coordinates in double precision
while retaining float32 source gradients; JAX x64 must be enabled. Ordinary
`jit`, `vmap` and differentiation of `bind` remain available.
`pose_stack_from_atom37_and_topology()` and
`pose_stack_from_atom_array(atom37_coords=...)` provide eager entry points.
`pose_stack_from_atom37(coords, res_types, chain_id, context, slot_map=...)`
accepts tensor-only topology. Its standard layout covers canonical proteins;
custom layouts route arbitrary chemistry through the same shared validation.
For transformed execution, use `prepare_atom37_tensor_builder(..., n_slots=...)`
and its `prepare(example_coords)` method. Explicit disulfides, cyclic/covalent
bonds and suppressed polymer connections are retained. Unknown residue layouts
raise instead of replacing measured coordinates with ideal geometry.

The canonical compatibility functions `pose_stack_from_canonical_aa_atom37`
and `atomworks_from_pose_stack` use the shared standard protein encoding and
canonical chemistry subset. Use `prepare_canonical_aa_atom37_builder` before
transformations. This restricted 1..20 token format is separate from the full
molecular context API above; nonprotein tokens reject explicitly. Export keeps
coordinate derivatives, returns float32 slots with zeros for absent atoms, and
uses JAX int32 identity/chain arrays. The canonical constructor and prepared
builder accept `return_chain_ind`, `return_atom_mapping`, and
`return_block_has_missing_atoms`; requested metadata accompanies the pose in a
dictionary. Atom mappings refer to left-justified canonical residue indices,
including when the source contains padding.

`PoseStackBuilder.from_poses(stacks, device)` combines compatible chemical
contexts using the common palette, metadata and fragment-identity plans. It
preserves coordinate derivatives, constraints and sparse bond distances, pads
the batch, and retains the coordinate dtype. As in Torch, independently rebuilt
chemical databases must be resolved to one shared context before concatenation.

Pose layout helpers (`n_ats_per_block`, `real_atoms`, `n_res_per_pose`,
`block_identity_map`, and `expand_coords`) expose the shared molecular layout.
`expand_coords()` preserves coordinate derivatives and zeros padding. `split(i)`
retains that pose's input metadata and constraints. JAX arrays are immutable, so
`clone()` and `clone_sharing_topology()` share arrays; use `replace` for updates.

## Fragmented ligands

`pose_stack_from_atom_array` accepts the common ligand fragment annotation and
retains named cut bonds, atom parameters and each original molecule's identity.
Repeated copies of the same ligand remain separate groups. Shared reconstruction
plans preserve coordinate dtype, coordinate derivatives, per-atom observations
and constraint atoms. CPU checks cover real protein–ligand and repeated-ligand
systems, all 25 term values and physical gradients against the uncut molecule,
default hydrogen packing and ligand minimization. Installed CPU and actual CUDA
checks cover these paths. Fragment attribution uses the common original-column
policy and existing native pair kernels.

`atom_array_from_pose_stack` uses the common host exporter. It preserves declared
chemical and coordination bonds, per-atom observations, source residue annotations
and original fragmented-ligand/metal identities. `include_virtual_atoms=True`
retains site virtuals; `merge_fragments=False` keeps fragment labels. Multiple
models may differ in coordinates, but must share topology and metadata. Export
is eager host I/O and detaches coordinates from differentiation. Real zinc,
covalent modified residues, isopeptides, glycans and repeated ligands have CPU
roundtrip checks including all-term gradients, with installed CPU and CUDA gates.


`score_function.render_block_pair_scoring_module(pose)` retains directed block
entries, with the same weighted/unweighted and summed/per-term options as Torch.
Summing both block axes recovers the whole-pose physical energies and gradients.
Reference sums use the common float64 accumulator and cast once to the stored parameter dtype. Dense output
has quadratic block storage; all physical pair identities remain candidates so
coordinate motion cannot invalidate a captured neighbor list. Rendering is eager;
execution supports `jit`, `vmap` and first reverse coordinate derivatives.
The rendered scorer also provides `unweighted_scores(coords)` and
`score_interactions(coords, block_pair_indices, sum_terms=True, apply_weights=True)`.
Pair indices have shape `[pairs, 2]` and apply to every pose; repeated pairs count
repeatedly and an empty selection returns zero. Invalid indices raise in eager
execution and yield NaN under JAX transformations.
For a fixed selection, `selected = scorer.prepare_interactions(block_pair_indices)`
provides `selected.jitted()` and `selected.value_and_grad()` for repeated coordinate
evaluation. A new selection or parameter context requires a new prepared scorer.

`calculate_fragment_interactions(pose, partner_mask, sfxn=score_function)` returns
per-fragment partner scores and original column records. The Boolean partner mask
must exclude every ligand fragment. For transformed or repeated execution, prepare
`render_fragment_interactions(...)` once and call the returned scorer on coordinates;
its `mapping` describes the columns. These APIs retain both block orientations and
exclude fragment-fragment attribution. Real protein-ligand and repeated-ligand CPU
checks compare all channels and physical gradients to the uncut molecules.
Installed CPU and CUDA gates passed; complete matched mature runtime and
peak-memory parity remain unresolved.

Fragment attribution prepares only the requested fragment/partner pair identities
and reduces their native contributions directly, without allocating a full
block-by-block matrix. Coordinate-dependent physical cutoffs remain native.
The general block-pair API still returns its requested dense matrix.

Notebook `view`, `switchable_view`, `overlay_view` and `selection_gallery` use
the common renderer, with lazy JAX pose export. They accept Biotite arrays and
PDB text or paths as well as poses. Install the optional py3Dmol/IPython display
dependencies to render notebook HTML; viewing does not change molecular data.


An explicit native CPU ligand-preparation executor is available from
`tmol_jax.ligand._native_refinement.native_conformer_execution()`. Pass it as
`conformer_execution` to ligand preparation. It uses the common typed objective,
analytic gradients and host optimizer; prepared poses still score on their
selected JAX CPU/CUDA device. It requires matching native artifacts and remains
opt-in: seeded geometry and parameter identity against the historical mature
workflow, and complete cold/warm runtime and memory parity, remain open.


The record-based `tmol_jax.io.write_pose_stack_pdb` writer restores fragmented
ligands to their original residue names, author numbers, chains and insertion
codes through the common fragment policy. Pass `merge_fragments=False` to keep
separate fragment records. Its existing `pose_index` option selects one model;
`pose_index=None` writes all models. PDB coordinates retain three decimal places.
