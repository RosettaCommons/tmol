# Shared chemistry and execution boundaries

`tmol-core` owns the framework-independent `tmol_core` namespace. Torch and JAX
use the same chemical records, parsing, patching, residue refinement, packed
topology, parameter lookup and scientific preparation. Shared preparation
accepts structural records and NumPy arrays; it never imports Torch or JAX.
It includes custom residues and databases, variants, arbitrary connection
metadata, ligand charge overrides, kinematic stencils and movement policies.
Frontends own device arrays, annotation caches and their lifetime.

The `tmol` distribution retains the existing public Torch imports and
CPU/CUDA/Metal bindings. `tmol-jax` owns JAX arrays, tracing and differentiation;
`tmol-jax-native` provides its CPU and optional CUDA execution bindings.
Neither JAX package requires Torch at runtime. Each Python namespace has one
distribution owner; moving Torch imports to another name would not improve
scientific sharing.

Common typed native headers own whole score-term traversal, scalar physics and
analytic coordinate derivatives, including bonded and nonbonded interactions,
hydrogen bonds, solvation, disulfides, nucleic-acid and metal terms. They also own
kinematic planning, coordinate transforms and Dunbrack sampling algorithms.
Torch registration and JAX XLA FFI supply execution policies, streams and buffers.
These kernels use structural views rather than a NumPy/JAX tensor compatibility
layer. `scripts/build_jax_native.py` stages the shared headers and their licenses;
the former separate `tmol-jax-cuda` prototype is retired.

Shared chemistry does not imply identical frontend capabilities. Native
connection and torsion capacities remain explicit. JAX first coordinate
JVPs/gradients use the common analytic derivatives; higher derivatives and
complete packing/design workflows require additional support. Undefined torsions
at coincident or collinear atoms do not acquire a zero-gradient convention.
See [JAX capabilities](jax.md) and the [capability matrix](capability-matrix.json)
for tested domains, supported operations and remaining boundaries. CPU, CUDA and
Metal still need independent execution validation.

Build the core before installing a development frontend:

```bash
python -m pip install build
python scripts/build_core.py --outdir core-dist
python -m pip install core-dist/tmol_core-*.whl
# For JAX, build the frontend and its native CPU execution package:
python scripts/build_jax.py --outdir jax-dist
python scripts/build_jax_native.py --outdir jax-dist
python -m pip install jax-dist/*.whl
# Add --cuda --cuda-architectures "80;90" to build native CUDA execution.
```

The frontend build exports legacy Torch-serialized scoring tables once in a
developer environment. Its wheel and standalone sdist contain NumPy tables and
rebuild without Torch. Core builds require no framework and package canonical
chemical YAML resources independently of the checkout. Custom chemistry still
loads through `ChemicalDatabase.from_file(path)`; JAX's `TMOL_JAX_DATABASE`
override is preserved.

Run `pytest -o addopts= tests/core` for framework-independent contracts and the
shared `tests/scientific` cases with the relevant execution adapters. Existing
frontend suites cover public APIs, device behavior, packing and minimization.
Core code is BSD-3-Clause; vendored PySMILES retains its Apache-2.0 license and
attribution, and native packages include their compiled dependencies' licenses.

Conjugated sampling shares library projection, joint chi enumeration, budgets,
current-state eligibility, and sampler ownership masks in `tmol_core.pack.conjugated`.
Repeated local tree/scan plans and torsion-child lookup also use common host
metadata. Frontends measure input torsions and execute the existing native
forward/inverse transforms. This preserves coupled group geometry; it does not
provide a ring-closure solver or establish support in a frontend that has not
integrated the complete workflow.

Native bump pruning shares its preliminary graph and energy bounds through
`BumpGraphContext` and three allocation-free stages in
`pack/compiled/interaction_graph_bounds.hh`. Adapters zero scratch and energy
buffers, scan block-pair counts to size chunk storage, then scan chunk sizes to
size pair-energy storage. Offsets and scan totals use int64; native dispatch
sizes are checked. Empty block bounds retain the mature sentinel `1234`.
The allocating Torch controller calls these same stages. The two size-discovery
barriers remain part of the execution contract; a frontend must implement them
before claiming support for this packing workflow.

Sparse backbone, disulfide, metal and constraint preparation accepts optional
per-block correlation groups. Empty metadata preserves independent sampling;
equal nonnegative IDs select matching local conformer indices. Callers must
validate equal active counts and correspondence with the shared correlation
plan. Native preparation preserves connection identity and separate metal bridge
contributions. A two-residue cycle has one sparse conformer pair whose energy
and coordinate derivatives include both directed peptide bonds. Dependencies on
a third independently sampled backbone block still require an explicit workflow
boundary; pair selection alone cannot represent that higher-body interaction.

Structure-input protonation and terminal-hydrogen policy live in
`tmol_core.io.protonation_input` and `tmol_core.io.input_termini`. They accept
Biotite atom arrays and explicit chemical databases, retaining AtomWorks hydrogen
assignment, custom residue identities, metal coordination and explicit termini.
The default patched chemistry is shared with the Torch parameter context; loading
an explicit database path still creates fresh chemistry. The bounded weak-owner
cache is framework neutral, while the Torch adapter retains CUDA stream readiness.
These host policies import neither framework.

The optional `tmol-core[ligand]` dependencies provide structure chemistry and
ligand preparation without installing either framework.

`tmol_core.ligand` owns ligand typing, protonation routing, charge overrides,
parameter-file loading/writing, residue registration, polymer/conjugate chemistry
and chemical fragmentation. Preparation accepts an explicit parameter context
and preserves its subtype. File entry points take an atom-array reader; structure
arrays and prepared parameter files need no reader. Torch supplies its public
default database and converts returned canonical ordering to its existing class.
Pose tensor assembly remains in the Torch adapter.

Distance-geometry preparation shares ideal targets, bounds, planar embedding,
random seeds and restart selection. Its `ConformerExecution` protocol supplies
planar refinement, chiral annealing and final stress refinement. The common refinement objectives use six array primitives. Torch retains its
existing dtypes and LBFGS execution; JAX supplies compiled differentiation and
its own L-BFGS execution. Supplying complete
MOL2 coordinates and authoritative charges needs no executor; regeneration
requires an executor explicitly at the core boundary. Both frontends supply
default executors. Shared molecular tests cover generated L/D ligands and aspirin
and supplied MOL2 through full scoring derivatives and minimization. This does
not establish equal optimizer trajectories or end-to-end runtime/memory parity.

FastRelax schedule normalization, score-weight ramps, stage lifetime hooks,
default task registration, and per-pose acceptance live in `tmol_core.relax`.
Frontends provide packing/minimization execution and array/metadata selection.
The public Torch defaults and callback signatures remain available; sharing the
protocol does not establish equivalence of the frontend optimizers.

For a connected third backbone residue in either endpoint's correlation group,
backbone scoring resolves its atom from the matching local conformer and that
conformer's actual residue type. The same global atom index receives its weighted
coordinate derivative. The optional final `lockstep_group_for_block` view selects
this behavior; an empty view retains fixed-background resolution. Adapters must
validate group counts, local-state correspondence and compatible connection
layouts. A third independently varying block remains outside this pairwise model.


`tmol_core.io.atom_array_from_file` and `file_from_atom_array` share the mature
structure readers and writers, including PDB/CIF model and assembly selection,
compressed/binary CIF, MOL2/SDF, alternate locations, CCD supplements, supplied
bonds, metal coordination, author identities and unresolved-atom placeholders.
The default polymer-port policy uses only shared chemical/canonical metadata.
Torch retains pose tensor conversion and its existing public file APIs.

The mature segmented Armijo optimizer now shares its native state transitions,
compact H0=I direction, circular history, restart/retirement and independent
convergence policy. Shared setup also owns settings validation, legacy checkpoint
migration, segment layout and history allocation rules. Torch retains parameter
storage, segmented reductions and
linear-algebra execution. The common policy imports neither framework; a JAX
executor supplies functional array storage and native transition calls through
`tmol_jax.optimization.armijo_minimize`. Complete independent-segment and
molecular objective checks cover the new execution path. The existing
strong-Wolfe API remains available; general Cartesian/kinematic defaults have not
yet switched to Armijo. Matched optimizer trajectories, performance and memory
parity remain separate acceptance gates.

Host PDB metadata and split-block identity maps live in `tmol_core.io`. Torch
retains its original imports; JAX keeps its legacy label aliases. Both consume
the same optional atom-annotation projection and packing metadata policy.
