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
These host policies import neither framework. Ligand conformer optimization and
the complete public ligand preparation pipeline still require further extraction.
