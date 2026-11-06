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
