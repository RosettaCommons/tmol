# Shared chemistry and execution boundaries

`tmol-core` owns the framework-independent `tmol_core` namespace: chemical
records, YAML and unit parsing, residue patching and validation, electrostatic/
reference/disulfide parameter records and loaders, and the vendored PySMILES
reader/writer. Torch and JAX consume this same implementation. Existing
imports under `tmol.database.chemical`, `tmol.database._patched_chemdb`, and
`tmol_jax.database.chemical` remain available. JAX's patched database additionally
refines the shared raw records into its host topology arrays.

The `tmol` distribution retains the public Torch API and native CPU/CUDA/Metal
bindings. Renaming it to `tmol-torch` would require installation and metadata
migration without improving this dependency boundary; `import tmol` is unchanged.
`tmol-jax` owns JAX arrays, tracing, differentiation and execution. Neither it nor
`tmol-core` requires Torch at runtime. Each namespace has one distribution owner.

`tmol_core.scoring.elec` resolves per-residue charges and count-pair
representatives on the host. Its lookup order is exact variant, terminal
conjugation, individual patches, then base residue. Missing charges raise by
default; a frontend preserving zero filling requests it explicitly. Returned
NumPy arrays belong to the caller; device materialization and cache lifetime
remain frontend responsibilities.

Native JAX CUDA adapters compile the existing canonical scalar potentials using
XLA FFI headers. Portable JAX potentials remain separate implementations needed
for CPU execution and JAX transformations; this extraction does not make all
physics shared. No new tensor abstraction or native shared-library ABI is added.

Build the core before installing a development frontend:

```bash
python -m pip install build
python scripts/build_core.py --outdir core-dist
python -m pip install core-dist/tmol_core-*.whl
# Then follow the Torch editable-build instructions, or build both pure packages:
python scripts/build_jax.py --outdir jax-dist
python -m pip install jax-dist/tmol_core-*.whl jax-dist/tmol_jax-*.whl
```

The JAX build exports legacy Torch-serialized scoring tables once in a developer
environment. Its wheel and standalone sdist contain NumPy tables and rebuild
without Torch. Core builds require no framework; they package the canonical
chemical YAML files from the repository. Installed core resources are independent
of the source checkout. Custom chemistry still loads through
`ChemicalDatabase.from_file(path)`. JAX's `TMOL_JAX_DATABASE` override is preserved.

Run `pytest -o addopts= tests/core` for shared chemistry contracts and the relevant
frontend chemistry tests for integration. Existing frontend/device suites remain
necessary for scoring, gradients, packing and minimization. Core code is BSD-3-Clause;
the shared PySMILES files retain their Apache-2.0 license and attribution.

Residue refinement also lives in `tmol_core.chemical.restypes`. Both frontends
use the same `RefinedResidueType`: atom and bond indices, bond orders and ring
membership, aliases, torsions, arbitrary connection paths, clipped graph
distances, and internal coordinates. The inputs are shared raw residue records;
the derived arrays are NumPy arrays. Existing frontend imports remain available,
and each frontend retains its own residue-set defaults and annotation lifetime.
This includes the full chemical database and custom residues; it does not imply
that every frontend implements every score term or modeling operation.

Generic-bonded preparation is provided by
`tmol_core.scoring.genbonded.GenBondedPreparation(chemical_database, genbonded_database)`.
`prepare_block`, `prepare_packed`, and `inter_tables` produce the same NumPy
subgraphs, parameters, type hierarchies, connection metadata, and lookup tables
used by Torch. Parameter precedence, ring membership, nucleic-acid/backbone
ownership, ligand-fragment source identities, and invalid-reference validation
remain in that common implementation. Frontends own annotation caches and device
transfer.

The native `genbonded_pose_kernels.hh` accepts typed input and output views;
callers supply zeroed energy/derivative buffers and an execution policy. The
whole-pose forward and weighted backward algorithms are shared with Torch's
allocating dispatch, including inter-block proper and improper torsions. This
header does not require Torch. Other terms still require further extraction;
sharing scalar potentials alone is not whole-core sharing.


Packed topology is prepared by `tmol_core.chemical.packed.PackedBlockTypesData`.
It accepts any ordered refined residue types and their chemical database, and
returns NumPy atom, bond, connection, and torsion metadata. Atom typing, ideal
coordinates in physical atom order, and path distances are available lazily.
Frontends transfer these arrays and choose storage capacity; a padded connection
slot does not change the true connection count. Torch reserves one slot for
native fixed-width views, including ligand-only palettes. The native scoring
backend's connection limit remains an explicit frontend constraint.

Chemical atom flags, ordered LJ/LK tables, and complete hydrogen-bond family
tables are prepared in `tmol_core.scoring.atom_types`, `ljlk`, and `hbond`.
Real atoms must resolve to registered types and finite active parameters. Missing
unused LJ/LK types retain explicit NaN rows; padding is not a valid real type.
Hydrogen-bond polynomial preparation retains the canonical float64 coefficients
and validates complete family-pair coverage before a backend chooses its numeric
representation. Torch keeps its public tensor containers and device caches.

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
