# Shared chemistry and execution boundaries

TMol shares chemistry, parameter preparation, topology and native scientific
algorithms through `tmol_core` and framework-independent C++ headers. The public
Torch package remains `tmol`; JAX owns `tmol_jax`. Existing imports stay available.
Neither `tmol-core` nor the installed JAX runtime requires Torch. Native headers
may live under `tmol/` without depending on Torch registration or tensor storage.

Sharing an implementation does not establish frontend feature or performance
parity. Each frontend must implement its public APIs, execution adapters and
supported transformations, and validate complete molecular workflows.

| Layer | Common implementation | Frontend responsibility |
| --- | --- | --- |
| Chemistry | Raw records, YAML/units, patching, validation, refined residues, bond orders/rings, aliases, torsions and ideal coordinates | Database defaults and annotation lifetime |
| Parameters | Score records, generated overlays, residue parameter lookup, atom typing and table preparation | Archive readers, device transfer and cache lifetime |
| Topology | Arbitrary-residue packed metadata, connections, graph distances, canonical construction and molecular identity plans | Coordinate storage, padding capacity and device execution |
| Scoring | Whole-pose and sparse native traversal, scalar physics and coordinate derivatives | Allocation, streams, registration, differentiation and portable execution where supported |
| Packing | Sampler plans and native execution, interaction graphs, bump filtering, annealing and coordinate imposition | Task APIs, execution scheduling and supported design workflows |
| Optimization | Native segmented Armijo transitions and LBFGS state policy; FastRelax schedules | Objectives, parameter storage, reductions and linear algebra |
| Structure I/O | Readers/writers, chemistry contexts, ligand preparation, fragmentation, metadata and export plans | Tensor assembly and differentiable coordinate replay |

## Chemistry and parameter identity

`tmol_core.chemical.restypes.RefinedResidueType` refines shared raw records into
NumPy atom/bond indices, arbitrary connection paths, clipped graph distances and
internal coordinates. `tmol_core.chemical.packed.PackedBlockTypesData` prepares
ordered palettes without assuming canonical proteins. True connection counts
remain distinct from padded capacity; native execution limits are explicit.

`tmol_core.database.scoring.load_scoring_parameters` assembles standard scoring
records and generated overlays. Frontends supply Dunbrack, Rama and omega archive
readers and may adapt the raw HBond representation. File validation, parameter
precedence and missing-parameter errors stay in the shared preparation code.
For example, electrostatic charge lookup uses exact variants, terminal
conjugation, individual patches, then the base residue. Zero filling requires an
explicit request; it is not a substitute for missing active parameters.

Dunbrack probability logarithms use double precision before rounding to the
library storage dtype and applying the shared spline prefilter. Native torsion
angles use `atan2` to retain precision near planar geometries. Historical
float32 logarithms and `acos` torsions can differ numerically; dtype preservation
does not imply bitwise agreement with those older calculations.

`tmol_core.scoring.channels` defines global score indices and native output-lane
composition. Frontend enums retain their types and serialization paths. Score
file version parsing is shared; frontend score functions retain their constructors,
options and weight storage.

## Native execution and derivatives

Shared kernels accept typed views and caller-owned buffers. Torch registration
and JAX FFI supply allocations, devices, streams and derivative bindings. Shared
whole-pose and sparse kernels cover nonbonded, bonded/statistical, nucleic-acid,
metal and supported constraint physics. Scalar potential sharing alone is only
part of this boundary; traversal, connection identity and derivative accumulation
also belong to the common implementation.

The same separation applies to sampling, kinematics, graph preparation, annealing
and coordinate imposition. Size-discovery barriers and temporary-buffer lifetimes
are execution requirements. An adapter must preserve them instead of substituting
Python algorithms or silently dropping unsupported cases.

Native first derivatives do not imply arbitrary higher derivatives or every JAX
transformation. Chi construction supports sampled-angle derivatives with fixed
input coordinates; inverse-kinematics coordinate derivatives are not implemented.
Copied current conformers retain their coordinate derivatives. Atom37 input
completion treats sampled coordinates as selected values while preserving
supplied-atom mappings and frozen residues' built-atom derivatives.

Correlated sampling validates equal active counts, local-state correspondence and
connection layouts. A connected third backbone residue in an endpoint's lockstep
group uses its matching conformer and residue type. A third independently varying
block is outside this pairwise model. Separate metal bridge contributions and both
directed peptide bonds of a two-residue cycle are retained.

## Structure preparation and batching

Shared readers and writers handle PDB/CIF models and assemblies, compressed and
binary CIF, MOL2/SDF, alternate locations, CCD supplements, supplied bonds, metal
coordination, author identities and unresolved atoms. Protonation and terminal
policy use explicit chemical contexts and preserve custom residue identities.
The default patched chemistry is shared; loading an explicit path creates fresh
chemistry. Host caches do not own frontend device-stream synchronization.

`tmol_core.ligand` owns typing, protonation routing, charge overrides, parameter
files, residue registration, conjugate chemistry and chemical fragmentation.
Distance-geometry preparation shares targets, bounds, planar embedding, seeds and
restart selection. Its `ConformerExecution` protocol requires an executor for
planar/chiral/stress refinement. Complete supplied coordinates and authoritative
charges need no executor; this does not establish JAX regeneration support.

Atom37 host routing validates residue/atom ranges and unique source slots;
frameworks scatter coordinates using its paired indices. The default protein
encoding is shared data matching AtomWorks 3.0.4 without importing its ML layer.
Atom37 routing and slot policy preserve residue-specific terminal atoms and
custom layouts. `tmol_core.pose.concatenation` validates chemical-database
extensions by residue identity, merges ordered palettes, and preserves original
batch offsets, fragment identities and PDB metadata. It reads shapes and host
metadata only. Frontends copy coordinate/topology buffers and concatenate
constraints on the requested device.

The molecular sequence grammar and ideal extended-chain geometry live in
`tmol_core.pose.sequence` and `tmol_core.io.extended`. They use chemical residue
records and host topology arrays, including explicit residue names, chain
breaks and ligand SMILES. Frontends measure and apply torsions and place
coordinates on their devices. Shared query and mutation policy preserves
padding, author insertion codes, unknown one-letter identities and frozen
background blocks; sampler execution remains in each frontend. The one-letter
mutation API retains its mature amino-acid contract rather than accepting
arbitrary chemical identities as sequence letters.

## Building and validating

```bash
python -m pip install build
python scripts/build_core.py --outdir core-dist
python -m pip install core-dist/tmol_core-*.whl
# Pure frontend packages; native execution requires its matching artifact.
python scripts/build_jax.py --outdir jax-dist
```

Core resources include the canonical chemical YAML files and are independent of
the source checkout. Custom databases load through `ChemicalDatabase.from_file`;
JAX retains `TMOL_JAX_DATABASE`. The core installs its structure-chemistry
dependencies without either framework. `tmol-core[ligand]` adds optional OpenBabel
for preparation paths requiring its charge models or protonation fallback.

AtomWorks 3.0.4 can import Torch during structure preparation when Torch is
installed. Until an upstream release contains the import fix, use the explicit
commit-qualified constraint at `packages/constraints/atomworks-no-torch.txt`
for mixed Torch/JAX environments:

```bash
python -m pip install --upgrade --force-reinstall --no-deps \
  -r packages/constraints/atomworks-no-torch.txt
python -m pip install -c packages/constraints/atomworks-no-torch.txt \
  core-dist/tmol_core-*.whl
```

The first command replaces an existing same-version AtomWorks installation.
The second resolves the core's dependencies against the same source archive.
This temporary personal-fork commit changes optional ML imports; explicit Torch
conversion and RNG tracking still use Torch. The archive SHA256 is pinned and
pip records its origin in `direct_url.json`. Ordinary PyPI installation does not
apply this constraint and does not yet guarantee that Torch stays unloaded.

Run `pytest -o addopts= tests/core` for common contracts and the affected frontend
and device suites for integration. Compare matched molecular inputs, parameters,
energies, gradients, topology and full cold/warm workflows, including peak host
and device memory. Kernel tests alone do not establish end-to-end parity. See
[development](user_guide/development.md) and [benchmarking](user_guide/benchmarking.md).
Core code is BSD-3-Clause; vendored PySMILES retains Apache-2.0 attribution.
