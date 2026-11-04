# Shared chemistry and execution boundaries

`tmol-core` owns the framework-independent `tmol_core` namespace: chemical
records, YAML and unit parsing, residue patching and validation, and the vendored
PySMILES reader/writer. Torch and JAX consume this same implementation. Existing
imports under `tmol.database.chemical`, `tmol.database._patched_chemdb`, and
`tmol_jax.database.chemical` remain available. JAX's patched database additionally
refines the shared raw records into its host topology arrays.

The `tmol` distribution retains the public Torch API and native CPU/CUDA/Metal
bindings. Renaming it to `tmol-torch` would require installation and metadata
migration without improving this dependency boundary; `import tmol` is unchanged.
`tmol-jax` owns JAX arrays, tracing, differentiation and execution. Neither it nor
`tmol-core` requires Torch at runtime. Each namespace has one distribution owner.

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
