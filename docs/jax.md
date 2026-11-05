# JAX frontend (experimental)

`tmol-jax` provides Torch-free canonical-protein scoring, coordinate gradients
and Cartesian minimization, plus shared native generic-bonded, Cartesian-bonded, backbone, nucleic-acid torsion and metal coordination
scoring for explicitly parameterized noncanonical molecules. LJ/LK and electrostatic terms also accept
explicit ligand, D-residue, nucleic-acid and metal inputs; complete scoring for
those domains remains unsupported. It shares immutable parameter contexts, chemical records, topology,
parameter preparation and whole-pose bonded physics with the Torch frontend through [`tmol-core`](shared_core.md).
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

For CUDA, install a matching runtime, such as `jax[cuda13]`. An optional native
adapter accelerates float32 LJ/LK, electrostatics, hydrogen bonds and LK-ball
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
Minimization accepts atom or xyz movement masks, freezes padding and returns a
new pose. Check `result["converged"]` and `result["status"]`.

`run_cart_relax` performs a no-pack repulsive-weight ramp with optional mainchain
coordinate constraints. It retains the best full-weight structure. It does not
implement FastRelax. Complete nucleic-acid, metal and ligand/noncanonical
scoring, repacking and kinematic minimization remain unsupported.

For already parameterized chemistry, `pose_stack_from_residue_blocks()` accepts
a batch of `ResidueBlock(residue_type, atom_names, coords)` records and explicit
named connection ports. Full residue names include variants; every atom must be
present exactly once. The loader preserves atom identity and floating precision,
and does not infer bonds or change chemistry. `return_context=True` returns the
pose and a context whose `parameter_database` must be used for scoring. Molecular
input support does not imply complete scoring support: see the current
[capability matrix](capability-matrix.json).

Generic, Cartesian bonded, backbone, nucleic-acid torsion and metal coordination scoring require the shared native CPU/CUDA package
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
sequential initially. Generic, Cartesian bonded, backbone, nucleic-acid torsion and metal coordination support first-order JVP and reverse gradients in
float32/64. Their native derivative kernels do not provide higher derivatives. The older
native nonbonded custom-VJP route does not support JVP.

Before comparing frontends, verify residue/atom identities and use identical
coordinates and parameters: loaders can idealize missing atoms and hydrogens
differently. PDB loading selects the first model/alternate location; represent
chain breaks explicitly. Float32 reductions and torsion formulas differ between
execution routes, so energies and gradients need scientifically justified
term-specific tolerances. The separate L-BFGS implementation need not follow the
same trajectory or stopping point as Torch.

Record setup, compilation and steady execution separately; synchronize with
`block_until_ready()` for timing. Portable pair tables still have quadratic
storage. The frontend does not change global JAX precision settings.
The initial implementation derives from
[tudorcotet/jtmol](https://github.com/tudorcotet/jtmol); see `tmol_jax/NOTICE`.
