# JAX frontend (experimental)

`tmol-jax` is a separately installable, Torch-free JAX frontend for TMol. Its
runtime imports `tmol_jax` and never `torch`; the existing PyTorch API, kernels
and wheels are unchanged. It is experimental: use it for fast beta2016 scoring,
gradients and Cartesian minimization of canonical proteins, and validate results
against the PyTorch frontend before relying on them.

## Supported

- beta2016 score terms for canonical amino acids, with coordinate gradients
  (`jax.grad`) and `jax.jit`;
- PDB or Biotite `AtomArray`/`AtomArrayStack` input, same-topology and mixed batches;
- reusable rendered scorers (`ScoreFunction.render_whole_pose_scoring_module`,
  `.jitted`, `.jitted_value_and_grad`);
- masked Cartesian L-BFGS minimization (`run_cart_min`);
- conservative no-pack Cartesian relaxation with an `fa_ljrep` ramp and optional
  mainchain constraints (`run_cart_relax`).

Not supported: nucleic acids, metals, ligands and other noncanonical residues
(these fail explicitly), kinematic minimization, repacking and FastRelax.

## Build and install

From a TMol developer environment (`tmol`, `numpy` and `build` installed):

```sh
python scripts/build_jax.py --outdir dist/jax
pip install dist/jax/tmol_jax-*.whl
```

The builder takes the version from the root `pyproject.toml` and converts the
canonical parameter tables to NumPy archives; Torch is used only for that
conversion. The wheel and standalone sdist need no Torch, and the sdist rebuilds
without it. For GPUs also install a matching JAX runtime, for example
`pip install 'jax[cuda13]>=0.11.2'` (the wheel declares `cu12`/`cu13` extras).
Validation uses Python 3.12 and JAX 0.11.2.

The optional `tmol-jax-cuda` wheel replaces the dense float32 pair sums for LJ/LK,
electrostatics, LK-ball and hydrogen bonds with thin JAX FFI adapters around
TMol's Torch-independent scalar headers (it needs `nvcc`):

```sh
python scripts/build_jax_cuda.py --outdir dist/jax-cuda --cuda-architectures "80;90"
pip install dist/jax-cuda/tmol_jax_cuda-*.whl
```

Without the adapter (or if it fails to load) the exact pure-JAX path is used. Its
ABI marker is checked before any handler is registered, so mismatched wheels fail
loudly. The adapter evaluates exact count-pair-qualified pairs from the current
coordinates on every call: there is no frozen neighbor list. Arrays of at least
2,048 atoms use conservative 16-atom tile bounds that skip only tile pairs beyond
the largest cutoff. For source development export the tables with
`python scripts/build_jax.py --export-data DIR` and set `TMOL_JAX_DATABASE=DIR`.

## Use

```python
import jax
from tmol_jax import (
    beta2016_score_function, pose_stack_from_pdb, run_cart_min, run_cart_relax,
)

pose = pose_stack_from_pdb("input.pdb")
score_function = beta2016_score_function()
score = score_function.render_whole_pose_scoring_module(pose)
energies = jax.jit(score)(pose.coords)
gradient = jax.jit(jax.grad(lambda x: score(x).sum()))(pose.coords)

minimized, result = run_cart_min(
    pose, score_function, pose.atom_is_real, max_iter=100, return_result=True
)
relaxed, diagnostics = run_cart_relax(
    pose, score_function, move_mask=pose.atom_is_real,
    constrain_mainchain=True, n_cycles=1, return_result=True,
)
```

A rendered scorer snapshots its weights, term options and topology; reuse it for
coordinate changes and render again after changing weights, types, connectivity
or batch layout. `sum_terms=False` returns channels in `ScoreType` order and
`apply_weights=False` the unweighted energies. `run_cart_min` renders once, keeps
topology fixed, accepts atom or xyz movement masks (padding is always frozen), and
returns a new pose; inspect `result["converged"]` and `result["status"]`.
`run_cart_relax` returns the best structure seen under the full-weight score and
tethers N, CA and C to their input coordinates when `constrain_mainchain=True`.
The frontend does not change global JAX precision settings; call
`block_until_ready()` when timing, and count compilation and setup separately.

## Validation and limitations

The test suite covers CPU and native CUDA scoring and gradients, canonical
hydrogen-bond tables, Torch-free installation and small-protein I/O. Float32
energies and gradients are not bitwise reproducible across frontends: native
kernels use atomic accumulation, and the JAX torsion implementation uses `atan2`
where the native implementation uses `acos`. Compare results with tolerances
appropriate to the structure, precision and score terms in use.

The loaders can produce different missing-atom coordinates and residue
orientations. A comparison on independently loaded coordinates includes those
preparation differences. Before comparing scorer energies or coordinate
gradients, verify residue and atom identities and use the same coordinates.
PDB loading uses the first model and first alternate location; chain breaks
must be explicit in the input.

Cartesian minimization uses a separate JAX L-BFGS implementation with a
strong-Wolfe line search. Its default iteration limit differs from the PyTorch
minimizer, so matching input coordinates does not imply matching trajectories,
termination or endpoints. Inspect the returned convergence diagnostics and
cross-score resulting coordinates when assessing agreement. The no-pack
repulsive-ramp relaxation is not a substitute for PyTorch FastRelax.

Performance depends on structure size, batch shape, device, compilation and
whether the native CUDA adapter is installed. Measure setup and compilation
separately from repeated scoring, and synchronize device work when timing.
Scoring performance does not establish minimization or relaxation performance.

The initial implementation is adapted from
[tudorcotet/jtmol](https://github.com/tudorcotet/jtmol) under Apache-2.0; see
`tmol_jax/NOTICE`.
