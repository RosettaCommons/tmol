#!/usr/bin/env bash
# Build and test tmol-jax in a clean JAX-only environment on an allocated GPU.
set -euo pipefail

: "${GITHUB_WORKSPACE:?}"
: "${GPU_ALLOC_SENTINEL:?}"
JAX_DIST=${JAX_DIST:-jax-dist}
JAX_CUDA_DIST=${JAX_CUDA_DIST:-jax-cuda-dist}
JAX_BUILD_PYTHON=${JAX_BUILD_PYTHON:-.venv/bin/python}

# shellcheck source=/dev/null
source .github/ci/gpu_env.sh
touch_gpu_sentinel
strip_cuda_compat_from_ld_path

"${JAX_BUILD_PYTHON}" -m pip install build jax 'setuptools>=77' wheel
"${JAX_BUILD_PYTHON}" scripts/build_jax_native.py --cuda \
  --outdir "${JAX_CUDA_DIST}" --cuda-architectures native

work=$(mktemp -d)
trap 'rm -rf "${work}"' EXIT
python -m venv "${work}/env"
case "$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -1)" in
  12) extra=cuda12 ;;
  13) extra=cuda13 ;;
  *) echo "CUDA 12 or 13 is required" >&2; exit 1 ;;
esac
"${work}/env/bin/python" -m pip install --upgrade pip
"${work}/env/bin/python" -m pip install pytest 'ipython>=8.0' "jax[${extra}]>=0.11.2"
core_wheels=("${JAX_DIST}"/tmol_core-*.whl)
"${work}/env/bin/python" -m pip install \
  "${core_wheels[0]}[ligand]" "${JAX_DIST}"/tmol_jax-*.whl \
  "${JAX_CUDA_DIST}"/tmol_jax_native-*.whl

"${JAX_BUILD_PYTHON}" scripts/jax_release_manifest.py --revision "$(git rev-parse HEAD)" \
  --core-wheel "${JAX_DIST}"/tmol_core-*.whl --wheel "${JAX_DIST}"/tmol_jax-*.whl \
  --native-wheel "${JAX_CUDA_DIST}"/tmol_jax_native-*.whl --output "${work}/release.json"

bash .github/ci/stage_jax_tests.sh "${work}/source"

version=$("${work}/env/bin/python" -I -c \
  'from importlib.metadata import version; print(version("tmol-jax"))')
(
  cd "${work}/source"
  export JAX_ENABLE_X64=true
  TMOL_TEST_BACKENDS=jax-cuda "${work}/env/bin/python" -I -m pytest -ra tests
  "${work}/env/bin/python" -I \
    "${GITHUB_WORKSPACE}/scripts/smoke_installed_jax.py" \
    --pdb tmol/tests/data/pdb/1ubq.pdb --version "${version}" \
    --device cuda --expect-no-torch --require-native-cuda \
    --manifest "${work}/release.json"
)
