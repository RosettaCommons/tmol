#!/usr/bin/env bash
# Install the pure wheel without Torch and run JAX tests outside the checkout.
set -euo pipefail

: "${GITHUB_WORKSPACE:?}"
JAX_DIST=${JAX_DIST:-jax-dist}

work=$(mktemp -d)
trap 'rm -rf "${work}"' EXIT
python -m venv "${work}/env"
"${work}/env/bin/python" -m pip install --upgrade pip
"${work}/env/bin/python" -m pip install pytest "${JAX_SPEC:-jax>=0.11.2}"
core_wheels=("${JAX_DIST}"/tmol_core-*.whl)
"${work}/env/bin/python" -m pip install "${core_wheels[0]}[ligand]" "${JAX_DIST}"/tmol_jax-*.whl
"${work}/env/bin/python" -m pip install build 'setuptools>=77' wheel
"${work}/env/bin/python" scripts/build_jax_native.py --outdir "${work}/native-dist"
"${work}/env/bin/python" -m pip install "${work}/native-dist"/tmol_jax_native-*.whl

"${work}/env/bin/python" scripts/jax_release_manifest.py --revision "$(git rev-parse HEAD)" \
  --core-wheel "${JAX_DIST}"/tmol_core-*.whl --wheel "${JAX_DIST}"/tmol_jax-*.whl \
  --native-wheel "${work}/native-dist"/tmol_jax_native-*.whl --output "${work}/release.json"

bash .github/ci/stage_jax_tests.sh "${work}/source"

version=$("${work}/env/bin/python" -I -c \
  'from importlib.metadata import version; print(version("tmol-jax"))')
(
  cd "${work}/source"
  TMOL_TEST_BACKENDS=jax-cpu JAX_PLATFORMS=cpu "${work}/env/bin/python" -I -m pytest -ra tests
  JAX_PLATFORMS=cpu "${work}/env/bin/python" -I \
    "${GITHUB_WORKSPACE}/scripts/smoke_installed_jax.py" \
    --pdb tmol/tests/data/pdb/1ubq.pdb --version "${version}" \
    --device cpu --expect-no-torch --manifest "${work}/release.json"
)
