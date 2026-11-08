#!/usr/bin/env bash
# Stage scientific tests and their data, without exposing source runtime modules.
set -euo pipefail
destination=${1:?test source destination required}
mkdir -p "${destination}/tests" "${destination}/tmol/tests" \
  "${destination}/tmol/database/default"
cp tests/__init__.py "${destination}/tests/"
for suite in core scientific; do
  cp -R "tests/${suite}" "${destination}/tests/${suite}"
done
cp -R tmol/tests/data "${destination}/tmol/tests/data"
cp -R tmol/database/default/scoring "${destination}/tmol/database/default/scoring"
cp -R tmol/database/score_functions "${destination}/tmol/database/score_functions"
