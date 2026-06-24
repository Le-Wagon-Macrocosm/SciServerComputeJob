#!/usr/bin/env bash
# SciServer Compute Job: SHAP feature-importance for the 3 tabular stacking models (task-parallel).
#
#   bash run_shap_tabular.sh --smoke   # 40 rows, verify it runs end to end
#   bash run_shap_tabular.sh           # full (N_EXPLAIN=400) -> shap_tabular_results.tar.gz + uploads md to gs
#
# Pulls catalog_v4 + the model pkls from GCS via the google-cloud-storage lib (no gcloud CLI on SciServer);
# needs sciserver-uploader.json present. Splits each base's explained rows into chunks across all cores.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
PY="$(command -v python || command -v python3)"
[ -z "$PY" ] && { echo "[run] no python"; exit 127; }
echo "[run] $("$PY" --version 2>&1)  cores=$(nproc 2>/dev/null || echo '?')  args=$*"

echo "[run] installing deps ..."
"$PY" -m pip install --quiet --upgrade pip 2>/dev/null || true
"$PY" -m pip install --quiet shap scikit-learn pandas pyarrow joblib matplotlib google-cloud-storage 2>&1 | tail -1 || \
  echo "[run] pip install failed -> assuming deps already present"

KEY="${GCS_KEY:-sciserver-uploader.json}"
[ ! -f "${CATALOG:-catalog_v4.parquet}" ] && [ ! -f "$KEY" ] && \
  echo "[run] WARNING: no catalog and no key '$KEY' here — the job can't fetch data."

echo "[run] launching shap_tabular_job.py ..."
PYTHONUNBUFFERED=1 "$PY" -u shap_tabular_job.py "$@"
status=$?
echo "[run] exit=$status  artifacts:"; ls -lh shap_tabular_results*.tar.gz shap_out/*.md 2>/dev/null || echo "[run] (nothing)"
exit $status
