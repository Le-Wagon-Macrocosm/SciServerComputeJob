#!/usr/bin/env bash
# SciServer Compute Job entry point for the fusion tabular baselines (2 feature sets, smart per-base
# outlier-analysis decision, task-parallel across all cores).
#
#   bash run_tabular_baseline.sh --smoke   # 4k pool, tiny models — verify the pipeline runs end to end
#   bash run_tabular_baseline.sh           # full run -> tabular_baseline_results.tar.gz (+ uploads pkls to GCS)
#
# Pulls catalog_v4 + split CSVs from GCS if absent, installs deps, then runs tabular_baseline_job.py
# (passing through any args). On the Large domain (32 cores) the 18 OOF fits per phase run concurrently.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="$(command -v python || command -v python3)"
if [ -z "$PY" ]; then echo "[run] no python interpreter found"; exit 127; fi
echo "[run] $("$PY" --version 2>&1)  ($PY)  cwd=$HERE  cores=$(nproc 2>/dev/null || echo '?')  args=$*"

echo "[run] installing deps ..."
"$PY" -m pip install --quiet --upgrade pip 2>/dev/null || true
"$PY" -m pip install --quiet scikit-learn pandas pyarrow joblib google-cloud-storage 2>&1 | tail -1 || \
  echo "[run] pip install failed -> assuming deps already present"

# GCS is handled inside the job via the google-cloud-storage lib (SciServer has no gcloud CLI).
# It needs the SA key present to pull catalog/splits and upload the pkls.
KEY="${GCS_KEY:-sciserver-uploader.json}"
CATALOG="${CATALOG:-catalog_v4.parquet}"
if [ ! -f "$CATALOG" ] && [ ! -f "$KEY" ]; then
  echo "[run] WARNING: neither $CATALOG nor key '$KEY' found here — the job can't fetch the catalog."
  echo "[run]          put catalog_v4.parquet and/or sciserver-uploader.json in $HERE (or set CATALOG / GCS_KEY)."
fi

echo "[run] launching tabular_baseline_job.py ..."
PYTHONUNBUFFERED=1 "$PY" -u tabular_baseline_job.py "$@"   # -u + flush=True => real-time progress in the job log
status=$?

echo "[run] exit=$status  artifacts:"
ls -lh tabular_baseline_results*.tar.gz baseline_out*/summary_*.json 2>/dev/null || echo "[run] (nothing — check errors above)"
exit $status
