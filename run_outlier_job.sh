#!/usr/bin/env bash
# SciServer Compute Job entry point for the outlier-analysis CV.
#
#   bash run_outlier_job.sh --smoke     # 0.2k rows, verify the pipeline runs end to end
#   bash run_outlier_job.sh             # full 400k run -> outlier_cv_results.tar.gz
#
# Installs the Python deps, makes sure catalog_v1.parquet is present (pulls from GCS if not),
# then runs outlier_cv_job.py (passing through any args). Run it from the dir that holds
# outlier_cv_job.py (and ideally catalog_v1.parquet + the GCS key).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="$(command -v python || command -v python3)"     # SciServer has `python`; locally it may be `python3`
if [ -z "$PY" ]; then echo "[run] no python interpreter found"; exit 127; fi
echo "[run] $("$PY" --version 2>&1)  ($PY)  cwd=$HERE  args=$*"

# 1) dependencies (no-op if already installed; tolerated if pip is restricted)
echo "[run] installing deps ..."
"$PY" -m pip install --quiet --upgrade pip 2>/dev/null || true
"$PY" -m pip install --quiet scikit-learn pandas pyarrow 2>&1 | tail -1 || \
  echo "[run] pip install failed -> assuming deps already present"

# 2) catalog: use local copy if present, else download from GCS
CATALOG="${CATALOG:-catalog_v1.parquet}"
GCS="gs://macrocosm-lewagon/data/sample_v1/catalog_v1.parquet"
KEY="${GCS_KEY:-sciserver-uploader.json}"
if [ ! -f "$CATALOG" ] && [[ "$CATALOG" != gs://* ]]; then
  echo "[run] $CATALOG missing -> downloading from $GCS"
  [ -f "$KEY" ] && gcloud auth activate-service-account --key-file "$KEY" || true
  gcloud storage cp "$GCS" "$CATALOG"
fi

# 3) run the job (forwards --smoke and any other flags)
echo "[run] launching outlier_cv_job.py ..."
"$PY" outlier_cv_job.py "$@"
status=$?

echo "[run] exit=$status  artifacts:"
ls -lh outlier_cv_results*.tar.gz 2>/dev/null || echo "[run] (no tar found — check errors above)"
exit $status
