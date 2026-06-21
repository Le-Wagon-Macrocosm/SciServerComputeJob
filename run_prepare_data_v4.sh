#!/usr/bin/env bash
# SciServer Compute Job entry point for prepare_data_v4.py — fill the MISSING
# sample_v4 cutouts (the new healthy galaxies that refilled catalog_v4 after the
# quality cuts, idx 545017..599999) into sample_v4 shards 90..99.
#
# Each band is cut with its OWN WCS (64x64) then centre-cropped to 24x24, matching
# sample_v4. Shard 90 is merged over its existing 5017 healthy rows; 91..99 are new.
#
#   bash run_prepare_data_v4.sh --shard 90-99          # all missing shards, in sequence
#   bash run_prepare_data_v4.sh --shard 91             # one shard
#   SHARDS=90-99 bash run_prepare_data_v4.sh           # SHARDS env -> --shard
#   bash run_prepare_data_v4.sh --smoke                # e2e check (cut a few, no upload)
#
# Extra flags forward to prepare_data_v4.py (--size, --crop, --workers, --sas, ...).
# On a Compute Job the SDSS volume usually mounts with a space:
#   --sas "/home/idies/workspace/SDSS SAS"
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="$(command -v python || command -v python3)"
if [ -z "$PY" ]; then echo "[run] no python interpreter found"; exit 127; fi
echo "[run] $("$PY" --version 2>&1)  ($PY)  cwd=$HERE  args=$*"

echo "[run] installing deps ..."
"$PY" -m pip install --quiet --upgrade pip 2>/dev/null || true
"$PY" -m pip install --quiet --user astropy google-cloud-storage gcsfs pyarrow pandas numpy 2>&1 | tail -1 || \
  echo "[run] pip install failed -> assuming deps already present"

CATALOG="${CATALOG:-gs://macrocosm-lewagon/data/sample_v4/new_objids_v4.parquet}"
KEY="${GCS_KEY:-sciserver-uploader.json}"
if [ ! -f "$KEY" ]; then
  echo "[run] WARNING: GCS key '$KEY' not found in $HERE — upload will fail."
  echo "[run]          put sciserver-uploader.json here or set GCS_KEY=/path/to/key.json"
fi

status=0
if [[ "$*" == *--smoke* ]]; then
  echo "[run] smoke e2e check (no upload)"
  "$PY" prepare_data_v4.py --catalog "$CATALOG" --key "$KEY" "$@"
else
  EXTRA=()
  [ -n "${SHARDS:-}" ] && EXTRA=(--shard "$SHARDS")
  "$PY" prepare_data_v4.py --catalog "$CATALOG" --key "$KEY" "${EXTRA[@]}" "$@"
fi
status=$?

echo "[run] exit=$status"
exit $status
