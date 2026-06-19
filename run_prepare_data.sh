#!/usr/bin/env bash
# SciServer Compute Job entry point for the image-stamp build (prepare_data.py).
#
# Cuts 64x64x5 ugriz stamps from the mounted SDSS DR17 frames and pushes each
# shard straight to GCS (gs://macrocosm-lewagon/data/sample_v2/). Each band is
# cut with its OWN WCS, so the 5 channels land registered (the sample_v1 build
# used one band's WCS for all 5 -> 3-11 px misregistration; sample_v2 fixes it).
#
#   # one shard (0..of-1):
#   bash run_prepare_data.sh --of 64 --shard 0
#
#   # a contiguous RANGE of shards in this one container (runs them in sequence):
#   SHARDS=0-7 bash run_prepare_data.sh --of 64
#
# Any extra flags are forwarded to prepare_data.py (--size, --dtype, --workers,
# --sas, --prefix, --force, ...). On a Compute Job the SDSS volume usually mounts
# with a space, so pass:  --sas "/home/idies/workspace/SDSS SAS"
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="$(command -v python || command -v python3)"     # SciServer has `python`; locally it may be `python3`
if [ -z "$PY" ]; then echo "[run] no python interpreter found"; exit 127; fi
echo "[run] $("$PY" --version 2>&1)  ($PY)  cwd=$HERE  args=$*"

# 1) dependencies (no-op if already installed; tolerated if pip is restricted)
echo "[run] installing deps ..."
"$PY" -m pip install --quiet --upgrade pip 2>/dev/null || true
"$PY" -m pip install --quiet --user astropy google-cloud-storage gcsfs pyarrow pandas numpy 2>&1 | tail -1 || \
  echo "[run] pip install failed -> assuming deps already present"

# 2) inputs (overridable via env)
CATALOG="${CATALOG:-gs://macrocosm-lewagon/data/sample_v1/catalog_v1.parquet}"  # catalog unchanged from v1
KEY="${GCS_KEY:-sciserver-uploader.json}"
if [ ! -f "$KEY" ]; then
  echo "[run] WARNING: GCS key '$KEY' not found in $HERE — upload will fail. "
  echo "[run]          put sciserver-uploader.json here or set GCS_KEY=/path/to/key.json"
fi

# 3) run. --smoke cuts a few galaxies to verify e2e (no shard, no upload);
#    SHARDS=a-b runs that contiguous range in sequence; otherwise pass --shard yourself.
status=0
if [[ "$*" == *--smoke* ]]; then
  echo "[run] smoke e2e check (no upload)"
  "$PY" prepare_data.py --catalog "$CATALOG" --key "$KEY" "$@"
  status=$?
elif [ -n "${SHARDS:-}" ]; then
  lo="${SHARDS%-*}"; hi="${SHARDS#*-}"
  echo "[run] building shards $lo..$hi in sequence"
  for s in $(seq "$lo" "$hi"); do
    echo "[run] === shard $s ==="
    "$PY" prepare_data.py --catalog "$CATALOG" --key "$KEY" --shard "$s" "$@"
    rc=$?; [ "$rc" -ne 0 ] && status=$rc && echo "[run] shard $s exited $rc"
  done
else
  "$PY" prepare_data.py --catalog "$CATALOG" --key "$KEY" "$@"
  status=$?
fi

echo "[run] exit=$status"
exit $status
