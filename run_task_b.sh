#!/usr/bin/env bash
# Task B — sample_v1 -> registered + cropped sample_v3 (pulls/pushes GCS).
#   bash run_task_b.sh --shard 0-99 --workers 32
# Needs Task A's frame_offset.csv + objid_frame.csv in this dir (or pass --offsets/--catalog),
# and sciserver-uploader.json here (GCS via the google-cloud-storage lib + SA key; no gsutil needed).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
PY="$(command -v python || command -v python3)"
echo "[run] $("$PY" --version 2>&1)  cwd=$HERE  args=$*"
"$PY" -m pip install --quiet --user scipy numpy pandas pyarrow google-cloud-storage 2>&1 | tail -1 || \
  echo "[run] pip install skipped (assuming deps present)"
"$PY" task_b_build_v3.py "$@"
