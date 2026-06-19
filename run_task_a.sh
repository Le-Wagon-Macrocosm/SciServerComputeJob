#!/usr/bin/env bash
# Task A — per-frame band-registration offsets (header-only, reads the mounted SAS).
# Split into 18 fragments; run all in one job, or disjoint fragments as separate jobs:
#   bash run_task_a.sh --fragment 0-17 --sas "/home/idies/workspace/SDSS SAS" --workers 32
#   bash run_task_a.sh --fragment 0-5  --sas "/home/idies/workspace/SDSS SAS"   # one job's share
# Then merge:  python merge_offsets.py
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
PY="$(command -v python || command -v python3)"
echo "[run] $("$PY" --version 2>&1)  cwd=$HERE  args=$*"
"$PY" -m pip install --quiet --user astropy pandas pyarrow numpy google-cloud-storage 2>&1 | tail -1 || \
  echo "[run] pip install skipped (assuming deps present)"
"$PY" task_a_offsets.py "$@"
echo "[run] artifacts:"; ls -lh objid_frame.csv frame_offset_*.csv 2>/dev/null
