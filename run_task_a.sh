#!/usr/bin/env bash
# Task A — per-frame band-registration offsets (header-only, reads the mounted SAS).
#   bash run_task_a.sh --sas "/home/idies/workspace/SDSS SAS" --workers 32
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
PY="$(command -v python || command -v python3)"
echo "[run] $("$PY" --version 2>&1)  cwd=$HERE  args=$*"
"$PY" -m pip install --quiet --user astropy pandas pyarrow numpy 2>&1 | tail -1 || \
  echo "[run] pip install skipped (assuming deps present)"
"$PY" task_a_offsets.py "$@"
echo "[run] artifacts:"; ls -lh objid_frame.csv frame_offset.csv 2>/dev/null
