# SciServer Compute Jobs — Macrocosm photo-z

SciServer batch jobs for the **Macrocosm** photo-z project. Two jobs live here:

1. **Image-stamp build** (`prepare_data.py`) — cut the ugriz cutouts from the SDSS frames.
2. **Outlier CV** (`outlier_cv_job.py`) — 5-fold CV on the tabular catalog to find the hard set.

Designed for the **SciServer Compute Job — Large Jobs Domain** (32 cores, 240 GB), with the
SDSS DR17 SAS data volume mounted (needed for the image build).

---

## 1. Image-stamp build (`prepare_data.py`) → `sample_v2`

Cuts a 64×64×5 ugriz stamp per galaxy from the mounted SDSS frames and pushes each shard
straight to `gs://macrocosm-lewagon/data/sample_v2/` (zero download). **Each band is cut with
its own WCS**, so the 5 channels land registered. The old `sample_v1` build reused one band's
(u's) WCS for all 5 → the g/r/i/z channels were offset by the inter-band astrometry (3–11 px,
frame-dependent), corrupting every color. `sample_v2` is the registered rebuild; the catalog is
unchanged (idx alignment preserved → reuse `catalog_v1.parquet`).

- `prepare_data.py` — Step B: per (run,camcol,field) group, open the 5 frames once, cut each band
  with its own WCS, stack, upload shard `images_<start>_<end>.npy`. Idempotent (existing shard skipped).
- `freeze_catalog.py` — Step A: the one-time CasJobs query that produced `catalog_v1.parquet`
  (objid + ra/dec + run/camcol/field + tabular features + redshift). **Already run — only here for
  reproducibility; you do NOT re-run it for the registration fix.**
- `run_prepare_data.sh` — entry point: installs deps, runs one shard or a contiguous range.

```bash
# one shard (0..of-1):
bash run_prepare_data.sh --of 64 --shard 0 --sas "/home/idies/workspace/SDSS SAS"

# multiple shards in one container (range / list / mix), built in sequence:
bash run_prepare_data.sh --of 64 --shard 0-7      --sas "/home/idies/workspace/SDSS SAS"
bash run_prepare_data.sh --of 64 --shard 0,3,5    --sas "/home/idies/workspace/SDSS SAS"
SHARDS=0-7 bash run_prepare_data.sh --of 64       --sas "/home/idies/workspace/SDSS SAS"   # SHARDS env == --shard
```
`--shard` takes a single `5`, a range `0-7`, a list `0,3,5`, or a mix `0-3,8,10-11`.
Run **several Compute Jobs over disjoint shard sets** to parallelise across containers.
Put `sciserver-uploader.json` next to the scripts (or set `GCS_KEY=/path/to/key.json`). In a
Compute Job the SDSS volume mounts with a space, so pass `--sas "/home/idies/workspace/SDSS SAS"`
(the interactive default is `/home/idies/workspace/sdss_sas`).

---

## 2. Outlier CV (`outlier_cv_job.py`)

Trains **HGB / RF / MLP** in 5-fold cross-validation on a 400k subset of the SDSS catalog, collects
out-of-fold predictions, per-fold metrics, and the all-3-model outlier intersection (the "hard set"),
then tars everything up.

- `outlier_cv_job.py` — the job (load → clean -9999 → features → 5-fold CV → write files → tar).
- `run_outlier_job.sh` — entry point: installs deps, fetches `catalog_v1.parquet` (local or from GCS), runs the job.

```bash
bash run_outlier_job.sh --smoke   # 0.2k rows, verify the pipeline end to end (seconds)
bash run_outlier_job.sh           # full 400k run -> outlier_cv_results.tar.gz
```
Output (`outlier_cv_results.tar.gz`):
`oof_predictions.parquet` · `metrics_per_fold.csv` · `metrics_summary.csv` · `hard_objids.csv` · `run_info.json`

---

> Do **not** commit `catalog_v1.parquet`, the SA key (`sciserver-uploader.json`), or output tars.
