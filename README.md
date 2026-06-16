# SciServer Compute Job — photo-z outlier CV

Batch job for the **Macrocosm** photo-z outlier study. Trains **HGB / RF / MLP** in 5-fold
cross-validation on a 400k subset of the SDSS catalog, collects out-of-fold predictions, per-fold
metrics, and the all-3-model outlier intersection (the "hard set"), then tars everything up.

Designed for the **SciServer Compute Job — Large Jobs Domain** (32 cores, 240 GB).

## Files
- `outlier_cv_job.py` — the job (load → clean -9999 → features → 5-fold CV → write files → tar).
- `run_outlier_job.sh` — entry point: installs deps, fetches `catalog_v1.parquet` (local or from GCS), runs the job.

## Run
```bash
bash run_outlier_job.sh --smoke   # 0.2k rows, verify the pipeline end to end (seconds)
bash run_outlier_job.sh           # full 400k run -> outlier_cv_results.tar.gz
```
Put `catalog_v1.parquet` next to the scripts (or let the script pull it from
`gs://macrocosm-lewagon/data/sample_v1/catalog_v1.parquet` with the `sciserver-uploader` key).

## Output (`outlier_cv_results.tar.gz`)
`oof_predictions.parquet` · `metrics_per_fold.csv` · `metrics_summary.csv` · `hard_objids.csv` · `run_info.json`

> Do **not** commit `catalog_v1.parquet`, the SA key, or the output tar.
