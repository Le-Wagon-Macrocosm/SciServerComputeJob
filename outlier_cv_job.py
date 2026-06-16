#!/usr/bin/env python3
"""Outlier-analysis CV job for SciServer Compute (Large domain: 32 cores, 240 GB).

Heavy compute on the server, light analysis on your laptop. This trains HGB / RF / MLP in
5-fold CV on a 400k subset of the photo-z catalog, collects:
  - out-of-fold predictions for every galaxy (the key artifact),
  - per-fold per-model metrics (MAE / sigma_MAD / outlier rate),
  - the 3-model outlier intersection (objids),
writes them to ./outlier_cv_out/ as several files, and tars to outlier_cv_results.tar.gz.

Download the tar, then locally rebuild everything from oof_predictions.parquet:
    dz = (pred - redshift) / (1 + redshift)   per model  -> histograms, outliers, etc.
and join hard_objids.csv to your local catalog for the 55-column distribution analysis.

Catalog: uses ./catalog_v1.parquet if present (recommended: copy it into the job dir once);
otherwise downloads from GCS (needs gcloud + the sciserver-uploader key). Override paths with
env vars:  CATALOG=/path/or/gs://...   N=400000   GCS_KEY=sciserver-uploader.json
"""
import os, sys, time, json, tarfile, subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error

GCS_CATALOG = "gs://macrocosm-lewagon/data/sample_v1/catalog_v1.parquet"
CATALOG = os.environ.get("CATALOG", "catalog_v1.parquet")
N       = int(os.environ.get("N", 400_000))
GCS_KEY = os.environ.get("GCS_KEY", "sciserver-uploader.json")
OUTDIR  = "outlier_cv_out"

# Smoke test: `python outlier_cv_job.py --smoke` (or SMOKE=1) runs 0.2k rows + tiny models,
# just to verify the whole pipeline (load -> CV -> write files -> tar) runs to the end.
SMOKE = ("--smoke" in sys.argv) or bool(os.environ.get("SMOKE"))
if SMOKE:
    N = 200
    OUTDIR = "outlier_cv_out_smoke"
    RF_TREES, MLP_ITERS, HGB_ITERS = 30, 60, 40
else:
    RF_TREES, MLP_ITERS, HGB_ITERS = 150, 300, 300

FEATS = ["dered_u", "dered_g", "dered_r", "dered_i", "dered_z", "g-r", "u-g", "r-i", "i-z",
         "log_expRad_r", "log_deVRad_r", "log_petroRad_r", "log_petroR50_r", "log_petroR90_r",
         "fracDeV_r", "conc_r"]


def get_catalog():
    """Local parquet if present, else pull from GCS (activating the SA key if available)."""
    if CATALOG.startswith("gs://"):
        return pd.read_parquet(CATALOG)
    if not os.path.exists(CATALOG):
        print(f"[job] {CATALOG} not found -> downloading {GCS_CATALOG}", flush=True)
        if os.path.exists(GCS_KEY):
            subprocess.run(["gcloud", "auth", "activate-service-account",
                            "--key-file", GCS_KEY], check=False)
        subprocess.run(["gcloud", "storage", "cp", GCS_CATALOG, CATALOG], check=True)
    return pd.read_parquet(CATALOG)


def build_features(cat):
    num = cat.select_dtypes("number").columns
    cat[num] = cat[num].mask(cat[num] <= -100)                       # clean SDSS -9999 sentinel
    for a, b in [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]:
        cat[f"{a}-{b}"] = (cat[f"dered_{a}"] - cat[f"dered_{b}"]).clip(-1, 4)
    for s in ["expRad_r", "deVRad_r", "petroRad_r", "petroR50_r", "petroR90_r"]:
        cat["log_" + s] = np.log1p(cat[s].clip(lower=0))
    cat["conc_r"] = cat["petroR90_r"] / cat["petroR50_r"].replace(0, np.nan)
    return cat


def make_models():
    return {
        "HGB": HistGradientBoostingRegressor(max_iter=HGB_ITERS, learning_rate=0.1,
                                             early_stopping=True, random_state=0),
        "RF":  RandomForestRegressor(n_estimators=RF_TREES, min_samples_leaf=2, n_jobs=-1, random_state=0),
        "MLP": make_pipeline(StandardScaler(),
                   MLPRegressor(hidden_layer_sizes=(128, 64, 32), alpha=1e-4, batch_size=256,
                                early_stopping=True, n_iter_no_change=12, max_iter=MLP_ITERS, random_state=0)),
    }


def smad(dz):
    return 1.4826 * np.median(np.abs(dz - np.median(dz)))


def main():
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"[job] {'SMOKE TEST ' if SMOKE else ''}cores={os.cpu_count()}  N={N}", flush=True)

    cat = build_features(get_catalog())
    D = cat[FEATS + ["redshift", "objid"]].replace([np.inf, -np.inf], np.nan).dropna()
    D = D.sample(min(N, len(D)), random_state=0).reset_index(drop=True)
    print(f"[job] D={len(D)} rows, {len(FEATS)} features  ({time.time()-t0:.0f}s)", flush=True)

    X, y, oid = D[FEATS], D["redshift"], D["objid"]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    oof = pd.DataFrame({"objid": oid.values, "redshift": y.values, "fold": -1,
                        "pred_HGB": np.nan, "pred_RF": np.nan, "pred_MLP": np.nan})
    metrics, inter_ids = [], []
    for f, (tr, te) in enumerate(kf.split(X)):
        tf = time.time()
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        oof.loc[te, "fold"] = f
        masks = {}
        for name, m in make_models().items():
            m.fit(Xtr, ytr)
            pred = m.predict(Xte)
            oof.loc[te, f"pred_{name}"] = pred
            dz = (pred - yte.values) / (1 + yte.values)
            masks[name] = np.abs(dz) > 0.05
            metrics.append((f, name, mean_absolute_error(yte, pred), smad(dz), float(masks[name].mean())))
        inter = masks["HGB"] & masks["RF"] & masks["MLP"]
        inter_ids += list(oid.iloc[te].values[inter])
        print(f"[job] fold {f}: inter={int(inter.sum())}  "
              + " ".join(f"{k}={int(v.sum())}" for k, v in masks.items())
              + f"  ({time.time()-tf:.0f}s)", flush=True)

    met = pd.DataFrame(metrics, columns=["fold", "model", "MAE", "sigma_MAD", "outlier_rate"])
    summary = met.groupby("model")[["MAE", "sigma_MAD", "outlier_rate"]].agg(["mean", "std"]).round(5)
    hard = pd.DataFrame({"objid": pd.Series(inter_ids, dtype="int64")})

    # --- write multiple files ---
    oof.to_parquet(f"{OUTDIR}/oof_predictions.parquet", index=False)   # objid, redshift, fold, pred_*
    met.to_csv(f"{OUTDIR}/metrics_per_fold.csv", index=False)
    summary.to_csv(f"{OUTDIR}/metrics_summary.csv")
    hard.to_csv(f"{OUTDIR}/hard_objids.csv", index=False)
    json.dump({"n_rows": int(len(D)), "n_features": len(FEATS), "features": FEATS,
               "n_folds": 5, "seed_sample": 0, "seed_kfold": 42,
               "hard_count": int(len(hard)), "hard_frac": round(len(hard) / len(D), 5),
               "cores": os.cpu_count(), "runtime_sec": round(time.time() - t0, 1),
               "sklearn": __import__("sklearn").__version__},
              open(f"{OUTDIR}/run_info.json", "w"), indent=2)

    # --- tar it up ---
    tarpath = "outlier_cv_results_smoke.tar.gz" if SMOKE else "outlier_cv_results.tar.gz"
    with tarfile.open(tarpath, "w:gz") as t:
        t.add(OUTDIR, arcname=os.path.basename(OUTDIR))

    print("\n[job] per-model metrics (mean over folds):\n",
          met.groupby("model")[["MAE", "sigma_MAD", "outlier_rate"]].mean().round(4), flush=True)
    print(f"[job] hard set: {len(hard)} ({len(hard)/len(D):.2%})", flush=True)
    print(f"[job] wrote {tarpath} ({os.path.getsize(tarpath)/1e6:.1f} MB)  "
          f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
