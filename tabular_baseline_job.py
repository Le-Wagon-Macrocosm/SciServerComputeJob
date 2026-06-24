#!/usr/bin/env python3
"""Tabular baseline job for SciServer Compute (Large: 32 cores, 240 GB).

Trains the FUSION tabular baselines for TWO feature sets, with a smart per-base outlier-analysis
decision (train each base on the full pool OR with the 3-model hard set removed, whichever lowers
its sigma_MAD), then a LinearRegression meta over the 3 base preds. Maximises core usage by running
every independent (set x model x fold x variant) fit as its own task in one process pool (RF n_jobs=1,
so tasks don't fight over cores); the only barriers are full-OOF -> hard-set -> clean-OOF -> final.

Feature sets:
  set1 = 5 dered mags + 4 colours                                           (9 feats)
  set2 = z mag + z mag-error + 4 colours + star/galaxy sep + concentration  (8 feats)

Per set it writes baseline_out/{base_<set>_<RF|HGB|MLP>.pkl, meta_<set>.pkl, summary_<set>.json},
tars to tabular_baseline_results.tar.gz, and (if a GCS key is present) uploads the pkls to
gs://macrocosm-lewagon/models/ so the Colab fusion notebooks can pull them.

env:  CATALOG=/path/or/gs://...   N_POOL=300000   GCS_KEY=sciserver-uploader.json   CORES=<n>
smoke: `python tabular_baseline_job.py --smoke`
"""
import os, sys, time, json, tarfile, subprocess
import numpy as np, pandas as pd, joblib
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# GCS via the google-cloud-storage lib + SA key (SciServer has NO gcloud CLI)
BUCKET      = "macrocosm-lewagon"
CAT_BLOB    = "data/sample_v4.5/catalog_v4.parquet"
SPLITS_BLOB = "data/splits"                 # /<name>.csv
MODELS_BLOB = "models"                      # upload base_*/meta_* here
CATALOG = os.environ.get("CATALOG", "catalog_v4.parquet")
GCS_KEY = os.environ.get("GCS_KEY", "sciserver-uploader.json")
N_POOL  = int(os.environ.get("N_POOL", 300_000))
CORES   = int(os.environ.get("CORES", os.cpu_count() or 8))
OUTDIR  = "baseline_out"
ORDER, FOLDS, THR = ["RF", "HGB", "MLP"], 3, 0.05

SMOKE = ("--smoke" in sys.argv) or bool(os.environ.get("SMOKE"))
if SMOKE:
    N_POOL, OUTDIR = 4000, "baseline_out_smoke"
    RF_TREES, HGB_ITERS, MLP_ITERS = 30, 40, 60
else:
    RF_TREES, HGB_ITERS, MLP_ITERS = 150, 300, 300


# ---------------- feature sets (catalog_v4 columns -> X) ----------------
def _col(cat, n):
    return cat[n].mask(cat[n] <= -100).to_numpy("float32")

def feats_set1(cat):
    du, dg, dr, di, dz = (_col(cat, f"dered_{b}") for b in "ugriz")
    TAB = ["dered_u", "dered_g", "dered_r", "dered_i", "dered_z", "u-g", "g-r", "r-i", "i-z"]
    f = {"dered_u": du, "dered_g": dg, "dered_r": dr, "dered_i": di, "dered_z": dz,
         "u-g": np.clip(du - dg, -1, 4), "g-r": np.clip(dg - dr, -1, 4),
         "r-i": np.clip(dr - di, -1, 4), "i-z": np.clip(di - dz, -1, 4)}
    return np.stack([f[c] for c in TAB], 1).astype("float32"), TAB

def feats_set2(cat):
    du, dg, dr, di, dz = (_col(cat, f"dered_{b}") for b in "ugriz")
    R50, R90 = _col(cat, "petroR50_r"), _col(cat, "petroR90_r")
    TAB = ["dered_z", "modelMagErr_z", "u-g", "g-r", "r-i", "i-z", "sg_sep", "conc_r"]
    f = {"dered_z": dz, "modelMagErr_z": _col(cat, "modelMagErr_z"),
         "u-g": np.clip(du - dg, -1, 4), "g-r": np.clip(dg - dr, -1, 4),
         "r-i": np.clip(dr - di, -1, 4), "i-z": np.clip(di - dz, -1, 4),
         "sg_sep": _col(cat, "psfMag_r") - _col(cat, "modelMag_r"),
         "conc_r": R90 / np.where(R50 == 0, np.nan, R50)}
    return np.stack([f[c] for c in TAB], 1).astype("float32"), TAB

SETS = {"set1": feats_set1, "set2": feats_set2}


def mk_base(name):
    if name == "RF":  return RandomForestRegressor(n_estimators=RF_TREES, min_samples_leaf=2, n_jobs=1, random_state=0)
    if name == "HGB": return HistGradientBoostingRegressor(max_iter=HGB_ITERS, learning_rate=0.1, early_stopping=True, random_state=0)
    return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 64, 32), alpha=1e-4, batch_size=1024,
                         learning_rate_init=1e-3, early_stopping=True, n_iter_no_change=12, max_iter=MLP_ITERS, random_state=0))

def smad(zt, zp):
    zt, zp = np.asarray(zt, float), np.asarray(zp, float); d = (zp - zt) / (1 + zt)
    return float(1.4826 * np.median(np.abs(d - np.median(d))))
def outl(zt, zp):
    zt, zp = np.asarray(zt, float), np.asarray(zp, float)
    return float(np.mean(np.abs((zp - zt) / (1 + zt)) > THR))


# --- the atomic parallel task: fit one base on rows `tr`, predict rows `te` (or whole pool) ---
def _fit_predict(X, y, tr, te, model):
    return mk_base(model).fit(X[tr], y[tr]).predict(X[te])
def _fit_final(X, y, rows, model):
    return mk_base(model).fit(X[rows], y[rows])


def _bucket():
    from google.cloud import storage
    return storage.Client.from_service_account_json(GCS_KEY).bucket(BUCKET)

def get_csv(local, blob):
    if not os.path.exists(local):
        _bucket().blob(blob).download_to_filename(local)
    return pd.read_csv(local)


def main():
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"[job] {'SMOKE ' if SMOKE else ''}cores={CORES} N_POOL={N_POOL}", flush=True)

    # catalog + splits (pull from GCS via the lib if not already local)
    if not os.path.exists(CATALOG):
        print(f"[job] {CATALOG} missing -> downloading gs://{BUCKET}/{CAT_BLOB}", flush=True)
        _bucket().blob(CAT_BLOB).download_to_filename(CATALOG)
    cat = pd.read_parquet(CATALOG)
    objid = cat["objid"].to_numpy("int64"); z = cat["redshift"].to_numpy("float64")
    train_ids = set(get_csv("train_objids.csv", f"{SPLITS_BLOB}/train_objids.csv")["objid"].astype("int64"))
    val_ids   = set(get_csv("val_objids.csv",   f"{SPLITS_BLOB}/val_objids.csv")["objid"].astype("int64"))
    is_tr, is_va = np.isin(objid, list(train_ids)), np.isin(objid, list(val_ids))
    print(f"[job] catalog {len(cat):,} | train {is_tr.sum():,} | val {is_va.sum():,}  ({time.time()-t0:.0f}s)", flush=True)

    # build per-set data (pool for OOF/final, val for eval)
    S = {}
    for key, fn in SETS.items():
        X, TAB = fn(cat)
        X[~np.isfinite(X)] = np.nan
        keep = ~np.isnan(X).any(1)
        pool = np.where(keep & is_tr)[0]
        np.random.RandomState(0).shuffle(pool); pool = pool[:N_POOL]
        vidx = np.where(keep & is_va)[0]
        S[key] = dict(TAB=TAB, Xp=X[pool], yp=z[pool], Xv=X[vidx], zv=z[vidx], n_val=len(vidx))
        print(f"[job] {key}: {len(TAB)} feats | pool {len(pool):,} | val50k {len(vidx):,}", flush=True)

    kf = KFold(FOLDS, shuffle=True, random_state=42)
    splits = {k: list(kf.split(S[k]["Xp"])) for k in SETS}

    # ---- Phase A: full OOF (all set x model x fold tasks in parallel) ----
    tA = time.time()
    jobsA = [(k, m, fi, tr, te) for k in SETS for m in ORDER for fi, (tr, te) in enumerate(splits[k])]
    resA = Parallel(n_jobs=CORES, backend="loky", verbose=5)(
        delayed(_fit_predict)(S[k]["Xp"], S[k]["yp"], tr, te, m) for (k, m, fi, tr, te) in jobsA)
    oof_full = {k: {m: np.full(len(S[k]["yp"]), np.nan) for m in ORDER} for k in SETS}
    omask    = {k: {m: np.zeros(len(S[k]["yp"]), bool)  for m in ORDER} for k in SETS}
    for (k, m, fi, tr, te), pred in zip(jobsA, resA):
        oof_full[k][m][te] = pred
        yte = S[k]["yp"][te]; omask[k][m][te] = np.abs((pred - yte) / (1 + yte)) > THR
    hard = {k: (omask[k]["RF"] & omask[k]["HGB"] & omask[k]["MLP"]) for k in SETS}
    for k in SETS: print(f"[job] {k} hard set: {int(hard[k].sum())} ({hard[k].mean():.2%})", flush=True)
    print(f"[job] phase A (full OOF, {len(jobsA)} fits) {time.time()-tA:.0f}s", flush=True)

    # ---- Phase B: clean OOF (remove hard from each fold's train) ----
    tB = time.time()
    jobsB = [(k, m, fi, tr[~hard[k][tr]], te) for k in SETS for m in ORDER for fi, (tr, te) in enumerate(splits[k])]
    resB = Parallel(n_jobs=CORES, backend="loky", verbose=5)(
        delayed(_fit_predict)(S[k]["Xp"], S[k]["yp"], trc, te, m) for (k, m, fi, trc, te) in jobsB)
    oof_clean = {k: {m: np.full(len(S[k]["yp"]), np.nan) for m in ORDER} for k in SETS}
    for (k, m, fi, trc, te), pred in zip(jobsB, resB): oof_clean[k][m][te] = pred
    flags = {k: {m: smad(S[k]["yp"], oof_clean[k][m]) < smad(S[k]["yp"], oof_full[k][m]) for m in ORDER} for k in SETS}
    for k in SETS:
        for m in ORDER:
            sf, sc = smad(S[k]["yp"], oof_full[k][m]), smad(S[k]["yp"], oof_clean[k][m])
            print(f"[job] {k} {m}: full {sf:.5f} vs rm-hard {sc:.5f} -> {'REMOVE-HARD' if flags[k][m] else 'FULL'}", flush=True)
    print(f"[job] phase B (clean OOF, {len(jobsB)} fits) {time.time()-tB:.0f}s", flush=True)

    # ---- Phase C: final frozen bases (chosen variant) in parallel ----
    tC = time.time()
    jobsC = [(k, m, (np.where(~hard[k])[0] if flags[k][m] else np.arange(len(S[k]["yp"])))) for k in SETS for m in ORDER]
    resC = Parallel(n_jobs=CORES, backend="loky", verbose=5)(
        delayed(_fit_final)(S[k]["Xp"], S[k]["yp"], rows, m) for (k, m, rows) in jobsC)
    bases = {k: {} for k in SETS}
    for (k, m, rows), est in zip(jobsC, resC): bases[k][m] = est
    print(f"[job] phase C (final bases, {len(jobsC)} fits) {time.time()-tC:.0f}s", flush=True)

    # ---- meta + val50k eval + save per set ----
    summary = {}
    for k in SETS:
        sel = np.column_stack([(oof_clean[k] if flags[k][m] else oof_full[k])[m] for m in ORDER])
        lr = LinearRegression().fit(sel, S[k]["yp"])
        zb = lr.predict(np.column_stack([bases[k][m].predict(S[k]["Xv"]) for m in ORDER]))
        sm, ou = smad(S[k]["zv"], zb), outl(S[k]["zv"], zb)
        summary[k] = {"features": S[k]["TAB"], "remove_hard": flags[k], "n_val": S[k]["n_val"],
                      "val50k_sigma_MAD": round(sm, 5), "val50k_outlier": round(ou, 4),
                      "meta_coef": [round(float(c), 4) for c in lr.coef_]}
        for m in ORDER:
            joblib.dump({"model": bases[k][m], "name": m, "feature_set": k, "features": S[k]["TAB"],
                         "remove_hard": flags[k][m], "target": "redshift (raw z)"},
                        f"{OUTDIR}/base_{k}_{m}.pkl", compress=3)
        joblib.dump({"model": lr, "base_order": ORDER, "feature_set": k}, f"{OUTDIR}/meta_{k}.pkl", compress=3)
        json.dump(summary[k], open(f"{OUTDIR}/summary_{k}.json", "w"), indent=2)
        print(f"[job] {k} val50k: sigma_MAD={sm:.5f} outlier={ou*100:.2f}% | {flags[k]}", flush=True)

    json.dump({"sets": summary, "cores": CORES, "n_pool": N_POOL, "runtime_sec": round(time.time() - t0, 1),
               "sklearn": __import__("sklearn").__version__}, open(f"{OUTDIR}/run_info.json", "w"), indent=2)

    tar = "tabular_baseline_results_smoke.tar.gz" if SMOKE else "tabular_baseline_results.tar.gz"
    with tarfile.open(tar, "w:gz") as t: t.add(OUTDIR, arcname=os.path.basename(OUTDIR))
    print(f"[job] wrote {tar}", flush=True)

    # upload the pkls to GCS (via the lib) so the Colab fusion notebooks can pull them
    if not SMOKE and os.path.exists(GCS_KEY):
        try:
            b = _bucket()
            for k in SETS:
                for m in ORDER:
                    b.blob(f"{MODELS_BLOB}/base_{k}_{m}.pkl").upload_from_filename(f"{OUTDIR}/base_{k}_{m}.pkl")
                b.blob(f"{MODELS_BLOB}/meta_{k}.pkl").upload_from_filename(f"{OUTDIR}/meta_{k}.pkl")
            print(f"[job] uploaded base_*/meta_* pkls -> gs://{BUCKET}/{MODELS_BLOB}/", flush=True)
        except Exception as e:
            print(f"[job] GCS upload failed ({e}); the local tar still has everything", flush=True)

    print(f"[job] DONE total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
