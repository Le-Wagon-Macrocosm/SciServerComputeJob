#!/usr/bin/env python3
"""SHAP (Shapley) feature-importance for the 3 tabular photo-z stacking models, on SciServer (32 cores).

  16feat — v4 baseline_stack (16 feats)               -> shap_out/shap_16feat.md
  set1   — 5 mags + 4 colours (9)                      -> shap_out/shap_set1.md
  set2   — z mag + z-err + 4 colours + sg-sep + conc   -> shap_out/shap_set2.md

Each model is a stack: 3 frozen bases (RF/HGB/MLP) -> LinearRegression meta. The meta is LINEAR over
the base preds, so SHAP(z) = sum_m w_m * SHAP(base_m). RF/HGB use the exact TreeExplainer, MLP uses a
Permutation explainer. The cost is dominated by the deep RFs, so each base's explained rows are SPLIT
into chunks and the (model x base x chunk) tasks run in one process pool to fill all cores.

Pulls catalog_v4 + the model pkls from GCS (google-cloud-storage lib + sciserver-uploader.json), writes
3 markdown reports, tars to shap_tabular_results.tar.gz, and uploads the .md to gs://.../results/shap/.

env:  CATALOG=...  N_EXPLAIN=400  N_BG=50  CORES=<n>  GCS_KEY=sciserver-uploader.json   smoke: --smoke
"""
import os, sys, time, json, glob, tarfile
import numpy as np, pandas as pd, joblib, shap
from joblib import Parallel, delayed
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib.externals.loky import get_reusable_executor

BUCKET = "macrocosm-lewagon"
CAT_BLOB = "data/sample_v4.5/catalog_v4.parquet"
MODELS_BLOB = "models"
RESULTS_BLOB = "results/shap"
CATALOG = os.environ.get("CATALOG", "catalog_v4.parquet")
GCS_KEY = os.environ.get("GCS_KEY", "sciserver-uploader.json")
CORES = int(os.environ.get("CORES", os.cpu_count() or 8))
ORDER = ["RF", "HGB", "MLP"]
SMOKE = ("--smoke" in sys.argv) or bool(os.environ.get("SMOKE"))
OUTDIR = "shap_out_smoke" if SMOKE else "shap_out"      # smoke kept separate so it can't fake "done" on a real run
N_EXPLAIN = int(os.environ.get("N_EXPLAIN", 40 if SMOKE else 400))
N_BG = int(os.environ.get("N_BG", 30 if SMOKE else 50))
CHUNK_ROWS = int(os.environ.get("CHUNK_ROWS", 8 if SMOKE else 10))   # rows per checkpointed SHAP chunk

MODELS = {
    "16feat": dict(title="16-feat v4 baseline stack", bundle="baseline_stack_v4.pkl"),
    "set1":   dict(title="set1 - 5 mags + 4 colours", base="base_set1_{}.pkl", meta="meta_set1.pkl"),
    "set2":   dict(title="set2 - z mag + z-err + 4 colours + star/galaxy sep + concentration",
                   base="base_set2_{}.pkl", meta="meta_set2.pkl"),
}


def _bucket():
    from google.cloud import storage
    return storage.Client.from_service_account_json(GCS_KEY).bucket(BUCKET)

def _pull(blob, local):
    if not os.path.exists(local):
        _bucket().blob(blob).download_to_filename(local)
    return local


# ---- features by name from the masked catalog ----
def feature_frame(cat, names):
    c = {n: cat[n].mask(cat[n] <= -100).to_numpy("float32") for n in
         ["dered_u","dered_g","dered_r","dered_i","dered_z","modelMagErr_z","psfMag_r","modelMag_r",
          "petroR50_r","petroR90_r","fracDeV_r","expRad_r","deVRad_r","petroRad_r"]}
    def logr(n): v = c[n].copy(); v[v < 0] = np.nan; return np.log1p(v)
    t = {"dered_u":c["dered_u"],"dered_g":c["dered_g"],"dered_r":c["dered_r"],"dered_i":c["dered_i"],"dered_z":c["dered_z"],
         "u-g":np.clip(c["dered_u"]-c["dered_g"],-1,4),"g-r":np.clip(c["dered_g"]-c["dered_r"],-1,4),
         "r-i":np.clip(c["dered_r"]-c["dered_i"],-1,4),"i-z":np.clip(c["dered_i"]-c["dered_z"],-1,4),
         "modelMagErr_z":c["modelMagErr_z"],"sg_sep":c["psfMag_r"]-c["modelMag_r"],"fracDeV_r":c["fracDeV_r"],
         "conc_r":c["petroR90_r"]/np.where(c["petroR50_r"]==0,np.nan,c["petroR50_r"]),
         "log_expRad_r":logr("expRad_r"),"log_deVRad_r":logr("deVRad_r"),"log_petroRad_r":logr("petroRad_r"),
         "log_petroR50_r":np.log1p(np.where(c["petroR50_r"]<0,np.nan,c["petroR50_r"])),
         "log_petroR90_r":np.log1p(np.where(c["petroR90_r"]<0,np.nan,c["petroR90_r"]))}
    X = np.stack([t[n] for n in names], 1).astype("float32"); X[~np.isfinite(X)] = np.nan
    return X

def group(name):
    if name.startswith("dered_"): return "magnitude"
    if name in ("u-g","g-r","r-i","i-z"): return "colour"
    if "Err" in name: return "photometric error"
    if name == "sg_sep": return "star/galaxy sep"
    if name == "conc_r": return "concentration"
    if "Rad" in name or "frac" in name: return "size/shape"
    return "other"


# ---- the parallel task: SHAP for one base on one chunk of explained rows ----
_EXP = {}
def _explainer(path, is_mlp):
    if path not in _EXP:
        a = joblib.load(path); est = a["model"] if isinstance(a, dict) and "model" in a else a
        _EXP[path] = (None if is_mlp else shap.TreeExplainer(est), est)
    return _EXP[path]

def chunk_task(path, is_mlp, Xc, bg, ckpt):
    """SHAP for one small slice of explained rows -> save to ckpt npy. Skips if already computed
    (so a re-run after a time-limit kill reuses every finished chunk)."""
    if os.path.exists(ckpt):
        return
    exp, est = _explainer(path, is_mlp)
    vals = (shap.PermutationExplainer(est.predict, bg)(Xc).values if is_mlp
            else exp.shap_values(Xc, check_additivity=False))
    np.save(ckpt, np.asarray(vals, "float32"))


def save_outputs(key, title, feats, coef, sv_total, xe):
    """Write everything for one model: md table, json numbers, npz raw values, bar + beeswarm png."""
    imp = np.abs(sv_total).mean(0)
    o = np.argsort(imp)[::-1]; tot = imp.sum() + 1e-12
    rows = [(feats[i], group(feats[i]), float(imp[i]), 100*imp[i]/tot) for i in o]
    gt = {}
    for _, g, v, _ in rows: gt[g] = gt.get(g, 0.0) + v
    gs = sorted(gt.items(), key=lambda kv: -kv[1])
    coefd = {m: round(float(c), 4) for m, c in zip(ORDER, coef)}

    # --- markdown ---
    L = [f"# SHAP feature importance - {title}\n",
         f"Stacking model (RF + HGB + MLP -> LinearRegression). SHAP on the **final** z prediction "
         f"({len(xe)} explained galaxies); meta is linear so SHAP(z)=sum w*SHAP(base). Importance = "
         f"mean |SHAP| in redshift units; % = share of total.\n",
         f"**{len(feats)} features.** Meta weights (RF, HGB, MLP): {[round(float(x),3) for x in coef]}.",
         f"Figures: `shap_{key}_bar.png`, `shap_{key}_beeswarm.png` · numbers: `shap_{key}.json` · "
         f"raw SHAP values: `shap_{key}_values.npz`.\n",
         "## Per-feature importance", "| rank | feature | group | mean &#124;SHAP&#124; | % |", "|---|---|---|---|---|"]
    for r, (n, g, v, p) in enumerate(rows, 1): L.append(f"| {r} | `{n}` | {g} | {v:.5f} | {p:.1f}% |")
    L += ["\n## By feature group", "| group | % of total importance |", "|---|---|"]
    for g, v in gs: L.append(f"| {g} | {100*v/tot:.1f}% |")
    L.append(f"\n**Top driver:** `{rows[0][0]}` ({rows[0][3]:.1f}%). Dominant group: **{gs[0][0]}** ({100*gs[0][1]/tot:.0f}%).")
    open(f"{OUTDIR}/shap_{key}.md", "w").write("\n".join(L) + "\n")

    # --- json (numbers, reusable) ---
    json.dump({"model": title, "n_explained": int(len(xe)), "meta_coef": coefd,
               "importance": {feats[i]: round(float(imp[i]), 6) for i in o},
               "pct": {feats[i]: round(100*float(imp[i])/tot, 2) for i in o},
               "group_pct": {g: round(100*v/tot, 2) for g, v in gs}},
              open(f"{OUTDIR}/shap_{key}.json", "w"), indent=2)

    # --- raw SHAP values + feature matrix (full reuse; tiny) ---
    np.savez_compressed(f"{OUTDIR}/shap_{key}_values.npz",
                        shap_values=sv_total.astype("float32"), X=xe.astype("float32"),
                        features=np.array(feats), importance=imp.astype("float32"))

    # --- bar plot (mean |SHAP| importance) ---
    tb = rows[:min(20, len(rows))][::-1]
    plt.figure(figsize=(7, max(3, 0.4*len(tb))))
    plt.barh([r[0] for r in tb], [r[2] for r in tb], color="#4a6fa5")
    plt.xlabel("mean |SHAP|  (redshift units)"); plt.title(f"SHAP importance — {key}"); plt.tight_layout()
    plt.savefig(f"{OUTDIR}/shap_{key}_bar.png", dpi=110); plt.close()

    # --- beeswarm summary (direction + spread per feature) ---
    try:
        plt.figure()
        shap.summary_plot(sv_total, features=xe, feature_names=list(feats), show=False, max_display=min(20, len(feats)))
        plt.title(f"SHAP summary — {key}"); plt.tight_layout()
        plt.savefig(f"{OUTDIR}/shap_{key}_beeswarm.png", dpi=110, bbox_inches="tight"); plt.close()
    except Exception as e:
        print(f"[job] {key} beeswarm skipped: {e}", flush=True)
    return rows[:4]


def main():
    t0 = time.time(); os.makedirs(OUTDIR, exist_ok=True)
    print(f"[job] {'SMOKE ' if SMOKE else ''}cores={CORES} N_EXPLAIN={N_EXPLAIN}", flush=True)
    if not os.path.exists(CATALOG):
        print(f"[job] downloading catalog ...", flush=True); _pull(CAT_BLOB, CATALOG)
    cat = pd.read_parquet(CATALOG)

    # resolve per-model base pkls + meta (extract the 16feat bundle into uniform per-base files)
    resolved = {}
    for key, cfg in MODELS.items():
        if "bundle" in cfg:
            b = joblib.load(_pull(f"{MODELS_BLOB}/{cfg['bundle']}", cfg["bundle"]))
            paths = {}
            for m in ORDER:
                p = f"base_16feat_{m}.pkl"; joblib.dump({"model": b["bases"][m]}, p); paths[m] = p
            resolved[key] = dict(title=cfg["title"], paths=paths, coef=b["meta"].coef_, feats=b["features"])
        else:
            paths = {m: _pull(f"{MODELS_BLOB}/{cfg['base'].format(m)}", cfg["base"].format(m)) for m in ORDER}
            feats = joblib.load(paths["RF"])["features"]
            meta = joblib.load(_pull(f"{MODELS_BLOB}/{cfg['meta']}", cfg["meta"]))["model"]
            resolved[key] = dict(title=cfg["title"], paths=paths, coef=meta.coef_, feats=feats)
        print(f"[job] {key}: {len(resolved[key]['feats'])} feats", flush=True)

    for key, R in resolved.items():
        if os.path.exists(f"{OUTDIR}/shap_{key}.json"):          # model already finished on a prior run
            print(f"[job] {key}: already done -> skip", flush=True); continue
        tk = time.time(); feats = R["feats"]; X = feature_frame(cat, feats)
        keep = np.where(~np.isnan(X).any(1))[0]
        np.random.RandomState(0).shuffle(keep)                  # deterministic -> chunk ckpts align across runs
        bg = X[keep[:N_BG]]; xe = X[keep[N_BG:N_BG+N_EXPLAIN]]
        nch = max(CORES, int(np.ceil(len(xe)/CHUNK_ROWS)))
        slices = [s for s in np.array_split(np.arange(len(xe)), nch) if len(s)]
        svb = {}
        for m in ORDER:                                         # one base at a time -> only its RF in workers
            is_mlp = (m == "MLP")
            Parallel(n_jobs=CORES, backend="loky", verbose=5)(
                delayed(chunk_task)(R["paths"][m], is_mlp, xe[sl], bg, f"{OUTDIR}/_ck_{key}_{m}_{ci}.npy")
                for ci, sl in enumerate(slices))
            sv = np.zeros((len(xe), len(feats)), "float32")
            for ci, sl in enumerate(slices): sv[sl] = np.load(f"{OUTDIR}/_ck_{key}_{m}_{ci}.npy")
            svb[m] = sv
            get_reusable_executor().shutdown(wait=True)         # release this base's RF before the next
            print(f"[job] {key}/{m} chunks done ({time.time()-tk:.0f}s)", flush=True)
        w = dict(zip(ORDER, [float(c) for c in R["coef"]]))
        sv_total = sum(w[m] * svb[m] for m in ORDER)
        top = save_outputs(key, R["title"], feats, R["coef"], sv_total, xe)
        for f in glob.glob(f"{OUTDIR}/_ck_{key}_*.npy"): os.remove(f)   # drop chunk ckpts (model is done)
        if not SMOKE and os.path.exists(GCS_KEY):               # push this finished model to gs immediately
            try:
                b = _bucket()
                for fn in glob.glob(f"{OUTDIR}/shap_{key}*"):
                    b.blob(f"{RESULTS_BLOB}/{os.path.basename(fn)}").upload_from_filename(fn)
                print(f"[job] {key} uploaded -> gs://{BUCKET}/{RESULTS_BLOB}/", flush=True)
            except Exception as e:
                print(f"[job] {key} upload skipped ({e})", flush=True)
        print(f"[job] {key} done ({time.time()-tk:.0f}s) | top: " + ", ".join(f"{n}({p:.0f}%)" for n,_,_,p in top), flush=True)

    tar = "shap_tabular_results_smoke.tar.gz" if SMOKE else "shap_tabular_results.tar.gz"
    with tarfile.open(tar, "w:gz") as t: t.add(OUTDIR, arcname=os.path.basename(OUTDIR))
    if not SMOKE and os.path.exists(GCS_KEY):
        try:
            b = _bucket()
            for fn in sorted(os.listdir(OUTDIR)):                      # md + json + npz + png
                b.blob(f"{RESULTS_BLOB}/{fn}").upload_from_filename(f"{OUTDIR}/{fn}")
            b.blob(f"{RESULTS_BLOB}/{tar}").upload_from_filename(tar)   # the bundle too
            print(f"[job] uploaded {OUTDIR}/* + {tar} -> gs://{BUCKET}/{RESULTS_BLOB}/", flush=True)
        except Exception as e:
            print(f"[job] upload failed ({e}); local {tar} has everything", flush=True)
    print(f"[job] DONE total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
