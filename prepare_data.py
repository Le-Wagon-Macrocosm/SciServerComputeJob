#!/usr/bin/env python3
"""Step B — build image stamps on SciServer and push straight to GCS (zero download).

Reads the frozen catalog (scripts/freeze_catalog.py output), cuts a 64x64x5 ugriz
stamp per galaxy from the mounted SDSS DR17 frames, and uploads the result as a
shard `.npy` to GCS. Nothing large is ever downloaded — cutouts are made on the
server and only the final stamps leave SciServer.

Sharding (so the 892k build can run across several runners / Compute Jobs):
    --of K        split the catalog into K CONTIGUOUS blocks (preserves the
                  run/camcol/field ordering, so each shard reuses its own frames)
    --shard i     build block i only  (0 <= i < K)
Run one process per `i`. Disjoint shards => no coordination. Idempotent: a shard
whose output already exists on GCS is skipped (rerun a failed `i` freely).

Inside one shard, work is parallelised over (run, camcol, field) groups: each
group opens its 5 band frames once and cuts every galaxy that lands on them.

On SciServer (persistent volume, container with SDSS SAS mounted):
    !pip install --user astropy google-cloud-storage gcsfs pyarrow
    python scripts/prepare_data.py \
        --catalog gs://macrocosm-lewagon/data/sample_v1/catalog_v1.parquet \
        --key /home/idies/workspace/Storage/<you>/persistent/sciserver-uploader.json \
        --of 64 --shard 0
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from multiprocessing import Pool

from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord

warnings.simplefilter("ignore", FITSFixedWarning)

# Default is the INTERACTIVE-container mount. In a Compute Job the same data
# volume mounts under its display name, e.g. "/home/idies/workspace/SDSS SAS"
# (with a space) — override with --sas in that case.
SAS = "/home/idies/workspace/sdss_sas"
BANDS = "ugriz"
RERUN = 301


def frame_path(run, camcol, field, band):
    return (f"{SAS}/dr17/eboss/photoObj/frames/{RERUN}/{run}/{camcol}/"
            f"frame-{band}-{run:06d}-{camcol}-{field:04d}.fits.bz2")


# ---- per-frame-group worker (runs in a subprocess) --------------------------
# Globals set once per worker via Pool initializer (avoids pickling them per task).
_SIZE = 64
_DTYPE = "float16"


def _init(size, dtype, sas):
    global _SIZE, _DTYPE, SAS
    _SIZE, _DTYPE, SAS = size, dtype, sas


def cut_group(group):
    """group = (run, camcol, field, [(idx, ra, dec), ...]).
    Open the 5 band frames once, cut every galaxy. Returns [(idx, stamp_or_None)].
    A missing/broken frame -> None stamps for the whole group (logged by caller)."""
    run, camcol, field, gals = group
    try:
        # Each band frame has its OWN astrometric solution (drift-scan: the 5
        # filters image the sky at different times -> per-band WCS). Keep every
        # band's WCS and cut each band with its own, so all 5 channels are
        # centred on (ra,dec) and land registered. Using one band's WCS for all
        # (the old bug) offset g/r/i/z by the inter-band astrometry (3-11 px).
        data, wcss = [], []
        for b in BANDS:
            with fits.open(frame_path(run, camcol, field, b)) as hdu:
                data.append(hdu[0].data)
                wcss.append(WCS(hdu[0].header))
    except Exception:
        return [(idx, None) for idx, _, _ in gals]

    out = []
    for idx, ra, dec in gals:
        try:
            pos = SkyCoord(ra, dec, unit="deg")
            chans = [Cutout2D(d, pos, (_SIZE, _SIZE), wcs=w,
                              mode="partial", fill_value=0).data
                     for d, w in zip(data, wcss)]
            out.append((idx, np.stack(chans, -1).astype(_DTYPE)))
        except Exception:
            # galaxy lands off its listed frame (rare SDSS edge case) -> zero-fill
            # this one (counted as `miss`); one bad object must not kill the job.
            out.append((idx, None))
    return out


# ---- catalog loading --------------------------------------------------------
def load_catalog(path, key):
    """Local path -> read directly. gs:// -> download once with the SA key, read."""
    if path.startswith("gs://"):
        from google.cloud import storage
        bucket_name, blob_name = path[5:].split("/", 1)
        local = "/tmp/" + os.path.basename(blob_name)
        if not os.path.exists(local):           # cache: download once, reuse across shards
            client = storage.Client.from_service_account_json(key)
            client.bucket(bucket_name).blob(blob_name).download_to_filename(local)
        path = local
    return pd.read_parquet(path)


def smoke(args):
    """Benchmark per-frame cutting cost at stamp sizes 16/24/32/64 on the first
    `--corp` frames. Single-process (so the number IS the per-frame cost); the
    real build divides wall-clock by --workers. Cuts but never uploads."""
    SIZES = (16, 24, 32, 64)
    BYTES = {"float16": 2, "float32": 4}[args.dtype]

    df = load_catalog(args.catalog, args.key)
    df = df.sort_values("idx").reset_index(drop=True)
    n_gal = len(df)
    n_frames_total = df.groupby(["run", "camcol", "field"]).ngroups
    print(f"[smoke] catalog: {n_gal:,} galaxies in {n_frames_total:,} frames; "
          f"SAS={args.sas}  dtype={args.dtype}  workers(real build)={args.workers}")

    # first --corp frames (contiguous in build order = realistic frame locality)
    groups = []
    for (r, c, f), g in df.groupby(["run", "camcol", "field"], sort=False):
        groups.append((int(r), int(c), int(f),
                       list(zip(g.idx.astype(int), g.ra.astype(float), g.dec.astype(float)))))
        if len(groups) >= args.corp:
            break
    n_smoke_gal = sum(len(g[3]) for g in groups)
    print(f"[smoke] timing {len(groups)} frames ({n_smoke_gal:,} galaxies) at each size\n")
    print(f"  {'size':>4} | {'s/frame':>8} | {'frames/s':>8} | {'gal/s':>7} | {'miss':>5} | "
          f"{'full build (1 proc)':>19} | {f'wall @{args.workers}w':>11} | {'dataset GB':>10}")
    print("  " + "-" * 96)

    for size in SIZES:
        _init(size, args.dtype, args.sas)        # set globals used by cut_group
        miss = ncut = 0
        t0 = time.time()
        for grp in groups:
            for _idx, stamp in cut_group(grp):
                if stamp is None:
                    miss += 1
                else:
                    ncut += 1
        el = time.time() - t0
        spf = el / max(len(groups), 1)
        fps = len(groups) / el if el else 0.0
        gps = n_smoke_gal / el if el else 0.0
        full_1proc_h = n_frames_total * spf / 3600
        wall_h = full_1proc_h / max(args.workers, 1)
        ds_gb = n_gal * size * size * len(BANDS) * BYTES / 1e9
        print(f"  {size:>4} | {spf:>8.3f} | {fps:>8.1f} | {gps:>7.1f} | {miss:>5} | "
              f"{full_1proc_h:>16.1f} h | {wall_h:>9.1f} h | {ds_gb:>8.1f} GB")

    if miss == n_smoke_gal:                       # every cut missed -> bad SAS mount
        print(f"\n[smoke] WARNING: ALL stamps missed — the SAS mount '{args.sas}' is almost "
              f"certainly wrong (no frames found there). Fix --sas before a real run.")
    print(f"\n[smoke] full build = {n_frames_total:,} frames; 'wall @{args.workers}w' assumes "
          f"perfect {args.workers}-way scaling (real is a bit less). No data uploaded.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", required=True,
                    help="catalog_v1.parquet (local path or gs://...)")
    ap.add_argument("--key", required=True,
                    help="sciserver-uploader service-account JSON (for GCS upload)")
    ap.add_argument("--of", type=int, default=64, dest="n_shards",
                    help="total number of contiguous shards (default 64)")
    ap.add_argument("--shard", type=int, default=None, help="which shard (0..of-1); not needed with --smoke")
    ap.add_argument("--smoke", action="store_true",
                    help="benchmark cutting speed at sizes 16/24/32/64 on a few frames, no upload")
    ap.add_argument("--corp", type=int, default=20,
                    help="--smoke only: how many frames to sample for the timing (default 20)")
    ap.add_argument("--size", type=int, default=64, help="stamp size px (default 64)")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"],
                    help="stored pixel dtype (default float16, half the bytes)")
    ap.add_argument("--sas", default=SAS,
                    help=f"SDSS SAS mount point (default {SAS}; in a Compute Job "
                         "this is usually '/home/idies/workspace/SDSS SAS')")
    ap.add_argument("--workers", type=int, default=4, help="parallel processes")
    ap.add_argument("--bucket", default="macrocosm-lewagon")
    ap.add_argument("--prefix", default="data/sample_v2")
    ap.add_argument("--tmp", default="/tmp", help="where to write the shard before upload")
    ap.add_argument("--force", action="store_true", help="rebuild even if output exists")
    args = ap.parse_args()

    if args.smoke:
        smoke(args)
        return

    assert args.shard is not None, "--shard is required (or use --smoke)"
    assert 0 <= args.shard < args.n_shards, "shard out of range"

    # NB: workers get the SAS mount via the Pool initializer (_init); main
    # itself never calls frame_path, so no module-global mutation is needed here.
    from google.cloud import storage
    client = storage.Client.from_service_account_json(args.key)
    bucket = client.bucket(args.bucket)

    df = load_catalog(args.catalog, args.key)
    df = df.sort_values("idx").reset_index(drop=True)
    n = len(df)
    print(f"[shard {args.shard}] catalog has {n:,} galaxies total")

    # contiguous block for this shard
    size = (n + args.n_shards - 1) // args.n_shards
    start, end = args.shard * size, min((args.shard + 1) * size, n)
    if start >= end:
        print(f"[shard {args.shard}] empty (n={n}, of={args.n_shards}) — nothing to do")
        return
    blob_name = f"{args.prefix}/images_{start:07d}_{end:07d}.npy"

    if not args.force and bucket.blob(blob_name).exists():
        print(f"[shard {args.shard}] gs://{args.bucket}/{blob_name} exists — skip "
              f"(use --force to rebuild)")
        return

    sub = df.iloc[start:end]
    groups = [(int(r), int(c), int(f),
               list(zip(g.idx.astype(int), g.ra.astype(float), g.dec.astype(float))))
              for (r, c, f), g in sub.groupby(["run", "camcol", "field"])]
    print(f"[shard {args.shard}] rows {start:,}..{end:,} ({end - start:,} galaxies), "
          f"{len(groups):,} frames, {args.workers} workers, {args.size}px {args.dtype}")

    out = np.zeros((end - start, args.size, args.size, len(BANDS)), dtype=args.dtype)
    done = miss = 0
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(args.size, args.dtype, args.sas)) as pool:
        for res in pool.imap_unordered(cut_group, groups, chunksize=1):
            for idx, stamp in res:
                if stamp is None:
                    miss += 1
                    continue
                out[idx - start] = stamp
            done += 1
            if done % 200 == 0:
                el = time.time() - t0
                print(f"[shard {args.shard}]   {done:,}/{len(groups):,} frames "
                      f"({el:.0f}s, {done / el:.1f} frames/s)")
    el = time.time() - t0
    ng = end - start
    print(f"[shard {args.shard}] cut {ng:,} galaxies from {len(groups):,} frames "
          f"in {el:.0f}s ({ng / el:.1f} gal/s, {len(groups) / el:.1f} frames/s)")
    # full-build projection at this rate, on this one container:
    print(f"[shard {args.shard}] => full {n:,} galaxies ≈ "
          f"{n / (ng / el) / 3600:.1f} h on one container at this rate")
    if miss:
        print(f"[shard {args.shard}] WARNING: {miss} galaxies had a missing/broken "
              f"frame -> zero-filled stamps")
    # safety: a wrong SAS mount makes EVERY frame open fail -> all-zero garbage.
    # Refuse to upload that; fail loudly so the path gets fixed instead.
    if miss > 0.2 * ng:
        raise SystemExit(
            f"[shard {args.shard}] ABORTING — {miss}/{ng} galaxies had missing frames. "
            f"The SAS mount '{args.sas}' is almost certainly wrong (no frames found there). "
            f"Not uploading all-zero data. Probe with scripts/check_sas.py, fix --sas, rerun.")

    local = os.path.join(args.tmp, os.path.basename(blob_name))
    np.save(local, out)
    gb = out.nbytes / 1e9
    print(f"[shard {args.shard}] uploading {gb:.2f} GB -> gs://{args.bucket}/{blob_name}")
    bucket.blob(blob_name).upload_from_filename(local)
    os.remove(local)
    print(f"[shard {args.shard}] done.")


if __name__ == "__main__":
    main()
