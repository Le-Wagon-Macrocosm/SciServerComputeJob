#!/usr/bin/env python3
"""Step B — build image stamps on SciServer and push straight to GCS (zero download).

Reads the frozen catalog (scripts/freeze_catalog.py output), cuts a 64x64x5 ugriz
stamp per galaxy from the mounted SDSS DR17 frames, and uploads the result as a
shard `.npy` to GCS. Nothing large is ever downloaded — cutouts are made on the
server and only the final stamps leave SciServer.

Sharding (so the 892k build can run across several runners / Compute Jobs):
    --of K        split the catalog into K CONTIGUOUS blocks (preserves the
                  run/camcol/field ordering, so each shard reuses its own frames)
    --shard i     build block(s) i. Accepts a single '5', a range '0-7', a
                  list '0,3,5', or a mix '0-3,8' (0 <= i < K), built in sequence.
Run several processes / Compute Jobs over disjoint shard sets => no coordination.
Idempotent: a shard whose output already exists on GCS is skipped (rerun freely).

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

# defaults: the project catalog on GCS, and the SA key sitting next to this script
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CATALOG = "gs://macrocosm-lewagon/data/sample_v1/catalog_v1.parquet"
DEFAULT_KEY = os.path.join(HERE, "sciserver-uploader.json")


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
    """End-to-end sanity check: cut the first `--corp` galaxies (default 3) and
    confirm each stamp comes out with the right shape/dtype, finite, and not
    empty. No timing, no upload — just proves the pipeline runs against the SAS
    mount. A MISS or all-zero stamp means the SAS path / frames are wrong."""
    df = load_catalog(args.catalog, args.key)
    df = df.sort_values("idx").reset_index(drop=True)
    sub = df.head(max(args.corp, 1))
    o2id = dict(zip(sub.idx.astype(int), sub.objid.astype(int)))
    print(f"[smoke] cutting {len(sub)} galaxies at {args.size}px from SAS={args.sas} "
          f"(dtype={args.dtype}) — no upload\n")

    groups = [(int(r), int(c), int(f),
               list(zip(g.idx.astype(int), g.ra.astype(float), g.dec.astype(float))))
              for (r, c, f), g in sub.groupby(["run", "camcol", "field"])]
    _init(args.size, args.dtype, args.sas)        # set globals used by cut_group

    ok = 0
    for grp in groups:
        for idx, stamp in cut_group(grp):
            oid = o2id.get(idx)
            if stamp is None:
                print(f"  objid {oid} (idx {idx}): MISS — no/broken frame or off-frame")
                continue
            f32 = stamp.astype("float32")
            finite = bool(np.isfinite(f32).all())
            p99 = [float(np.percentile(f32[:, :, b], 99)) for b in range(len(BANDS))]
            nonempty = any(p > 0 for p in p99)
            good = finite and nonempty
            ok += good
            print(f"  objid {oid} (idx {idx}): shape {stamp.shape} {stamp.dtype}  "
                  f"finite={finite}  p99(ugriz)=[{', '.join(f'{p:.3f}' for p in p99)}]  "
                  f"{'OK' if good else 'BAD (empty/non-finite)'}")

    print(f"\n[smoke] {ok}/{len(sub)} stamps OK — e2e {'PASSED' if ok == len(sub) else 'FAILED'}.")
    if ok < len(sub):
        print(f"[smoke] check --sas ('{args.sas}'); in a Compute Job it's usually "
              f"'/home/idies/workspace/SDSS SAS' (with a space).")
        raise SystemExit(1)


def parse_shards(spec, n_shards):
    """'5' -> [5];  '0-7' -> [0..7];  '0,3,5' -> [0,3,5];  '0-3,8' -> [0,1,2,3,8].
    De-duplicated, sorted, and range-checked against n_shards."""
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    shards = sorted(set(out))
    assert shards, f"no shards parsed from {spec!r}"
    for s in shards:
        assert 0 <= s < n_shards, f"shard {s} out of range [0, {n_shards})"
    return shards


def build_shard(args, shard, bucket, df, n):
    """Cut + upload one shard's contiguous block. Skips if the output exists
    (unless --force). Returns silently when there's nothing to do."""
    block = (n + args.n_shards - 1) // args.n_shards
    start, end = shard * block, min((shard + 1) * block, n)
    if start >= end:
        print(f"[shard {shard}] empty (n={n}, of={args.n_shards}) — nothing to do")
        return
    blob_name = f"{args.prefix}/images_{start:07d}_{end:07d}.npy"

    if not args.force and bucket.blob(blob_name).exists():
        print(f"[shard {shard}] gs://{args.bucket}/{blob_name} exists — skip "
              f"(use --force to rebuild)")
        return

    sub = df.iloc[start:end]
    groups = [(int(r), int(c), int(f),
               list(zip(g.idx.astype(int), g.ra.astype(float), g.dec.astype(float))))
              for (r, c, f), g in sub.groupby(["run", "camcol", "field"])]
    print(f"[shard {shard}] rows {start:,}..{end:,} ({end - start:,} galaxies), "
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
            # heartbeat: first 10 frames individually (catch a slow/stalled mount
            # immediately), then every 50. flush so it shows up in the job log live.
            if done <= 10 or done % 50 == 0:
                el = time.time() - t0
                eta = (len(groups) - done) / (done / el) / 60 if el and done else 0
                print(f"[shard {shard}]   {done:,}/{len(groups):,} frames  "
                      f"{el:.0f}s  {done / el:.2f} frames/s  ~{eta:.0f} min left",
                      flush=True)
    el = time.time() - t0
    ng = end - start
    print(f"[shard {shard}] cut {ng:,} galaxies from {len(groups):,} frames "
          f"in {el:.0f}s ({ng / el:.1f} gal/s, {len(groups) / el:.1f} frames/s)")
    if miss:
        print(f"[shard {shard}] WARNING: {miss} galaxies had a missing/broken "
              f"frame -> zero-filled stamps")
    # safety: a wrong SAS mount makes EVERY frame open fail -> all-zero garbage.
    # Refuse to upload that; fail loudly so the path gets fixed instead.
    if miss > 0.2 * ng:
        raise SystemExit(
            f"[shard {shard}] ABORTING — {miss}/{ng} galaxies had missing frames. "
            f"The SAS mount '{args.sas}' is almost certainly wrong (no frames found there). "
            f"Not uploading all-zero data. Verify with --smoke, fix --sas, rerun.")

    local = os.path.join(args.tmp, os.path.basename(blob_name))
    np.save(local, out)
    gb = out.nbytes / 1e9
    print(f"[shard {shard}] uploading {gb:.2f} GB -> gs://{args.bucket}/{blob_name}")
    bucket.blob(blob_name).upload_from_filename(local)
    os.remove(local)
    print(f"[shard {shard}] done.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default=DEFAULT_CATALOG,
                    help=f"catalog_v1.parquet (local path or gs://...; default {DEFAULT_CATALOG})")
    ap.add_argument("--key", default=DEFAULT_KEY,
                    help=f"sciserver-uploader service-account JSON for GCS upload "
                         f"(default: sciserver-uploader.json next to this script)")
    ap.add_argument("--of", type=int, default=64, dest="n_shards",
                    help="total number of contiguous shards (default 64)")
    ap.add_argument("--shard", default=None,
                    help="which shard(s) (0..of-1): a single '5', a range '0-7', a list "
                         "'0,3,5', or a mix '0-3,8,10-11'. Not needed with --smoke")
    ap.add_argument("--smoke", action="store_true",
                    help="e2e sanity check: cut a few galaxies, verify the stamps, no upload")
    ap.add_argument("--corp", type=int, default=3,
                    help="--smoke only: how many galaxies to cut for the check (default 3)")
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
    shards = parse_shards(args.shard, args.n_shards)

    # NB: workers get the SAS mount via the Pool initializer (_init); main
    # itself never calls frame_path, so no module-global mutation is needed here.
    from google.cloud import storage
    client = storage.Client.from_service_account_json(args.key)
    bucket = client.bucket(args.bucket)

    df = load_catalog(args.catalog, args.key)
    df = df.sort_values("idx").reset_index(drop=True)
    n = len(df)
    print(f"catalog has {n:,} galaxies; building shard(s) {shards} of {args.n_shards} "
          f"-> gs://{args.bucket}/{args.prefix}/")

    for s in shards:
        build_shard(args, s, bucket, df, n)


if __name__ == "__main__":
    main()
