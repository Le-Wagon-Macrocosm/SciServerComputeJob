#!/usr/bin/env python3
"""prepare_data_v4 — cut the MISSING sample_v4 cutouts and merge them into GCS.

Background: catalog_v4 is the quality-cleaned 600k set. After the health cuts removed
~55k contaminated galaxies, it was refilled to 600k with NEW healthy galaxies, and the
surviving images were repacked so catalog_v4's idx is contiguous 0..599999:
    idx     0 .. 545016   surviving healthy  -> images already in sample_v4 shards 0..90
    idx 545017 .. 599999   NEW galaxies       -> NO image yet (this script cuts them)
The new ones live in sample_v4 shards 90 (tail rows 5017..5999) and 91..99.

Like sample_v2's prepare_data, EACH BAND is cut with its OWN WCS — SDSS drift-scans, so
the 5 filters image the sky at different times and have per-band astrometry 3-13 px apart.
Cutting every band at one band's WCS (the old sample_v1 bug) offsets g/r/i/z and corrupts
colour. We cut 64x64 per band, NO crop — sample_v4 is the correctly-registered 64px set
(repacked from sample_v2), so the new cutouts must also be 64x64.

Per target shard the script STARTS FROM the existing sample_v4 shard on GCS (preserving the
already-correct healthy rows, repacked from sample_v2) and overwrites ONLY the new-galaxy
rows — so shard 90's 5017 healthy rows are kept and only its 983 new rows get filled.
Shards 91..99 start from zeros.

Idempotent-ish: rerunning re-cuts and re-uploads the listed shards (always overwrites).

On SciServer (container with the SDSS SAS volume mounted):
    !pip install --user astropy google-cloud-storage gcsfs pyarrow
    python prepare_data_v4.py --shard 90-99 --workers 32 \
        --catalog gs://macrocosm-lewagon/data/sample_v4/new_objids_v4.parquet \
        --key sciserver-uploader.json --sas "/home/idies/workspace/SDSS SAS"
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

# Interactive-container mount. In a Compute Job the volume mounts under its display
# name, usually "/home/idies/workspace/SDSS SAS" (with a space) — override with --sas.
SAS = "/home/idies/workspace/sdss_sas"
BANDS = "ugriz"
RERUN = 301
SHARD_N = 6000          # rows per shard (global 6000 grid, same as sample_v3/v4)
NTOTAL = 600000         # catalog_v4 length (clamps the last shard's end)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CATALOG = "gs://macrocosm-lewagon/data/sample_v4/new_objids_v4.parquet"
DEFAULT_KEY = os.path.join(HERE, "sciserver-uploader.json")
DEFAULT_BUCKET = "macrocosm-lewagon"
DEFAULT_PREFIX = "data/sample_v4"


def frame_path(run, camcol, field, band):
    return (f"{SAS}/dr17/eboss/photoObj/frames/{RERUN}/{run}/{camcol}/"
            f"frame-{band}-{run:06d}-{camcol}-{field:04d}.fits.bz2")


# ---- per-frame-group worker (subprocess; globals set once via Pool initializer) ----
_SIZE = 64       # cut size per band
_CROP = 0        # 0 (or >= size) = no crop, store full _SIZE (sample_v4 = 64px)
_DTYPE = "float16"


def out_size(size, crop):
    """Stored edge length: crop if it's a real centre-crop, else the full cut size."""
    return crop if 0 < crop < size else size


def _init(size, crop, dtype, sas):
    global _SIZE, _CROP, _DTYPE, SAS
    _SIZE, _CROP, _DTYPE, SAS = size, crop, dtype, sas


def cut_group(group):
    """group = (run, camcol, field, [(idx, ra, dec), ...]).
    Open the 5 band frames once, cut every galaxy with its OWN-band WCS, centre-crop.
    Returns [(idx, stamp_or_None)]; a missing/broken frame -> None for the whole group."""
    run, camcol, field, gals = group
    try:
        data, wcss = [], []
        for b in BANDS:
            with fits.open(frame_path(run, camcol, field, b)) as hdu:
                data.append(hdu[0].data)
                wcss.append(WCS(hdu[0].header))   # each band: its own astrometric solution
    except Exception:
        return [(idx, None) for idx, _, _ in gals]

    off = (_SIZE - _CROP) // 2 if (0 < _CROP < _SIZE) else 0
    out = []
    for idx, ra, dec in gals:
        try:
            pos = SkyCoord(ra, dec, unit="deg")
            chans = [Cutout2D(d, pos, (_SIZE, _SIZE), wcs=w, mode="partial", fill_value=0).data
                     for d, w in zip(data, wcss)]      # each centred on (ra,dec) in its band -> registered
            stamp = np.stack(chans, -1)                # (SIZE, SIZE, 5)
            if off:
                stamp = stamp[off:off + _CROP, off:off + _CROP, :]   # centre-crop to CROP
            out.append((idx, stamp.astype(_DTYPE)))
        except Exception:
            out.append((idx, None))                    # off-frame edge case -> zero-fill, counted as miss
    return out


# ---- GCS helpers (gsutil/gcloud absent on SciServer -> google-cloud-storage lib) ----
def _client(key):
    from google.cloud import storage
    return storage.Client.from_service_account_json(key)


def load_catalog(path, key):
    """Local path -> read directly. gs:// -> download once (cached in /tmp), read."""
    if path.startswith("gs://"):
        bucket_name, blob_name = path[5:].split("/", 1)
        local = "/tmp/" + os.path.basename(blob_name)
        if not os.path.exists(local):
            _client(key).bucket(bucket_name).blob(blob_name).download_to_filename(local)
        path = local
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def download_shard(bucket, blob_name, expect_rows, osz, dtype):
    """Existing sample_v4 shard as the base array (preserve healthy rows), or zeros if absent."""
    blob = bucket.blob(blob_name)
    if blob.exists():
        local = f"/tmp/{os.path.basename(blob_name)}"
        blob.download_to_filename(local)
        arr = np.load(local).astype(dtype)
        os.remove(local)
        if arr.shape == (expect_rows, osz, osz, len(BANDS)):
            return arr
        print(f"  WARNING: existing {blob_name} shape {arr.shape} != "
              f"{(expect_rows, osz, osz, len(BANDS))} -> starting from zeros")
    return np.zeros((expect_rows, osz, osz, len(BANDS)), dtype)


def parse_shards(spec):
    """'95' -> [95];  '90-99' -> [90..99];  '90,95' -> [90,95];  '90-92,99' -> [90,91,92,99]."""
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
    return shards


def build_shard(args, shard, bucket, df):
    """Cut the NEW galaxies that fall in this shard, merge over the existing shard, upload."""
    lo, hi = shard * SHARD_N, min((shard + 1) * SHARD_N, NTOTAL)
    sub = df[(df["idx"] >= lo) & (df["idx"] < hi)]
    if len(sub) == 0:
        print(f"[shard {shard}] no new galaxies in idx [{lo},{hi}) — nothing to do")
        return
    blob_name = f"{args.prefix}/images_{lo:07d}_{hi:07d}.npy"
    osz = out_size(args.size, args.crop)

    # base = existing shard (keeps healthy rows already there); new rows overwritten below
    out = download_shard(bucket, blob_name, hi - lo, osz, args.dtype)

    groups = [(int(r), int(c), int(f),
               list(zip(g.idx.astype(int), g.ra.astype(float), g.dec.astype(float))))
              for (r, c, f), g in sub.groupby(["run", "camcol", "field"])]
    crop_note = f"cut {args.size}px (no crop)" if osz == args.size else f"cut {args.size}px -> crop {osz}px"
    print(f"[shard {shard}] idx [{lo},{hi}): filling {len(sub):,} new galaxies "
          f"from {len(groups):,} frames, {args.workers} workers, "
          f"{crop_note} {args.dtype}", flush=True)

    done = miss = 0
    t0 = last = time.time()
    with Pool(args.workers, initializer=_init,
              initargs=(args.size, args.crop, args.dtype, args.sas)) as pool:
        for res in pool.imap_unordered(cut_group, groups, chunksize=1):
            for idx, stamp in res:
                if stamp is None:
                    miss += 1
                    continue
                out[idx - lo] = stamp        # overwrite the placeholder/zero row
            done += 1
            now = time.time()
            if done == 1 or now - last >= 30:
                last = now
                el = now - t0
                rate = done / el if el else 0
                eta = (len(groups) - done) / rate / 60 if rate else 0
                print(f"[shard {shard}]   {done:,}/{len(groups):,} frames  {el:.0f}s  "
                      f"{rate:.2f} frames/s  ~{eta:.0f} min left", flush=True)

    ng = len(sub)
    el = time.time() - t0
    print(f"[shard {shard}] cut {ng - miss:,}/{ng:,} new galaxies in {el:.0f}s "
          f"({ng / el:.1f} gal/s)" + (f"  [{miss} miss -> zero]" if miss else ""))
    # a wrong SAS mount makes every frame open fail -> all-zero garbage; refuse to upload that.
    if miss > 0.2 * ng:
        raise SystemExit(
            f"[shard {shard}] ABORTING — {miss}/{ng} new galaxies had missing frames. "
            f"The SAS mount '{args.sas}' is almost certainly wrong. Not uploading. "
            f"Verify with --smoke, fix --sas, rerun.")

    local = os.path.join(args.tmp, os.path.basename(blob_name))
    np.save(local, out)
    gb = out.nbytes / 1e9
    print(f"[shard {shard}] uploading {gb:.2f} GB -> gs://{args.bucket}/{blob_name}")
    bucket.blob(blob_name).upload_from_filename(local)
    os.remove(local)
    print(f"[shard {shard}] done.")


def smoke(args):
    """E2e sanity check: cut the first `--corp` new galaxies, verify shape/finite/non-empty.
    No download, no upload — just proves the per-band cut runs against the SAS mount."""
    df = load_catalog(args.catalog, args.key).sort_values("idx").reset_index(drop=True)
    sub = df.head(max(args.corp, 1))
    o2id = dict(zip(sub.idx.astype(int), sub.objid.astype(int))) if "objid" in sub else {}
    osz = out_size(args.size, args.crop)
    cn = f"cut {args.size}px (no crop)" if osz == args.size else f"cut {args.size}px -> crop {osz}px"
    print(f"[smoke] cutting {len(sub)} new galaxies ({cn}) from SAS={args.sas} — no upload\n")
    groups = [(int(r), int(c), int(f),
               list(zip(g.idx.astype(int), g.ra.astype(float), g.dec.astype(float))))
              for (r, c, f), g in sub.groupby(["run", "camcol", "field"])]
    _init(args.size, args.crop, args.dtype, args.sas)
    ok = 0
    for grp in groups:
        for idx, stamp in cut_group(grp):
            oid = o2id.get(idx, "?")
            if stamp is None:
                print(f"  objid {oid} (idx {idx}): MISS — no/broken frame or off-frame")
                continue
            f32 = stamp.astype("float32")
            finite = bool(np.isfinite(f32).all())
            p99 = [float(np.percentile(f32[:, :, b], 99)) for b in range(len(BANDS))]
            good = finite and any(p > 0 for p in p99)
            ok += good
            print(f"  objid {oid} (idx {idx}): shape {stamp.shape} {stamp.dtype}  "
                  f"finite={finite}  p99(ugriz)=[{', '.join(f'{p:.3f}' for p in p99)}]  "
                  f"{'OK' if good else 'BAD'}")
    print(f"\n[smoke] {ok}/{len(sub)} stamps OK — e2e {'PASSED' if ok == len(sub) else 'FAILED'}.")
    if ok < len(sub):
        print(f"[smoke] check --sas ('{args.sas}'); in a Compute Job it's usually "
              f"'/home/idies/workspace/SDSS SAS' (with a space).")
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default=DEFAULT_CATALOG,
                    help=f"new_objids_v4 (idx 545017..599999; parquet/csv, local or gs://; "
                         f"default {DEFAULT_CATALOG})")
    ap.add_argument("--key", default=DEFAULT_KEY,
                    help="sciserver-uploader service-account JSON (default: next to this script)")
    ap.add_argument("--shard", default=None,
                    help="which sample_v4 shard(s) to fill (90..99): '95', '90-99', '90,95', '90-92,99'")
    ap.add_argument("--smoke", action="store_true",
                    help="e2e sanity check: cut a few new galaxies, verify, no download/upload")
    ap.add_argument("--corp", type=int, default=3, help="--smoke only: how many to cut (default 3)")
    ap.add_argument("--size", type=int, default=64, help="per-band cut size px (default 64)")
    ap.add_argument("--crop", type=int, default=0,
                    help="centre-crop stored size px (default 0 = NO crop, store full --size=64 "
                         "to match sample_v2/sample_v4; set e.g. 24 to crop)")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--sas", default=SAS,
                    help=f"SDSS SAS mount (default {SAS}; Compute Job: '/home/idies/workspace/SDSS SAS')")
    ap.add_argument("--workers", type=int, default=8, help="parallel processes")
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--tmp", default="/tmp")
    args = ap.parse_args()

    if args.smoke:
        smoke(args)
        return

    assert args.shard is not None, "--shard is required (or use --smoke)"
    shards = parse_shards(args.shard)

    df = load_catalog(args.catalog, args.key)
    assert "idx" in df.columns, "catalog needs an 'idx' column (the contiguous catalog_v4 idx)"
    df = df.sort_values("idx").reset_index(drop=True)
    print(f"new-objids catalog: {len(df):,} galaxies, idx {int(df.idx.min())}..{int(df.idx.max())}; "
          f"filling shard(s) {shards} -> gs://{args.bucket}/{args.prefix}/", flush=True)

    bucket = _client(args.key).bucket(args.bucket)
    for s in shards:
        build_shard(args, s, bucket, df)


if __name__ == "__main__":
    main()
