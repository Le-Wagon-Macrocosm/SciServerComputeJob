#!/usr/bin/env python3
"""Task B — turn sample_v1 into the registered, cropped sample_v3.

For each shard: pull the v1 cutouts from GCS, look up each galaxy's per-frame band
offset (Task A's frame_offset.csv), sub-pixel shift every channel to register the
bands, crop the centre 24x24 (the offsets are <=13 px, well inside the 20 px margin,
so the crop is unaffected), and upload the result to sample_v3.

    python task_b_build_v3.py --shard 0-99 \
        --offsets frame_offset.csv --catalog objid_frame.csv --workers 32
"""
import os, argparse, subprocess, tempfile, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.ndimage import shift as nshift

BANDS = "ugriz"
SIZE, CROP = 64, 24
OFF = (SIZE - CROP) // 2          # 20
SHARD_N = 100
SRC = "gs://macrocosm-lewagon/data/sample_v1"
DST = "gs://macrocosm-lewagon/data/sample_v3"


def parse_shards(spec, n):
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1); out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    s = sorted(set(out))
    assert s and all(0 <= x < n for x in s), f"bad --shard {spec!r}"
    return s


def process_chunk(args):
    """Shift+crop a contiguous slice of the shard's cutouts (memmap, no copy of the whole array)."""
    path, lo, hi, offs = args
    a = np.load(path, mmap_mode="r")
    out = np.zeros((hi - lo, CROP, CROP, len(BANDS)), "float16")
    for k in range(hi - lo):
        img = np.asarray(a[lo + k], np.float32)
        o = offs[k]
        if np.isnan(o).any():                       # frame had no offset -> crop only
            out[k] = img[OFF:OFF + CROP, OFF:OFF + CROP].astype("float16")
            continue
        sh = np.empty_like(img)
        for bi in range(len(BANDS)):
            dx, dy = o[bi]
            sh[:, :, bi] = nshift(img[:, :, bi], (dy, dx), order=1, mode="constant", cval=0)
        out[k] = sh[OFF:OFF + CROP, OFF:OFF + CROP].astype("float16")
    return lo, out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", required=True, help="which shard(s): '5', '0-99', '0,3,5', '0-3,8'")
    ap.add_argument("--offsets", default="frame_offset.csv", help="Task A frame_offset.csv")
    ap.add_argument("--catalog", default="objid_frame.csv",
                    help="Task A objid_frame.csv (idx,objid,run,camcol,field), idx-sorted")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    shards = parse_shards(args.shard, SHARD_N)
    cat = pd.read_csv(args.catalog).sort_values("idx").reset_index(drop=True)
    n = len(cat); block = (n + SHARD_N - 1) // SHARD_N
    off_df = pd.read_csv(args.offsets)
    ocols = [f"{b}_{d}" for b in BANDS for d in ("dx", "dy")]
    print(f"catalog {n:,} galaxies, {len(off_df):,} frame offsets; shards {shards} -> {args.dst}",
          flush=True)

    for sh in shards:
        start, end = sh * block, min((sh + 1) * block, n)
        if start >= end:
            continue
        name = f"images_{start:07d}_{end:07d}.npy"
        if not args.force and subprocess.run(["gsutil", "-q", "stat", f"{args.dst}/{name}"]).returncode == 0:
            print(f"[shard {sh}] {args.dst}/{name} exists — skip", flush=True)
            continue

        t0 = time.time()
        local = os.path.join(tempfile.gettempdir(), f"v1_{sh}.npy")
        subprocess.run(["gsutil", "-q", "cp", f"{args.src}/{name}", local], check=True)

        sub = cat.iloc[start:end]
        m = sub.merge(off_df, on=["run", "camcol", "field"], how="left")
        offarr = m[ocols].to_numpy(np.float32).reshape(-1, len(BANDS), 2)   # (ng,5,2) (dx,dy)
        ng = end - start
        nmiss = int(np.isnan(offarr).any(axis=(1, 2)).sum())

        # parallel shift+crop over row-slices (memmap inside workers, no big pickling)
        step = max(1, (ng + args.workers - 1) // args.workers)
        tasks = [(local, lo, min(lo + step, ng), offarr[lo:min(lo + step, ng)])
                 for lo in range(0, ng, step)]
        out = np.zeros((ng, CROP, CROP, len(BANDS)), "float16")
        with Pool(args.workers) as pool:
            for lo, chunk in pool.imap_unordered(process_chunk, tasks):
                out[lo:lo + len(chunk)] = chunk

        dst_local = os.path.join(tempfile.gettempdir(), name)
        np.save(dst_local, out)
        subprocess.run(["gsutil", "-q", "cp", dst_local, f"{args.dst}/{name}"], check=True)
        os.remove(local); os.remove(dst_local)
        print(f"[shard {sh}] {ng:,} imgs -> {CROP}x{CROP} ({nmiss} no-offset) "
              f"in {time.time()-t0:.0f}s -> {args.dst}/{name}", flush=True)

    print("[done]", flush=True)


if __name__ == "__main__":
    main()
