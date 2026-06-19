#!/usr/bin/env python3
"""Task A (SciServer, SAS mounted) — build the per-frame band-registration offsets.

The sample_v1 cutouts are misregistered: every band was cut at the u-band WCS
pixel, so g/r/i/z are offset from the galaxy by the inter-band astrometry (3-13 px).
To fix sample_v1 -> v3 WITHOUT re-cutting, we only need, per frame, the pixel shift
that re-centres each band. That shift is pure metadata (the per-band WCS), so we read
ONLY the FITS header of each frame (decompress just the first bz2 block, never the
12 MB of pixels) and compute it.

Outputs two CSVs:
  objid_frame.csv : idx, objid, run, camcol, field      (every galaxy -> its frame)
  frame_offset.csv: run, camcol, field,
                    u_dx,u_dy, g_dx,g_dy, r_dx,r_dy, i_dx,i_dy, z_dx,z_dy
                    (per-band shift to APPLY to the v1 cutout, in pixels; u is 0,0;
                     evaluated at the frame centre. scipy uses shift=(dy,dx).)

Only the UNIQUE frames named in objid_frame.csv are touched.

On SciServer (container with SDSS SAS mounted):
    python task_a_offsets.py --sas "/home/idies/workspace/SDSS SAS" --workers 32
"""
import os, bz2, time, argparse, warnings
import numpy as np, pandas as pd
from multiprocessing import Pool
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
warnings.simplefilter("ignore", FITSFixedWarning)

SAS = "/home/idies/workspace/sdss_sas"
BANDS = "ugriz"
RERUN = 301
HEAD_BYTES = 300_000           # first ~300 KB covers the first 900 KB bz2 block -> the header
_SAS = SAS


def frame_path(run, camcol, field, band):
    return (f"{_SAS}/dr17/eboss/photoObj/frames/{RERUN}/{run}/{camcol}/"
            f"frame-{band}-{run:06d}-{camcol}-{field:04d}.fits.bz2")


def _init(sas):
    global _SAS
    _SAS = sas


def read_header(path):
    """Decompress only the first bz2 block of a frame and return its FITS header."""
    with open(path, "rb") as f:
        buf = f.read(HEAD_BYTES)
    raw = bz2.BZ2Decompressor().decompress(buf)
    if raw.find(b"END     ") < 0:                 # header longer than one block (rare): read all
        with open(path, "rb") as f:
            buf = f.read()
        raw = bz2.BZ2Decompressor().decompress(buf)
    end = raw.find(b"END     ")
    hdrlen = ((end // 2880) + 1) * 2880
    return fits.Header.fromstring(raw[:hdrlen].decode("latin1"))


def _shifts_at(wcss, world):
    """Per-band shift to re-centre at sky position `world`: P_u(world) - P_b(world)."""
    pu = np.array(wcss[0].world_to_pixel(world))
    return [tuple(pu - np.array(w.world_to_pixel(world))) for w in wcss]


def frame_offsets(key):
    """key=(run,camcol,field) -> (key, shifts_or_None, corner_residual).
    shifts: list of (dx,dy) per band at the frame centre. corner_residual: max |center-corner|
    shift difference over the 4 corners (bounds the per-frame-constant approximation error)."""
    run, camcol, field = key
    try:
        hdrs = [read_header(frame_path(run, camcol, field, b)) for b in BANDS]
    except Exception:
        return key, None, np.nan
    wcss = [WCS(h) for h in hdrs]
    n1, n2 = int(hdrs[0]["NAXIS1"]), int(hdrs[0]["NAXIS2"])
    cx, cy = (n1 - 1) / 2.0, (n2 - 1) / 2.0
    c0 = wcss[0].pixel_to_world(cx, cy)
    shifts = _shifts_at(wcss, c0)
    # diagnostic (~1% sample): how much do the shifts drift center->corner? This bounds
    # the per-frame-constant approximation error. Sampled to keep world_to_pixel cheap.
    res = np.nan
    if (run + camcol + field) % 97 == 0:
        res = 0.0
        for px, py in [(0, 0), (n1 - 1, 0), (0, n2 - 1), (n1 - 1, n2 - 1)]:
            sc = _shifts_at(wcss, wcss[0].pixel_to_world(px, py))
            res = max(res, max(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(sc, shifts)))
    return key, shifts, res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default="gs://macrocosm-lewagon/data/sample_v1/catalog_v1.parquet")
    ap.add_argument("--key", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "sciserver-uploader.json"))
    ap.add_argument("--sas", default=SAS, help='SDSS SAS mount (Compute Job: "/home/idies/workspace/SDSS SAS")')
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    # catalog (local path or gs://)
    cpath = args.catalog
    if cpath.startswith("gs://"):
        import subprocess, tempfile
        local = os.path.join(tempfile.gettempdir(), "catalog_v1.parquet")
        if not os.path.exists(local):
            subprocess.run(["gcloud", "storage", "cp", cpath, local], check=True)
        cpath = local
    cat = pd.read_parquet(cpath, columns=["idx", "objid", "run", "camcol", "field"])
    cat = cat.sort_values("idx")

    # --- A1: objid -> frame ---
    f1 = os.path.join(args.out_dir, "objid_frame.csv")
    cat.to_csv(f1, index=False)
    print(f"[A1] wrote {f1}  ({len(cat):,} galaxies)", flush=True)

    # --- A2: unique frame -> offset (header-only) ---
    frames = list(map(tuple, cat[["run", "camcol", "field"]].drop_duplicates()
                      .astype(int).itertuples(index=False, name=None)))
    print(f"[A2] {len(frames):,} unique frames, {args.workers} workers, header-only "
          f"(SAS={args.sas})", flush=True)

    rows = []; res_samples = []; done = miss = 0; t0 = last = time.time()
    with Pool(args.workers, initializer=_init, initargs=(args.sas,)) as pool:
        for key, shifts, res in pool.imap_unordered(frame_offsets, frames, chunksize=8):
            run, camcol, field = key
            if shifts is None:
                miss += 1
                rows.append([run, camcol, field] + [np.nan] * 10)
            else:
                rows.append([run, camcol, field] + [v for s in shifts for v in s])
                if not np.isnan(res):
                    res_samples.append(res)
            done += 1
            now = time.time()
            if done == 1 or now - last >= 30:
                last = now; el = now - t0; rate = done / el if el else 0
                eta = (len(frames) - done) / rate / 60 if rate else 0
                print(f"[A2]   {done:,}/{len(frames):,} frames  {el:.0f}s  {rate:.1f} frm/s  "
                      f"~{eta:.0f} min left  ({miss} unreadable)", flush=True)

    cols = ["run", "camcol", "field"] + [f"{b}_{d}" for b in BANDS for d in ("dx", "dy")]
    f2 = os.path.join(args.out_dir, "frame_offset.csv")
    pd.DataFrame(rows, columns=cols).to_csv(f2, index=False)
    rs = np.array(res_samples)
    print(f"[A2] wrote {f2}  ({len(rows):,} frames, {miss} unreadable)", flush=True)
    if len(rs):
        print(f"[A2] per-frame-constant approx error (center->corner shift drift): "
              f"median={np.median(rs):.2f}px  p95={np.percentile(rs,95):.2f}px  max={rs.max():.2f}px",
              flush=True)
    print("[done] next: Task B reads these CSVs + v1 npy from GCS -> shift -> crop24 -> v3", flush=True)


if __name__ == "__main__":
    main()
