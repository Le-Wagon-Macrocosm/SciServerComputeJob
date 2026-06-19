#!/usr/bin/env python3
"""Merge Task A's per-fragment frame_offset_<NN>.csv files into one frame_offset.csv.

    python merge_offsets.py                       # merges ./frame_offset_*.csv
    python merge_offsets.py --in-dir . --out frame_offset.csv
"""
import os, glob, argparse
import numpy as np, pandas as pd

KEY = ["run", "camcol", "field"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default=".")
    ap.add_argument("--out", default="frame_offset.csv")
    ap.add_argument("--catalog", default="objid_frame.csv",
                    help="objid_frame.csv — to cross-check that every needed frame got an offset")
    args = ap.parse_args()

    parts = sorted(glob.glob(os.path.join(args.in_dir, "frame_offset_*.csv")))
    assert parts, f"no frame_offset_*.csv found in {args.in_dir}"
    print(f"merging {len(parts)} fragments:")
    dfs = []
    for p in parts:
        d = pd.read_csv(p)
        print(f"  {os.path.basename(p)}: {len(d):,} frames")
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(KEY).sort_values(KEY).reset_index(drop=True)
    if before != len(df):
        print(f"  dropped {before - len(df):,} duplicate frame rows (overlapping fragments)")

    miss = int(df.drop(columns=KEY).isna().any(axis=1).sum())
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df):,} unique frames ({miss} unreadable/NaN)")

    # cross-check: does every frame referenced by a galaxy have an offset row?
    cp = os.path.join(args.in_dir, args.catalog)
    if os.path.exists(cp):
        need = pd.read_csv(cp, usecols=KEY).drop_duplicates()
        got = df[KEY]
        missing = need.merge(got.assign(_ok=1), on=KEY, how="left")["_ok"].isna().sum()
        status = "OK — all frames covered" if missing == 0 else f"WARNING: {missing:,} frames MISSING"
        print(f"coverage vs {args.catalog}: need {len(need):,}, have {len(got):,}  -> {status}")
        if missing:
            print("  (a fragment job probably didn't finish — rerun it, then merge again)")


if __name__ == "__main__":
    main()
