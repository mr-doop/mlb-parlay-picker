# parlay/train_model.py
from __future__ import annotations
import argparse, os, json, numpy as np, pandas as pd

from parlay.results_etl import build_results_from_snapshots
from parlay.train_utils import (
    build_training_frame, fit_logit, REFS, DEFAULT
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots_dir", required=True, help="Folder of snapshot ZIPs created by the app")
    ap.add_argument("--results_csv", default="", help="Optional: prebuilt data/pitcher_results.csv. If omitted, ETL will build it from snapshots.")
    ap.add_argument("--out", default="parlay/model_weights.json")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--max_iter", type=int, default=2000)
    args = ap.parse_args()

    # 1) Ensure results CSV exists (or build it automatically)
    if args.results_csv:
        results = pd.read_csv(args.results_csv)
    else:
        results = build_results_from_snapshots(args.snapshots_dir, out_csv="data/pitcher_results.csv", refresh=True)

    # 2) Assemble training set from snapshots + results
    from parlay.train_utils import load_snapshots_props
    snap_props = load_snapshots_props(args.snapshots_dir)
    D, F, y = build_training_frame(snap_props, results)
    if D.empty:
        raise SystemExit("No training rows assembled. Ensure snapshots contain props and ETL produced labels for the same dates/players.")
    # 3) Fit
    w = fit_logit(F, y, l2=args.l2, lr=args.lr, max_iter=args.max_iter)

    # 4) Save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(w, f, indent=2)
    # 5) Simple accuracy
    z = F.values @ np.array([w.get(f,0.0) for f in F.columns]) + w["intercept"]
    p = 1/(1+np.exp(-z))
    acc = float(((p>=0.5).astype(float) == y).mean())
    print(f"Training rows: {len(y)}  |  Accuracy: {acc:.3f}  →  saved {args.out}")

if __name__ == "__main__":
    main()