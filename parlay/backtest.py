# parlay/backtest.py
"""
Backtest the model on a folder of snapshots.

USAGE
-----
python -m parlay.backtest \
  --snapshots_dir snapshots/ \
  --weights parlay/model_weights.json \
  --results_csv data/pitcher_results.csv \
  --out_dir backtests/

Outputs:
  backtests/summary.csv
  backtests/by_market.csv
"""
from __future__ import annotations
import os, json, argparse, numpy as np, pandas as pd

from parlay.train_utils import load_snapshots_props, build_training_frame
from parlay.results_etl import build_results_from_snapshots

def _read_json(p): 
    with open(p,"r") as f: return json.load(f)

def _american_to_decimal(a):
    try:
        a = float(a)
        return 1 + (a/100.0) if a >= 0 else 1 + (100.0/abs(a))
    except Exception: return np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots_dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--results_csv", default="")
    ap.add_argument("--out_dir", default="backtests/")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.results_csv:
        results = pd.read_csv(args.results_csv)
    else:
        results = build_results_from_snapshots(args.snapshots_dir, out_csv="data/pitcher_results.csv", refresh=True)

    props = load_snapshots_props(args.snapshots_dir)
    D, F, y = build_training_frame(props, results)
    if D.empty:
        raise SystemExit("No rows aligned between snapshots and results. Rebuild ETL or adjust date window.")

    W = _read_json(args.weights)
    # score
    cols = list(F.columns)
    vec = np.array([W.get(c, 0.0) for c in cols], dtype="float64")
    z = F.values @ vec + float(W.get("intercept", 0.0))
    p = 1/(1+np.exp(-z))
    D = D.copy()
    D["q_model"] = p
    D["decimal_odds"] = D["american_odds"].map(_american_to_decimal)
    D["hit"] = y

    # ROI if we flat-bet everything with q>=t
    def roi_at(t=0.60):
        pick = D[D["q_model"] >= t]
        if pick.empty: return np.nan, 0
        # EV (expected) and realized (using labels)
        dec = pick["decimal_odds"].astype(float).fillna(0.0)
        win = pick["hit"].astype(float)
        # Realized ROI (1u per pick): win*(dec-1) - (1-win)
        realized = float((win*(dec-1) - (1-win)).mean())
        return realized, len(pick)

    rows = []
    for t in [0.55, 0.60, 0.62, 0.65, 0.70, 0.75]:
        r, n = roi_at(t)
        rows.append(dict(threshold=t, n=n, realized_roi=r))

    # Per-market hit rate & ROI at 0.60
    t0 = 0.60
    sel = D[D["q_model"] >= t0].copy()
    def _roi(df):
        if df.empty: return np.nan
        dec = df["decimal_odds"].astype(float)
        win = df["hit"].astype(float)
        return float((win*(dec-1) - (1-win)).mean())
    bym = sel.groupby(D["market_type"].str.lower()).apply(_roi).reset_index()
    bym.columns = ["market_type","realized_roi_at_0.60"]

    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    bym.to_csv(os.path.join(args.out_dir, "by_market.csv"), index=False)
    print(f"Backtest wrote: {args.out_dir}/summary.csv and by_market.csv")

if __name__ == "__main__":
    main()