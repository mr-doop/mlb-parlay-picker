# parlay/train_model.py
"""
Train logistic weights for the enhanced prop model from historical data and
overwrite parlay/model_weights.json.

USAGE
-----
python -m parlay.train_model \
  --snapshots_dir snapshots/ \
  --results_csv data/pitcher_results.csv \
  --out parlay/model_weights.json

INPUTS
------
1) snapshots_dir: a folder of snapshot ZIPs produced by the app (each has
   events.csv and props.csv). We'll use props rows for markets:
      - pitcher_strikeouts (+ alternate)
      - pitcher_walks (+ alternate)
      - pitcher_outs
      - pitcher_record_a_win
   We need event_id->date to align with results; if missing, we fall back to
   the file timestamp (less ideal).

2) results_csv: your pitcher game results with at least:
      date (YYYY-MM-DD)
      player_name
      team_abbr (3-letter)
      opp_abbr (optional)
      ks (strikeouts), bb (walks), outs (total outs), win (0/1)
      pitch_count (optional)
   You can export these from your own DB or any box-score source.

OUTPUT
------
A JSON with weights compatible with parlay/feature_join.apply_enhanced_model.
No external ML library is required (pure NumPy logistic regression with L2).
"""

from __future__ import annotations
import argparse, glob, io, json, os, zipfile
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Keep these in sync with parlay.feature_join
REFS = {
    "opp_k_pct":      0.22,
    "opp_bb_pct":     0.08,
    "pitch_mix_fit":  0.00,
    "exp_ip":         5.50,
    "exp_pitches":    90.0,
    "park_k_factor":  1.00,
    "park_bb_factor": 1.00,
    "weather_factor": 1.00,
    "bullpen_fatigue":0.00,
    "form_k9":        8.50,
    "form_bb9":       3.30,
}

DEFAULT = {
  "intercept": -0.10,
  "opp_k_pct": 1.50,
  "opp_bb_pct": -1.20,
  "pitch_mix_fit": 0.50,
  "exp_ip": 0.80,
  "exp_pitches": 0.40,
  "park_k_factor": 0.30,
  "park_bb_factor": -0.20,
  "weather_factor": 0.20,
  "bullpen_fatigue": -0.30,
  "form_k9": 0.60,
  "form_bb9": -0.60
}

PARK = {
    "ARI":{"k":0.99,"bb":1.00},"ATL":{"k":0.99,"bb":0.98},"BAL":{"k":1.01,"bb":1.00},
    "BOS":{"k":0.98,"bb":1.01},"CHC":{"k":1.00,"bb":1.00},"CWS":{"k":1.00,"bb":0.99},
    "CIN":{"k":0.99,"bb":1.00},"CLE":{"k":1.01,"bb":1.01},"COL":{"k":0.95,"bb":1.02},
    "DET":{"k":1.02,"bb":1.01},"HOU":{"k":0.99,"bb":0.99},"KC":{"k":1.00,"bb":1.00},
    "LAA":{"k":1.00,"bb":0.99},"LAD":{"k":1.00,"bb":0.99},"MIA":{"k":1.02,"bb":1.01},
    "MIL":{"k":0.99,"bb":1.00},"MIN":{"k":1.01,"bb":1.00},"NYM":{"k":1.00,"bb":1.01},
    "NYY":{"k":0.99,"bb":0.99},"OAK":{"k":1.01,"bb":1.02},"PHI":{"k":0.99,"bb":0.99},
    "PIT":{"k":1.01,"bb":1.01},"SDP":{"k":1.02,"bb":1.00},"SFG":{"k":1.02,"bb":1.01},
    "SEA":{"k":1.01,"bb":1.00},"STL":{"k":1.00,"bb":1.00},"TBR":{"k":1.00,"bb":1.00},
    "TEX":{"k":0.99,"bb":0.99},"TOR":{"k":1.00,"bb":0.99},"WSH":{"k":1.00,"bb":1.01},
}

TEAM_ABBR = set(PARK.keys())

def _cap3(s): 
    return ("" if s is None else str(s)).upper()[:3]

def _sigmoid(z): 
    z = np.asarray(z, dtype="float64")
    return 1.0/(1.0+np.exp(-z))

def _american_to_prob(a):
    try:
        a = float(a)
        if a >= 0: return 100.0/(a+100.0)
        return (-a)/((-a)+100.0)
    except Exception:
        return np.nan

def _load_snapshots_dir(snapshots_dir: str) -> pd.DataFrame:
    """Return long props dataframe with event date joined when possible."""
    rows = []
    zips = sorted(glob.glob(os.path.join(snapshots_dir, "*.zip")))
    for zp in zips:
        try:
            with zipfile.ZipFile(zp) as z:
                with z.open("props.csv") as f:
                    props = pd.read_csv(f)
                # optional: events for date
                date_map = {}
                try:
                    with z.open("events.csv") as f:
                        ev = pd.read_csv(f)
                        ev["date"] = pd.to_datetime(ev.get("start")).dt.date.astype(str)
                        date_map = dict(zip(ev.get("event_id",""), ev["date"]))
                except Exception:
                    pass
                props["date"] = props.get("event_id","").map(date_map)
                # fallback: file mtime date
                if props["date"].isna().all():
                    d = datetime.fromtimestamp(os.path.getmtime(zp)).date().isoformat()
                    props["date"] = d
                rows.append(props)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True, sort=False)
    return df

def _load_results_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    # required columns
    needed = {"date","player_name","team_abbr","ks","bb","outs","win"}
    miss = needed - set(df.columns)
    if miss:
        raise ValueError(f"results_csv missing columns: {sorted(miss)}")
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    # optional pitch_count
    if "pitch_count" not in df.columns:
        df["pitch_count"] = np.nan
    # standardize team
    df["team_abbr"] = df["team_abbr"].map(_cap3)
    return df

def _rolling_form(results: pd.DataFrame) -> pd.DataFrame:
    """Compute trailing 5 form per pitcher-date (shifted so current game not included)."""
    r = results.copy().sort_values(["player_name","date"])
    r["ip"] = r["outs"].astype(float)/3.0
    def _agg(g):
        g = g.copy()
        g["ks_l5"]  = g["ks"].shift().rolling(5, min_periods=1).mean()
        g["bb_l5"]  = g["bb"].shift().rolling(5, min_periods=1).mean()
        g["ip_l5"]  = g["ip"].shift().rolling(5, min_periods=1).mean()
        g["pc_l5"]  = g["pitch_count"].shift().rolling(5, min_periods=1).mean()
        # rates per 9 using rolling sums
        g["K9_l5"] = (g["ks"].shift().rolling(5, min_periods=1).sum() /
                      (g["outs"].shift().rolling(5, min_periods=1).sum()/3.0).clip(lower=1.0)) * 9.0
        g["BB9_l5"] = (g["bb"].shift().rolling(5, min_periods=1).sum() /
                       (g["outs"].shift().rolling(5, min_periods=1).sum()/3.0).clip(lower=1.0)) * 9.0
        return g
    r = r.groupby("player_name", group_keys=False).apply(_agg)
    return r

def _opp_rates_fallback() -> pd.DataFrame:
    return pd.DataFrame([dict(team_abbr=ab, opp_k_pct=0.22, opp_bb_pct=0.08) for ab in TEAM_ABBR])

def _load_opp_rates(year: int) -> pd.DataFrame:
    """Try pybaseball; fallback to league avg."""
    try:
        from pybaseball import team_batting
        tb = team_batting(year)
        tb["PA"] = tb["AB"] + tb["BB"] + tb["HBP"].fillna(0) + tb["SF"].fillna(0)
        tb["opp_k_pct"] = tb["SO"] / tb["PA"]
        tb["opp_bb_pct"] = tb["BB"] / tb["PA"]
        map_rows = []
        for _, r in tb.iterrows():
            name = str(r.get("Team",""))
            ab = None
            for k in TEAM_ABBR:
                if k in name or name in k: ab = k; break
            if ab:
                map_rows.append(dict(team_abbr=ab, opp_k_pct=float(r["opp_k_pct"]), opp_bb_pct=float(r["opp_bb_pct"])))
        df = pd.DataFrame(map_rows)
        if df.empty: raise RuntimeError("empty")
        return df
    except Exception:
        return _opp_rates_fallback()

def build_training_frame(snap_props: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    if snap_props is None or snap_props.empty: return pd.DataFrame()
    df = snap_props.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    # keep relevant
    df = df[df["market_type"].str.contains("pitcher_", na=False)]
    df = df[["event_id","date","player_name","team_abbr","market_type","side","line","american_odds","home_abbr","away_abbr"]]
    df["team_abbr"] = df["team_abbr"].map(_cap3)
    df["player_name"] = df["player_name"].fillna("")
    df = df[df["player_name"].ne("")]

    # Opponent team at game level (based on home/away and player's team)
    df["opp_abbr"] = np.where(df["team_abbr"].eq(df["home_abbr"]), df["away_abbr"], df["home_abbr"]).astype(str).str.upper().str[:3]

    # Merge results per (date, player_name, team)
    R = results.copy()
    tr = _rolling_form(R)
    use = R.merge(tr[["player_name","date","ks_l5","bb_l5","ip_l5","pc_l5","K9_l5","BB9_l5"]], on=["player_name","date"], how="left")
    X = df.merge(use, on=["date","player_name","team_abbr"], how="left", suffixes=("",""))

    # Opponent rates by season (approx by date year)
    X["year"] = pd.to_datetime(X["date"]).dt.year
    opp_frames = []
    for yr, g in X.groupby("year"):
        rates = _load_opp_rates(int(yr))
        gg = g.merge(rates, left_on="opp_abbr", right_on="team_abbr", how="left", suffixes=("","_opp"))
        if "team_abbr_opp" in gg.columns:
            gg.drop(columns=["team_abbr_opp"], inplace=True)
        opp_frames.append(gg)
    X = pd.concat(opp_frames, ignore_index=True, sort=False)

    # Park factors
    X["park_k_factor"]  = X["home_abbr"].map(lambda t: PARK.get(_cap3(t), {"k":1.0})["k"]).fillna(1.0)
    X["park_bb_factor"] = X["home_abbr"].map(lambda t: PARK.get(_cap3(t), {"bb":1.0})["bb"]).fillna(1.0)
    X["weather_factor"] = 1.0  # keep neutral in training (no historical weather)
    X["pitch_mix_fit"]  = 0.0
    X["bullpen_fatigue"]= 0.0

    # Expected innings / pitch count / form from trailing windows
    X["exp_ip"]       = X["ip_l5"].fillna(5.6)
    X["exp_pitches"]  = X["pc_l5"].fillna(92.0)
    X["form_k9"]      = X["K9_l5"].fillna(8.5)
    X["form_bb9"]     = X["BB9_l5"].fillna(3.3)

    # Market prob
    X["p_market"] = X["american_odds"].map(_american_to_prob)

    # Build labels
    def label(row):
        mt = str(row["market_type"]).lower()
        side = str(row["side"]).lower()
        ln = float(row["line"]) if pd.notna(row["line"]) else np.nan
        ks, bb, outs, win = row.get("ks"), row.get("bb"), row.get("outs"), row.get("win")
        if "strikeout" in mt and pd.notna(ln) and pd.notna(ks):
            return 1.0 if (ks > ln if side=="over" else ks < ln) else np.nan  # drop pushes
        if "walks" in mt and pd.notna(ln) and pd.notna(bb):
            return 1.0 if (bb > ln if side=="over" else bb < ln) else np.nan
        if "outs" in mt and pd.notna(ln) and pd.notna(outs):
            return 1.0 if (outs > ln if side=="over" else outs < ln) else np.nan
        if "record_a_win" in mt and pd.notna(win):
            return 1.0 if (win==1 if side in ("over","yes") else win==0) else 0.0
        return np.nan

    X["y"] = X.apply(label, axis=1)
    X = X.dropna(subset=["y","p_market"]).reset_index(drop=True)

    # Feature matrix (center on REFS; scale where needed)
    feats = ["opp_k_pct","opp_bb_pct","pitch_mix_fit","exp_ip","exp_pitches",
             "park_k_factor","park_bb_factor","weather_factor","bullpen_fatigue",
             "form_k9","form_bb9"]
    for f in feats:
        if f not in X.columns: X[f] = REFS[f]
    F = pd.DataFrame(index=X.index)
    for f in feats:
        base = REFS[f]
        v = pd.to_numeric(X[f], errors="coerce").fillna(base)
        if f == "exp_pitches":
            F[f] = (v - base) / 20.0
        elif f == "form_k9":
            F[f] = (v - base) / 2.0
        else:
            F[f] = (v - base)
    # Intercept separate
    return X, F, X["y"].astype(float).values

def fit_logit(F: pd.DataFrame, y: np.ndarray, l2: float = 1.0, max_iter: int = 2000, lr: float = 0.05) -> Dict[str, float]:
    """Simple L2-regularized logistic regression via gradient descent."""
    X = F.values.astype("float64")
    n, d = X.shape
    w = np.zeros(d, dtype="float64")
    b = 0.0
    for it in range(max_iter):
        z = X @ w + b
        p = 1/(1+np.exp(-z))
        # gradients
        g_w = (X.T @ (p - y)) / n + l2 * w
        g_b = np.sum(p - y) / n
        # update
        w -= lr * g_w
        b -= lr * g_b
        if it % 200 == 0:
            ll = -np.mean(y*np.log(p+1e-9) + (1-y)*np.log(1-p+1e-9)) + 0.5*l2*np.sum(w*w)
            # print(f"iter {it}: loss {ll:.4f}")
    # Pack weights using DEFAULT keys
    out = {"intercept": float(b)}
    for i, f in enumerate(F.columns):
        out[f] = float(w[i])
    # Merge with defaults so any missing stays reasonable
    for k,v in DEFAULT.items():
        out.setdefault(k, v)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots_dir", required=True)
    ap.add_argument("--results_csv", required=True)
    ap.add_argument("--out", default="parlay/model_weights.json")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--max_iter", type=int, default=2000)
    args = ap.parse_args()

    snaps = _load_snapshots_dir(args.snapshots_dir)
    if snaps.empty:
        raise SystemExit("No snapshots found. Put your snapshot ZIPs in the directory.")

    results = _load_results_csv(args.results_csv)
    D, F, y = build_training_frame(snaps, results)
    if D.empty:
        raise SystemExit("No training rows assembled. Check your results CSV aligns with snapshots (date/player/team).")

    w = fit_logit(F, y, l2=args.l2, lr=args.lr, max_iter=args.max_iter)

    # Quick sanity metrics
    z = F.values @ np.array([w.get(f,0.0) for f in F.columns]) + w["intercept"]
    p = 1/(1+np.exp(-z))
    pred = (p >= 0.5).astype(float)
    acc = float((pred == y).mean())
    print(f"Training rows: {len(y)}  |  Accuracy: {acc:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(w, f, indent=2)
    print(f"Wrote weights → {args.out}")

if __name__ == "__main__":
    main()