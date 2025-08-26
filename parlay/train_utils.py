# parlay/train_utils.py
from __future__ import annotations
import os, glob, zipfile
from datetime import datetime
import numpy as np
import pandas as pd

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

def _cap3(s): return ("" if s is None else str(s)).upper()[:3]

def load_snapshots_props(snapshots_dir: str) -> pd.DataFrame:
    rows = []
    for zp in sorted(glob.glob(os.path.join(snapshots_dir, "*.zip"))):
        try:
            with zipfile.ZipFile(zp) as z:
                with z.open("props.csv") as f:
                    p = pd.read_csv(f)
                with z.open("events.csv") as f:
                    e = pd.read_csv(f)
                e["date"] = pd.to_datetime(e["start"]).dt.date.astype(str)
                dm = dict(zip(e["event_id"].astype(str), e["date"]))
                p["event_id"] = p.get("event_id","").astype(str)
                p["date"] = p["event_id"].map(dm)
                rows.append(p)
        except Exception:
            continue
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()

def _american_to_prob(a):
    try:
        a = float(a)
        return 100.0/(a+100.0) if a>=0 else (-a)/((-a)+100.0)
    except Exception: return np.nan

def _load_opp_rates(year: int) -> pd.DataFrame:
    try:
        from pybaseball import team_batting
        tb = team_batting(year)
        tb["PA"] = tb["AB"] + tb["BB"] + tb["HBP"].fillna(0) + tb["SF"].fillna(0)
        tb["opp_k_pct"] = tb["SO"] / tb["PA"]
        tb["opp_bb_pct"] = tb["BB"] / tb["PA"]
        rows = []
        from parlay.feature_join import TEAM_ABBR as MAP_ABBR
        for _, r in tb.iterrows():
            name = str(r.get("Team",""))
            # map fuzzy -> abbr
            abbr = None
            for k,v in MAP_ABBR.items():
                if name in k or k in name: abbr = v; break
            if abbr:
                rows.append(dict(team_abbr=abbr, opp_k_pct=float(r["opp_k_pct"]), opp_bb_pct=float(r["opp_bb_pct"])))
        df = pd.DataFrame(rows)
        if df.empty: raise RuntimeError("empty")
        return df
    except Exception:
        return pd.DataFrame([dict(team_abbr=ab, opp_k_pct=REFS["opp_k_pct"], opp_bb_pct=REFS["opp_bb_pct"]) for ab in PARK.keys()])

def build_training_frame(snap_props: pd.DataFrame, results: pd.DataFrame):
    if snap_props is None or snap_props.empty:
        return pd.DataFrame(), pd.DataFrame(), np.array([])
    df = snap_props.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[df["market_type"].str.contains("pitcher_", na=False)]
    df = df[["event_id","date","player_name","team_abbr","market_type","side","line","american_odds","home_abbr","away_abbr"]]
    df["team_abbr"] = df["team_abbr"].map(_cap3)
    df["player_name"] = df["player_name"].fillna("")
    df = df[df["player_name"].ne("")]
    df["opp_abbr"] = np.where(df["team_abbr"].eq(df["home_abbr"]), df["away_abbr"], df["home_abbr"]).astype(str).str.upper().str[:3]
    R = results.copy()
    R["date"] = pd.to_datetime(R["date"]).dt.date.astype(str)
    X = df.merge(R, on=["date","player_name","team_abbr"], how="left", suffixes=("",""))
    # Opponent rates per season
    X["year"] = pd.to_datetime(X["date"]).dt.year
    frames = []
    for yr, g in X.groupby("year"):
        rates = _load_opp_rates(int(yr))
        gg = g.merge(rates, left_on="opp_abbr", right_on="team_abbr", how="left", suffixes=("","_opp"))
        if "team_abbr_opp" in gg.columns: gg.drop(columns=["team_abbr_opp"], inplace=True)
        frames.append(gg)
    X = pd.concat(frames, ignore_index=True, sort=False)
    # Feature engineering (align with app)
    X["park_k_factor"]  = X["home_abbr"].map(lambda t: PARK.get(_cap3(t), {"k":1.0})["k"]).fillna(1.0)
    X["park_bb_factor"] = X["home_abbr"].map(lambda t: PARK.get(_cap3(t), {"bb":1.0})["bb"]).fillna(1.0)
    X["weather_factor"] = 1.0
    X["pitch_mix_fit"]  = 0.0
    X["bullpen_fatigue"]= 0.0
    # trailing form proxies (if ETL included them, great; else neutral)
    X["exp_ip"]       = X.get("ip_l5", pd.Series([np.nan]*len(X))).fillna(5.6)
    X["exp_pitches"]  = X.get("pc_l5", pd.Series([np.nan]*len(X))).fillna(92.0)
    X["form_k9"]      = X.get("K9_l5", pd.Series([np.nan]*len(X))).fillna(8.5)
    X["form_bb9"]     = X.get("BB9_l5", pd.Series([np.nan]*len(X))).fillna(3.3)
    X["p_market"]     = X["american_odds"].map(_american_to_prob)
    # Labels
    def label(row):
        mt = str(row["market_type"]).lower()
        side = str(row["side"]).lower()
        ln = float(row["line"]) if pd.notna(row["line"]) else np.nan
        ks, bb, outs, win = row.get("ks"), row.get("bb"), row.get("outs"), row.get("win")
        if "strikeout" in mt and pd.notna(ln) and pd.notna(ks):
            return 1.0 if (ks > ln if side=="over" else ks < ln) else np.nan
        if "walks" in mt and pd.notna(ln) and pd.notna(bb):
            return 1.0 if (bb > ln if side=="over" else bb < ln) else np.nan
        if "outs" in mt and pd.notna(ln) and pd.notna(outs):
            return 1.0 if (outs > ln if side=="over" else outs < ln) else np.nan
        if "record_a_win" in mt and pd.notna(win):
            return 1.0 if (win==1 if side in ("over","yes") else win==0) else 0.0
        return np.nan
    X["y"] = X.apply(label, axis=1)
    X = X.dropna(subset=["y","p_market"]).reset_index(drop=True)
    feats = ["opp_k_pct","opp_bb_pct","pitch_mix_fit","exp_ip","exp_pitches",
             "park_k_factor","park_bb_factor","weather_factor","bullpen_fatigue",
             "form_k9","form_bb9"]
    for f in feats:
        if f not in X.columns: X[f] = REFS[f]
    F = pd.DataFrame(index=X.index)
    for f in feats:
        base = REFS[f]
        v = pd.to_numeric(X[f], errors="coerce").fillna(base)
        if f == "exp_pitches": F[f] = (v - base) / 20.0
        elif f == "form_k9":   F[f] = (v - base) / 2.0
        else:                  F[f] = (v - base)
    return X, F, X["y"].astype(float).values

def fit_logit(F: pd.DataFrame, y: np.ndarray, l2: float = 1.0, lr: float = 0.05, max_iter: int = 2000):
    X = F.values.astype("float64")
    n, d = X.shape
    w = np.zeros(d, dtype="float64"); b = 0.0
    for it in range(max_iter):
        z = X @ w + b
        p = 1/(1+np.exp(-z))
        g_w = (X.T @ (p - y))/n + l2*w
        g_b = np.sum(p - y)/n
        w -= lr*g_w; b -= lr*g_b
    out = {"intercept": float(b)}
    for i, f in enumerate(F.columns): out[f] = float(w[i])
    from .train_utils import DEFAULT as DEF
    for k,v in DEF.items(): out.setdefault(k, v)
    return out