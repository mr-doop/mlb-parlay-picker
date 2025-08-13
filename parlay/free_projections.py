# parlay/free_projections.py
# Free (credit‑free) projections for pitcher props using public data
# Dependencies: numpy, pandas, requests, pybaseball, streamlit (for cache)

from __future__ import annotations
import math
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    from pybaseball import pitching_stats
except Exception:  # pybaseball present in your requirements
    pitching_stats = None


# ---------- Utilities ----------

def _poisson_over(mu: float, line: float) -> float:
    """P(X > line) for Poisson with mean mu. We treat line as real; threshold = floor(line)."""
    if mu is None or not np.isfinite(mu) or mu <= 0:
        return np.nan
    k = int(math.floor(line))
    # 1 - CDF(k)
    # sum_{i=0..k} e^{-mu} mu^i / i!
    # avoid loops for big k: but our lines are small (< 25), so this is fine.
    cdf = 0.0
    term = math.exp(-mu)  # i=0
    cdf += term
    for i in range(1, k + 1):
        term *= mu / i
        cdf += term
    return float(max(0.0, min(1.0, 1.0 - cdf)))

@st.cache_data(ttl=60*60, show_spinner=False)
def _load_pitcher_table(season: int) -> pd.DataFrame:
    """FanGraphs season table via pybaseball: Name, K%, BB%, IP, GS."""
    if pitching_stats is None:
        return pd.DataFrame()
    try:
        df = pitching_stats(season=season, qual=0)
    except Exception:
        return pd.DataFrame()
    # Normalize
    df = df.rename(columns={
        "Name": "player_name",
        "K%": "k_pct",
        "BB%": "bb_pct",
        "IP": "ip",
        "GS": "gs",
    })
    for col in ["k_pct","bb_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0
    for col in ["ip","gs"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # key: lowercase, no periods
    df["player_key"] = df["player_name"].fillna("").str.replace(".","", regex=False).str.lower().str.strip()
    return df[["player_name","player_key","k_pct","bb_pct","ip","gs"]]

def _bf_from_ip(ip: float) -> float:
    # MLB avg batters faced per IP ~ 4.3
    return float(ip) * 4.3 if pd.notna(ip) else np.nan

def _ip_per_start(ip: float, gs: float) -> float:
    if pd.isna(ip) or pd.isna(gs) or gs <= 0:
        return np.nan
    return float(ip) / float(gs)

def _adj_rate(p_rate: float, opp_rate: float, weight=0.5) -> float:
    """Blend pitcher’s rate with opponent tendency. weight ∈ [0,1] on opponent."""
    if pd.isna(p_rate) and pd.isna(opp_rate): return np.nan
    if pd.isna(p_rate): return float(opp_rate)
    if pd.isna(opp_rate): return float(p_rate)
    return (1.0 - weight)*float(p_rate) + weight*float(opp_rate)

# ---------- Main entry ----------

def build_free_q(pool: pd.DataFrame, slate_date: datetime, opp_rates: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of pool with q_model filled (where possible) for:
      - pitcher_strikeouts (+_alternate)
      - pitcher_walks (+_alternate)
      - pitcher_outs
      - pitcher_record_a_win
    Uses FanGraphs season table + opponent rates (last 21d) + simple Poisson model.
    """
    if pool is None or pool.empty:
        return pool

    season = slate_date.year
    fg = _load_pitcher_table(season)
    if fg.empty:
        return pool  # graceful fallback; UI will still use p_market

    # Build join keys
    df = pool.copy()
    df["player_key"] = df["player_name"].fillna("").str.replace(".","", regex=False).str.lower().str.strip()
    fg = fg.drop_duplicates(subset=["player_key"], keep="first")

    df = df.merge(fg, on="player_key", how="left", suffixes=("","_fg"))

    # Join opponent K%/BB% (left on opp_abbr)
    rates = (opp_rates or pd.DataFrame()).copy()
    rates = rates.rename(columns={"team_abbr":"opp_abbr_join"})
    df = df.merge(rates, left_on="opp_abbr", right_on="opp_abbr_join", how="left")
    df.drop(columns=["opp_abbr_join"], inplace=True, errors="ignore")

    # Compute base quantities
    df["ip_ps"]  = _ip_per_start(df["ip"], df["gs"])
    df["bf_mu"]  = _bf_from_ip(df["ip_ps"].fillna(5.5))  # if missing, assume ~5.5 IP per start
    df["k_rate"] = _adj_rate(df["k_pct"], df["opp_k_rate"], weight=0.45)
    df["bb_rate"]= _adj_rate(df["bb_pct"], df["opp_bb_rate"], weight=0.45)

    # Means for Poisson models
    df["mu_k"]   = df["bf_mu"] * df["k_rate"]
    df["mu_bb"]  = df["bf_mu"] * df["bb_rate"]
    df["mu_out"] = (df["ip_ps"].fillna(5.5)) * 3.0

    # Initialize q_model to existing p_market for safety
    if "q_model" not in df.columns:
        df["q_model"] = df.get("p_market")

    # Strikeouts / Walks / Outs
    def _compute_prob(row) -> float | np.nan:
        mk = row.get("market_key","")
        side = (row.get("side") or "").title()
        line = row.get("line")
        if pd.isna(line): return np.nan
        if "strikeouts" in mk:
            mu = row.get("mu_k")
        elif "walks" in mk:
            mu = row.get("mu_bb")
        elif "outs" in mk:
            mu = row.get("mu_out")
        else:
            return np.nan
        if pd.isna(mu): return np.nan
        if side == "Over":
            return _poisson_over(mu, float(line))
        elif side == "Under":
            # P(X ≤ line) = 1 - P(X > line)
            return 1.0 - _poisson_over(mu, float(line))
        else:
            return np.nan

    df["q_free"] = df.apply(_compute_prob, axis=1)

    # Win: team must win AND pitcher must record 15+ outs
    # Approx: q_win = p_team * P(outs >= 15)
    def _compute_win(row) -> float | np.nan:
        mk = row.get("market_key","")
        side = (row.get("side") or "").title()
        if "record_a_win" not in mk: return np.nan
        if side not in ("Yes","No"): return np.nan
        p_team = row.get("p_market")  # vig-free ML already in pool for ML; for a prop we fallback to implied
        if pd.isna(p_team):
            # fallback: implied from odds (already computed in app as p_market)
            p_team = row.get("p_market")
        p_outs15 = 1.0 - _poisson_over(row.get("mu_out", np.nan), 14.5) if pd.notna(row.get("mu_out")) else np.nan
        if pd.isna(p_team) or pd.isna(p_outs15): return np.nan
        yes = float(p_team) * float(p_outs15)
        return yes if side == "Yes" else (1.0 - yes)

    df["q_win"] = df.apply(_compute_win, axis=1)

    # Prefer the more specific calculation when present
    df["q_model"] = np.where(df["q_free"].notna(), df["q_free"], df["q_model"])
    df["q_model"] = np.where(df["q_win"].notna(),  df["q_win"],  df["q_model"])

    # Clean
    df.drop(columns=[
        "player_key","k_pct","bb_pct","ip","gs","ip_ps","bf_mu","k_rate","bb_rate","mu_k","mu_bb","mu_out","q_free","q_win"
    ], inplace=True, errors="ignore")

    return df