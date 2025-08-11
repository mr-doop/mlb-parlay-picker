# parlay/projections.py
# ---------------------------------------------------------------------
# Converts features + odds rows into projection-driven probabilities
# for pitcher Ks, walks, outs, and pitcher win. NaN-proof, mobile-safe.
# ---------------------------------------------------------------------

from __future__ import annotations
import math
from typing import Dict
import numpy as np
import pandas as pd

# ----------------------------- utilities -----------------------------

def _num(x, default=0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def _flag(x) -> int:
    if x is None: return 0
    try:
        s = str(x).strip().lower()
        if s in ("1","true","t","yes","y"): return 1
        if s in ("0","false","f","no","n"): return 0
        return 1 if float(s) >= 0.5 else 0
    except Exception:
        return 0

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

# ----------------------- simple probabilistic cores -------------------

def _poisson_tail_ge(k: int, lam: float) -> float:
    """P[X >= k] for Poisson(lam). Iterative for stability (k up to ~40)."""
    if lam <= 0: return 0.0 if k > 0 else 1.0
    k = max(0, int(math.floor(k)))
    # compute P(X<=k-1) and return 1 - CDF
    p = math.exp(-lam)
    cdf = p if k > 0 else 0.0
    for i in range(1, k):
        p = p * lam / i
        cdf += p
    return 1.0 - cdf

def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def prob_over_count(mu: float, dispersion: float, line: float) -> float:
    """
    Rough tail for discrete counts (Ks/BB). We use a Poisson with mean mu.
    line is typically x.5 (e.g., 4.5). For over: ceil(line).
    """
    if mu <= 0 or line is None or math.isnan(line): return np.nan
    k = int(math.floor(line + 1e-12)) + 1  # next integer above .5
    return _poisson_tail_ge(k, mu)

def project_expected_ip(pitch_count_mean: float, days_rest: float, leash_bias: float,
                        favorite_flag: int, bullpen_freshness: float) -> float:
    """
    Expected IP from pitch count, rest, leash, favorite, and bullpen state.
    - Base: pc / 15.0 (approx pitches per inning)
    - Adjust: favorites +0.2 IP, tired pen +up to +0.3 IP, hot run env handled elsewhere
    """
    base = pitch_count_mean / 15.0
    adj = 0.0
    adj += 0.20 * _flag(favorite_flag)
    adj += 0.02 * (days_rest - 5.0)           # ±0.1 IP for ±5 days
    adj += 0.10 * max(0.0, 6.5 - bullpen_freshness)   # more if pen is tired
    adj += leash_bias                          # manager tendency learned externally
    ip = _clamp(base + adj, 3.5, 7.3)         # keep in [3.5, 7.3] IP
    return ip

def project_ks_mu(expected_ip: float, pitcher_k_rate: float, opp_k_rate: float,
                  park_k_factor: float, run_env_delta: float, ump_k_bias: float = 0.0) -> float:
    """
    Ks mean ~ expected IP * BF/IP * K% with small multiplicative factors.
    """
    bf_per_ip = 3.9
    k_rate = (0.6 * pitcher_k_rate + 0.4 * opp_k_rate)
    mult = park_k_factor
    mult *= (1.0 - 0.06 * run_env_delta)      # hot/windy-out hurts Ks
    mult *= (1.0 + 0.02 * ump_k_bias)
    mu = expected_ip * bf_per_ip * k_rate * mult
    return max(0.0, mu)

def project_bb_mu(expected_ip: float, pitcher_bb_rate: float, opp_bb_rate: float,
                  run_env_delta: float, ump_k_bias: float = 0.0) -> float:
    """
    Walks mean grows slightly in runny environments and with 'tight' zones (inverse of K-bias).
    """
    bf_per_ip = 3.9
    bb_rate = (0.7 * pitcher_bb_rate + 0.3 * opp_bb_rate)
    mult = 1.0 + 0.05 * max(0.0, run_env_delta)
    mult *= (1.0 + 0.01 * max(0.0, -ump_k_bias))  # if ump lowers K, BB can rise a tad
    mu = expected_ip * bf_per_ip * bb_rate * mult
    return max(0.0, mu)

def project_outs_probs(expected_ip: float, leash_bias: float) -> Dict[float, float]:
    """
    Map of outs thresholds (x.5) -> P(Over). Normal approx with σ about 2.0 outs.
    """
    mu_outs = expected_ip * 3.0
    sigma = _clamp(2.2 - 0.4 * max(0.0, leash_bias), 1.2, 2.8)
    outs_lines = [13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5]
    probs = {}
    for L in outs_lines:
        z = (L - mu_outs) / sigma
        p_over = 1.0 - _norm_cdf(z)
        probs[L] = _clamp(p_over, 0.0, 1.0)
    return probs

def project_win_prob(team_ml_prob: float, p_ip_ge5: float) -> float:
    """
    Pitcher win needs both a team win and 5 IP qualification. Approximate coupling.
    """
    p = team_ml_prob * (0.25 + 0.75 * _clamp(p_ip_ge5, 0.0, 1.0))
    return _clamp(p, 0.0, 1.0)

# ----------------------------- main API ------------------------------

def apply_projections(legs_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge features onto odds rows and compute q_proj for markets we model:
    - PITCHER_KS, PITCHER_WALKS, PITCHER_OUTS, PITCHER_WIN
    Adds: q_proj, mu_ks, mu_bb, expected_ip and leaves other markets untouched.
    """
    df = legs_df.copy()
    feats = features_df.copy()

    # ---- ensure required columns exist with defaults ----
    defaults = {
        "pitcher_k_rate": 0.24,
        "pitcher_bb_rate": 0.08,
        "opp_k_rate": 0.22,
        "opp_bb_rate": 0.08,
        "last5_pitch_ct_mean": 90.0,
        "days_rest": 5.0,
        "leash_bias": 0.0,
        "favorite_flag": 0,
        "bullpen_freshness": 6.0,
        "park_k_factor": 1.00,
        "park_run_factor": 1.00,
        "ump_k_bias": 0.00,
        "team_ml_vigfree": 0.50,
        "run_env_delta": 0.0,
    }
    for c, d in defaults.items():
        if c not in feats.columns:
            feats[c] = d

    # Coerce numerics/flags in features
    num_cols = [c for c in defaults.keys() if c != "favorite_flag"]
    for c in num_cols:
        feats[c] = feats[c].apply(lambda v: _num(v, defaults[c]))
    feats["favorite_flag"] = feats["favorite_flag"].apply(_flag).astype(int)

    # Merge by player_id
    df = df.merge(feats, on="player_id", how="left", suffixes=("","_feat"))

    # Post-merge fill for players not present in features
    for c, d in defaults.items():
        if c not in df.columns:
            df[c] = d
        df[c] = df[c].apply(lambda v: _num(v, d)) if c != "favorite_flag" else df[c].apply(_flag).astype(int)

    # Ensure alt_line numeric
    if "alt_line" in df.columns:
        df["alt_line"] = pd.to_numeric(df["alt_line"], errors="coerce")

    # Compute expected IP rowwise using features
    df["expected_ip"] = df.apply(
        lambda r: project_expected_ip(
            _num(r.get("last5_pitch_ct_mean"), 90.0),
            _num(r.get("days_rest"), 5.0),
            _num(r.get("leash_bias"), 0.0),
            _flag(r.get("favorite_flag")),
            _num(r.get("bullpen_freshness"), 6.0)
        ),
        axis=1
    )

    # ----- PITCHER_KS -----
    mask_ks = (df["market_type"] == "PITCHER_KS") & df["alt_line"].notna()
    if mask_ks.any():
        df.loc[mask_ks, "mu_ks"] = df.loc[mask_ks].apply(
            lambda r: project_ks_mu(
                _num(r.get("expected_ip")),
                _num(r.get("pitcher_k_rate")),
                _num(r.get("opp_k_rate")),
                _num(r.get("park_k_factor"), 1.0),
                _num(r.get("run_env_delta"), 0.0),
                _num(r.get("ump_k_bias"), 0.0)
            ), axis=1
        )
        def _ks_prob_row(r):
            line = _num(r.get("alt_line"), math.nan)
            if math.isnan(line): return np.nan
            p_over = prob_over_count(_num(r.get("mu_ks"), 0.0), 0.20, line)
            side = str(r.get("side","")).upper()
            if side == "OVER":
                return p_over
            elif side == "UNDER":
                # For integer X and half line (x.5), P(Under) = 1 - P(Over)
                return 1.0 - p_over
            return np.nan
        df.loc[mask_ks, "q_proj"] = df.loc[mask_ks].apply(_ks_prob_row, axis=1)

    # ----- PITCHER_WALKS -----
    mask_bb = (df["market_type"] == "PITCHER_WALKS") & df["alt_line"].notna()
    if mask_bb.any():
        df.loc[mask_bb, "mu_bb"] = df.loc[mask_bb].apply(
            lambda r: project_bb_mu(
                _num(r.get("expected_ip")),
                _num(r.get("pitcher_bb_rate")),
                _num(r.get("opp_bb_rate")),
                _num(r.get("run_env_delta"), 0.0),
                _num(r.get("ump_k_bias"), 0.0)
            ), axis=1
        )
        def _bb_prob_row(r):
            line = _num(r.get("alt_line"), math.nan)
            if math.isnan(line): return np.nan
            p_over = prob_over_count(_num(r.get("mu_bb"), 0.0), 0.25, line)
            side = str(r.get("side","")).upper()
            if side == "OVER":  return p_over
            if side == "UNDER": return 1.0 - p_over
            return np.nan
        df.loc[mask_bb, "q_proj"] = df.loc[mask_bb].apply(_bb_prob_row, axis=1)

    # ----- PITCHER_OUTS -----
    mask_outs = (df["market_type"] == "PITCHER_OUTS") & df["alt_line"].notna()
    if mask_outs.any():
        def _outs_prob_row(r):
            tails = project_outs_probs(_num(r.get("expected_ip")), _num(r.get("leash_bias"), 0.0))
            line = _num(r.get("alt_line"), math.nan)
            if math.isnan(line): return np.nan
            # nearest half-step among generated keys
            key = min(tails.keys(), key=lambda k: abs(float(k) - float(line)))
            p_over = tails.get(key, np.nan)
            side = str(r.get("side","")).upper()
            if side == "OVER":  return p_over
            if side == "UNDER": return 1.0 - p_over if not math.isnan(p_over) else np.nan
            return np.nan
        df.loc[mask_outs, "q_proj"] = df.loc[mask_outs].apply(_outs_prob_row, axis=1)

    # ----- PITCHER_WIN -----
    mask_win = df["market_type"].eq("PITCHER_WIN")
    if mask_win.any():
        # team win probability proxy (vig-free if given; else market p)
        vf_team = df.get("team_ml_vigfree").fillna(df.get("vigfree_p", 0.5)).astype(float)
        # probability of 5 IP qualification (≈ outs >= 15)
        p_ip_ge5 = df.apply(
            lambda r: project_outs_probs(_num(r.get("expected_ip"), 5.5), _num(r.get("leash_bias"), 0.0)).get(14.5, 0.6),
            axis=1
        )
        q_win_series = df.loc[mask_win].apply(
            lambda r: project_win_prob(
                vf_team.loc[r.name],
                p_ip_ge5.loc[r.name]
            ), axis=1
        )
        df.loc[mask_win, "q_proj"] = q_win_series
        mask_win_no = mask_win & df["side"].str.upper().eq("NO")
        df.loc[mask_win_no, "q_proj"] = 1.0 - df.loc[mask_win_no, "q_proj"]

    return df