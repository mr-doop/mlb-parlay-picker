from __future__ import annotations
import math
import numpy as np
import pandas as pd
from typing import Optional

# ---------- Helpers ----------

def _clip(x, lo, hi):
    return max(lo, min(hi, x))

def _neg_binom_tail_prob(mu: float, alpha: float, k_threshold: float) -> float:
    """
    Approximate P(X >= ceil(k_threshold)) for Negative Binomial with mean mu and dispersion alpha.
    Uses a Poisson-gamma mixture approximation; falls back to normal approx if needed.
    alpha: dispersion (>0). Larger alpha -> more variance.
    """
    # Convert NB to parameters (r, p) where Var = mu + alpha*mu^2
    var = mu + alpha * (mu ** 2)
    if mu <= 0 or var <= 0:
        return 0.0
    # Normal approx for tail with continuity correction
    z = (math.ceil(k_threshold) - 0.5 - mu) / math.sqrt(var)
    # tail = 1 - Phi(z)
    return 0.5 * math.erfc(z / math.sqrt(2))

# ---------- Projections (very lightweight v0.1) ----------

def project_expected_ip(last5_pitch_ct_mean: float,
                        days_rest: float,
                        leash_bias: float,
                        favorite_flag: int,
                        bullpen_freshness: float) -> float:
    """Estimate expected IP from pitch count and simple context.
    Returns IP in innings (e.g., 5.8).
    """
    # Base from pitch count (assume ~15 pitches per inning)
    base_ip = (last5_pitch_ct_mean or 85.0) / 15.0  # default 85 pitches ~ 5.7 IP
    # Adjustments
    adj = 0.0
    adj += 0.15 * leash_bias            # [-0.15, +0.15] typical
    adj += 0.15 * favorite_flag         # +0.15 IP if fav (longer leash)
    adj -= 0.10 * _clip(bullpen_freshness / 7.5, 0, 1)  # fresh pen -> slightly quicker hook
    adj += 0.10 * _clip((days_rest - 4) / 2.0, -0.1, 0.2)  # 4-6 days sweet spot

    ip = _clip(base_ip + adj, 3.0, 7.5)
    return ip

def project_ks_mu(expected_ip: float,
                  pitcher_k_rate: float,
                  opp_k_rate: float,
                  park_k_factor: float,
                  ump_k_bias: float,
                  p_per_pa: float = 3.9) -> float:
    """Expected strikeouts (mean).
    pitcher_k_rate, opp_k_rate as fractions (e.g., 0.28 means 28% K per PA).
    park_k_factor and ump_k_bias around 1.00 baseline.
    """
    # Estimate batters faced
    bf = expected_ip * 4.2  # typical BF per IP (depends on WHIP; 4.2 is conservative)
    eff_k_rate = pitcher_k_rate * (0.5 + 0.5 * opp_k_rate / 0.22)  # scale opp vs 22% league avg
    eff_k_rate *= park_k_factor * ump_k_bias
    mu = _clip(bf * eff_k_rate, 0.5, 14.0)
    return mu

def project_bb_mu(expected_ip: float,
                  pitcher_bb_rate: float,
                  opp_bb_rate: float,
                  ump_bb_bias: float) -> float:
    bf = expected_ip * 4.2
    eff_bb = pitcher_bb_rate * (0.5 + 0.5 * opp_bb_rate / 0.08)  # vs 8% league avg
    eff_bb *= ump_bb_bias
    mu = _clip(bf * eff_bb, 0.0, 8.0)
    return mu

def prob_over_count(mu: float, alpha: float, line: float) -> float:
    """Return P(X >= line) for count variable with NB approx."""
    return float(_neg_binom_tail_prob(mu, alpha, line))

def project_outs_probs(expected_ip: float, leash_bias: float) -> dict:
    """Make a crude outs distribution using NB around mean outs with dispersion from leash."""
    mean_outs = expected_ip * 3.0
    alpha = _clip(0.10 + 0.25 * (1.0 - leash_bias), 0.05, 0.5)
    # Build a CDF up to 27 outs
    pmf = {}
    for outs in range(0, 28):
        # Use normal approx chunking for simplicity
        var = mean_outs + alpha * (mean_outs ** 2)
        z_lo = ((outs - 0.5) - mean_outs) / math.sqrt(var)
        z_hi = ((outs + 0.5) - mean_outs) / math.sqrt(var)
        cdf_lo = 0.5 * math.erfc(-z_lo / math.sqrt(2))
        cdf_hi = 0.5 * math.erfc(-z_hi / math.sqrt(2))
        pmf[outs] = max(0.0, cdf_hi - cdf_lo)
    # Tail helper
    def tail_at(thresh_half: float) -> float:
        t = int(thresh_half - 0.5)
        return sum(pmf.get(o, 0.0) for o in range(t+1, 28))
    return {x: tail_at(x) for x in [15.5, 16.5, 17.5, 18.5, 19.5]}

def project_win_prob(vigfree_team_ml_prob: float,
                     prob_ip_ge_5: float,
                     bullpen_hold_rate: float = 0.90) -> float:
    """Approximate SP win probability."""
    return _clip(vigfree_team_ml_prob * prob_ip_ge_5 * bullpen_hold_rate, 0.0, 0.9)

# ---------- Public entry points ----------

def apply_projections(legs_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge legs with features and compute q (model probability) for supported markets:
    - PITCHER_KS (OVER/UNDER threshold in alt_line)
    - PITCHER_WALKS (OVER/UNDER)
    - PITCHER_OUTS (OVER/UNDER)
    - PITCHER_WIN (YES/NO)
    Other markets will keep q from vigfree_p (caller should handle fallback).
    Required feature columns (by player_id):
      player_id, pitcher_k_rate, pitcher_bb_rate, opp_k_rate, opp_bb_rate,
      last5_pitch_ct_mean, days_rest, leash_bias, favorite_flag, bullpen_freshness,
      park_k_factor, ump_k_bias
    Optional: team_ml_vigfree for PITCHER_WIN if available; else fallback to leg vigfree.
    """
    df = legs_df.copy()
    feats = features_df.copy()
    # Merge by player_id where applicable
    df = df.merge(feats, on="player_id", how="left", suffixes=("","_feat"))
    # Expected IP first
    df["expected_ip"] = df.apply(lambda r: project_expected_ip(
        r.get("last5_pitch_ct_mean", 85.0) or 85.0,
        r.get("days_rest", 5) or 5,
        r.get("leash_bias", 0.0) or 0.0,
        int(r.get("favorite_flag", 0) or 0),
        r.get("bullpen_freshness", 6.0) or 6.0
    ), axis=1)

    # Ks
    mask_ks = df["market_type"].eq("PITCHER_KS") & df["alt_line"].notna()
    df.loc[mask_ks, "mu_ks"] = df[mask_ks].apply(lambda r: project_ks_mu(
        r["expected_ip"],
        float(r.get("pitcher_k_rate", 0.24) or 0.24),
        float(r.get("opp_k_rate", 0.22) or 0.22),
        float(r.get("park_k_factor", 1.0) or 1.0),
        float(r.get("ump_k_bias", 1.0) or 1.0)
    ), axis=1)
    # Dispersion for Ks
    df.loc[mask_ks, "q_proj"] = df[mask_ks].apply(lambda r: prob_over_count(
        r["mu_ks"], 0.20, float(r["alt_line"]) if str(r["side"]).upper()=="OVER" else float(r["alt_line"]) - 1.0
    ) if str(r["side"]).upper() in {"OVER","UNDER"} else np.nan, axis=1)
    # Flip for UNDER: P(X < L) = 1 - P(X >= L)
    under_mask = mask_ks & df["side"].str.upper().eq("UNDER")
    df.loc[under_mask, "q_proj"] = 1.0 - df.loc[under_mask, "q_proj"]

    # Walks
    mask_bb = df["market_type"].eq("PITCHER_WALKS") & df["alt_line"].notna()
    df.loc[mask_bb, "mu_bb"] = df[mask_bb].apply(lambda r: project_bb_mu(
        r["expected_ip"],
        float(r.get("pitcher_bb_rate", 0.08) or 0.08),
        float(r.get("opp_bb_rate", 0.08) or 0.08),
        float(r.get("ump_bb_bias", 1.0) or 1.0)
    ), axis=1)
    df.loc[mask_bb, "q_proj"] = df[mask_bb].apply(lambda r: prob_over_count(
        r["mu_bb"], 0.25, float(r["alt_line"]) if str(r["side"]).upper()=="OVER" else float(r["alt_line"]) - 1.0
    ) if str(r["side"]).upper() in {"OVER","UNDER"} else np.nan, axis=1)
    under_mask = mask_bb & df["side"].str.upper().eq("UNDER")
    df.loc[under_mask, "q_proj"] = 1.0 - df.loc[under_mask, "q_proj"]

    # Outs
    mask_outs = df["market_type"].eq("PITCHER_OUTS") & df["alt_line"].notna()
    # Tail probs at common half-outs (we'll interpolate if needed)
    tails = df[mask_outs].apply(lambda r: project_outs_probs(r["expected_ip"], float(r.get("leash_bias", 0.0) or 0.0)), axis=1)
    # Combine
    df.loc[mask_outs, "q_proj"] = [
        t.get(r["alt_line"], np.nan) if str(r["side"]).upper()=="OVER" else 1.0 - t.get(r["alt_line"], np.nan)
        for t, (_, r) in zip(tails, df[mask_outs].iterrows())
    ]

    # Pitcher Win
    mask_win = df["market_type"].eq("PITCHER_WIN")
    # get team ML vigfree if provided in features; else use the paired leg vigfree p (caller should pass that in df as vigfree_p)
    vf_team_ml = df.get("team_ml_vigfree", pd.Series(index=df.index, dtype=float)).fillna(df.get("vigfree_p", 0.5))
    # Compute P(IP >= 5)
    tails_ip5 = df.apply(lambda r: project_outs_probs(r.get("expected_ip", 5.5), float(r.get("leash_bias", 0.0) or 0.0)).get(14.5, 0.6), axis=1)
    df.loc[mask_win, "q_proj"] = [
        project_win_prob(vf_team_ml.iloc[i], tails_ip5.iloc[i], bullpen_hold_rate=0.90)
        for i in df.index
    ]
    # Flip for NO side
    mask_win_no = mask_win & df["side"].str.upper().eq("NO")
    df.loc[mask_win_no, "q_proj"] = 1.0 - df.loc[mask_win_no, "q_proj"]

    return df
