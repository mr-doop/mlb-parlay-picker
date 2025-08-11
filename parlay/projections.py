from __future__ import annotations
import math, numpy as np, pandas as pd
def _clip(x,lo,hi): return max(lo,min(hi,x))
def _nb_tail(mu,alpha,thr):
    var=mu+alpha*mu*mu
    if mu<=0 or var<=0: return 0.0
    z=(math.ceil(thr)-0.5-mu)/math.sqrt(var)
    return 0.5*math.erfc(z/(2**0.5))
def project_expected_ip(last5_pitch_ct_mean=85, days_rest=5, leash_bias=0.0, favorite_flag=0, bullpen_freshness=6.0):
    base=(last5_pitch_ct_mean or 85)/15.0
    adj=0.15*leash_bias + 0.15*favorite_flag - 0.10*_clip((bullpen_freshness/7.5),0,1) + 0.10*_clip((days_rest-4)/2.0,-0.1,0.2)
    return _clip(base+adj,3.0,7.5)
def project_ks_mu(expected_ip, pitcher_k_rate=0.24, opp_k_rate=0.22, park_k_factor=1.0, ump_k_bias=1.0):
    bf=expected_ip*4.2; eff= pitcher_k_rate*(0.5+0.5*opp_k_rate/0.22)*park_k_factor*ump_k_bias
    return _clip(bf*eff,0.5,14.0)
def project_bb_mu(expected_ip, pitcher_bb_rate=0.08, opp_bb_rate=0.08, ump_bb_bias=1.0):
    bf=expected_ip*4.2; eff=pitcher_bb_rate*(0.5+0.5*opp_bb_rate/0.08)*ump_bb_bias
    return _clip(bf*eff,0.0,8.0)
def prob_over_count(mu,alpha,line): return float(_nb_tail(mu,alpha,line))
def project_outs_probs(expected_ip, leash_bias=0.0):
    mean_outs=expected_ip*3.0; alpha=_clip(0.10+0.25*(1.0-leash_bias),0.05,0.5)
    pmf={}
    for outs in range(0,28):
        var=mean_outs+alpha*mean_outs**2
        zlo=((outs-0.5)-mean_outs)/math.sqrt(var); zhi=((outs+0.5)-mean_outs)/math.sqrt(var)
        cdf_lo=0.5*math.erfc(-zlo/(2**0.5)); cdf_hi=0.5*math.erfc(-zhi/(2**0.5))
        pmf[outs]=max(0.0,cdf_hi-cdf_lo)
    def tail_at(xh): t=int(xh-0.5); return sum(pmf.get(o,0.0) for o in range(t+1,28))
    return {x: tail_at(x) for x in [14.5,15.5,16.5,17.5,18.5,19.5]}
def project_win_prob(vf_ml, p_ip_ge5, bullpen_hold=0.90): return _clip(vf_ml*p_ip_ge5*bullpen_hold,0.0,0.9)
def apply_projections(legs_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    import math
    import numpy as np
    import pandas as pd

    df = legs_df.copy()
    feats = features_df.copy()

    # ---- defaults for missing values ----
    defaults = {
        "pitcher_k_rate": 0.24,
        "pitcher_bb_rate": 0.08,
        "opp_k_rate": 0.22,
        "opp_bb_rate": 0.08,
        "last5_pitch_ct_mean": 85.0,
        "days_rest": 5.0,
        "leash_bias": 0.0,            # -0.3 .. +0.3
        "favorite_flag": 0,           # 0/1
        "bullpen_freshness": 6.0,     # IP last 3 days (lower = fresher)
        "park_k_factor": 1.00,
        "ump_k_bias": 1.00,
        "team_ml_vigfree": 0.50
    }

    # ensure required columns exist in feats
    for c, d in defaults.items():
        if c not in feats.columns:
            feats[c] = d

    # helpers
    def _num(x, d):
        try:
            return float(x) if (x is not None and not pd.isna(x)) else float(d)
        except Exception:
            return float(d)

    def _flag(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return 0
        s = str(x).strip().lower()
        if s in ("1","true","t","yes","y"):
            return 1
        try:
            return 1 if float(s) >= 0.5 else 0
        except Exception:
            return 0

    # coerce features numerics/flag
    num_cols = [c for c in defaults.keys() if c != "favorite_flag"]
    for c in num_cols:
        feats[c] = feats[c].apply(lambda v: _num(v, defaults[c]))
    feats["favorite_flag"] = feats["favorite_flag"].apply(_flag).astype(int)

    # merge features onto legs
    df = df.merge(feats, on="player_id", how="left", suffixes=("","_feat"))

    # post-merge fill (covers players not present in feats)
    for c in num_cols:
        df[c] = df[c].apply(lambda v: _num(v, defaults[c]))
    if "favorite_flag" in df.columns:
        df["favorite_flag"] = df["favorite_flag"].apply(_flag).astype(int)
    else:
        df["favorite_flag"] = 0

    # compute expected IP (vectorized via apply)
    df["expected_ip"] = df.apply(
        lambda r: project_expected_ip(
            _num(r.get("last5_pitch_ct_mean"), defaults["last5_pitch_ct_mean"]),
            _num(r.get("days_rest"), defaults["days_rest"]),
            _num(r.get("leash_bias"), defaults["leash_bias"]),
            _flag(r.get("favorite_flag")),
            _num(r.get("bullpen_freshness"), defaults["bullpen_freshness"])
        ),
        axis=1
    )

    # ----- Ks -----
    mask_ks = (df["market_type"] == "PITCHER_KS") & df["alt_line"].notna()
    df.loc[mask_ks, "mu_ks"] = df[mask_ks].apply(
        lambda r: project_ks_mu(
            r["expected_ip"],
            _num(r.get("pitcher_k_rate"), defaults["pitcher_k_rate"]),
            _num(r.get("opp_k_rate"), defaults["opp_k_rate"]),
            _num(r.get("park_k_factor"), defaults["park_k_factor"]),
            _num(r.get("ump_k_bias"), defaults["ump_k_bias"])
        ),
        axis=1
    )
    def _ks_prob(r):
        line = _num(r.get("alt_line"), np.nan)
        if pd.isna(line): return np.nan
        p_over = prob_over_count(r["mu_ks"], 0.20, line)
        return p_over if str(r.get("side","")).upper()=="OVER" else (1.0 - prob_over_count(r["mu_ks"], 0.20, line-1.0))
    df.loc[mask_ks, "q_proj"] = df[mask_ks].apply(_ks_prob, axis=1)

    # ----- Walks -----
    mask_bb = (df["market_type"] == "PITCHER_WALKS") & df["alt_line"].notna()
    df.loc[mask_bb, "mu_bb"] = df[mask_bb].apply(
        lambda r: project_bb_mu(
            r["expected_ip"],
            _num(r.get("pitcher_bb_rate"), defaults["pitcher_bb_rate"]),
            _num(r.get("opp_bb_rate"), defaults["opp_bb_rate"]),
            _num(r.get("ump_k_bias"), 1.0)
        ),
        axis=1
    )
    def _bb_prob(r):
        line = _num(r.get("alt_line"), np.nan)
        if pd.isna(line): return np.nan
        p_over = prob_over_count(r["mu_bb"], 0.25, line)
        return p_over if str(r.get("side","")).upper()=="OVER" else (1.0 - prob_over_count(r["mu_bb"], 0.25, line-1.0))
    df.loc[mask_bb, "q_proj"] = df[mask_bb].apply(_bb_prob, axis=1)

    # ----- Outs -----
    mask_outs = (df["market_type"] == "PITCHER_OUTS") & df["alt_line"].notna()
    def _outs_prob(r):
        tails = project_outs_probs(r["expected_ip"], _num(r.get("leash_bias"), 0.0))
        line = _num(r.get("alt_line"), np.nan)
        if pd.isna(line): return np.nan
        # try exact; else nearest half-point
        val = tails.get(line)
        if val is None:
            # nearest half-step among generated keys
            key = min(tails.keys(), key=lambda k: abs(float(k) - float(line)))
            val = tails.get(key, np.nan)
        return val if str(r.get("side","")).upper()=="OVER" else (1.0 - val if pd.notna(val) else np.nan)
    df.loc[mask_outs, "q_proj"] = df[mask_outs].apply(_outs_prob, axis=1)

# ----- Pitcher Win -----
mask_win = df["market_type"].eq("PITCHER_WIN")

# team ML probability (vig-free if provided; fallback to market)
vf_team = df.get("team_ml_vigfree").fillna(df.get("vigfree_p", 0.5)).astype(float)

# probability the starter reaches 5 IP (≈ 15 outs) – use 14.5 as cutoff
p_ip_ge5 = df.apply(
    lambda r: project_outs_probs(
        _num(r.get("expected_ip"), defaults["days_rest"]),    # expected_ip already computed above
        _num(r.get("leash_bias"), 0.0)
    ).get(14.5, 0.6),
    axis=1
)

# Compute win prob ONLY on masked rows, return a Series aligned to mask index
q_win_series = df.loc[mask_win].apply(
    lambda r: project_win_prob(
        vf_team.loc[r.name],
        p_ip_ge5.loc[r.name]
    ),
    axis=1
)

# Assign back to masked rows
df.loc[mask_win, "q_proj"] = q_win_series

# If side is NO, flip the probability on those masked rows only
mask_win_no = mask_win & df["side"].str.upper().eq("NO")
df.loc[mask_win_no, "q_proj"] = 1.0 - df.loc[mask_win_no, "q_proj"]