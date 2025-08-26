# parlay/feature_join.py
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from typing import Dict, Iterable, Tuple

# ============================== Utils ===============================

def _player_key(name) -> str:
    """First initial + last name, lowercase; robust to NaN."""
    if not isinstance(name, (str, np.str_)):
        return ""
    s = name.replace(".", "").strip()
    if not s or s.lower() == "nan":
        return ""
    parts = [p for p in s.split() if p]
    if len(parts) == 1:
        return (parts[0][:1] + parts[0]).lower()
    return (parts[0][:1] + parts[-1]).lower()

def _canonical_market(s: str) -> str:
    s = (s or "").strip().lower().replace(" ", "").replace("-", "")
    if s in {"ks","k","strikeouts","pitcherks","pks"}:   return "PITCHER_KS"
    if s in {"outs","pitcherouts","pouts"}:             return "PITCHER_OUTS"
    if s in {"bb","walks","pitcherbb","pbb"}:           return "PITCHER_WALKS"
    if s in {"win","pitchertowin","pitcherwin"}:        return "PITCHER_WIN"
    if s in {"ml","moneyline"}:                         return "MONEYLINE"
    if s in {"runline","spread","rl"}:                  return "RUN_LINE"
    if s in {"altrunline","alternativerunline","arl"}:  return "ALT_RUN_LINE"
    return s.upper()

def _canonical_side(s: str) -> str:
    s = (s or "").strip().lower()
    if s in {"o","over"}:  return "OVER"
    if s in {"u","under"}: return "UNDER"
    if s in {"yes","y"}:   return "YES"
    if s in {"no","n"}:    return "NO"
    return s.upper()

def _to_float(x):
    try:
        if isinstance(x, str):
            x = x.replace("+", "")
        return float(x)
    except Exception:
        return np.nan

def _series(df: pd.DataFrame, name: str, default=None, dtype=None) -> pd.Series:
    """Return df[name] if it exists; otherwise same-length Series with 'default'."""
    if name in df.columns:
        s = df[name]
    else:
        fill = default if default is not None else np.nan
        s = pd.Series([fill] * len(df), index=df.index)
    if dtype is not None:
        try: s = s.astype(dtype)
        except Exception: pass
    return s

def _american_to_decimal(american: pd.Series) -> pd.Series:
    a = pd.to_numeric(american, errors="coerce")
    pos = 1 + (a.where(a > 0) / 100.0)
    neg = 1 + (100.0 / a.abs().where(a < 0))
    return pos.fillna(neg)

def _clamp_prob(s: pd.Series, lo=0.01, hi=0.99) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").clip(lo, hi)

def _safe_logit(p: pd.Series) -> pd.Series:
    p = _clamp_prob(p)
    return np.log(p / (1.0 - p))

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# ======================= Feature normalization ======================

def normalize_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize arbitrary user/auto features into:
        ['player_key','team_abbr','market_type','side','alt_line','q_proj']
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    rename = {
        "player": "player_name", "name": "player_name",
        "team": "team_abbr", "team_code": "team_abbr",
        "market": "market_type", "type": "market_type", "bet_type": "market_type",
        "side": "side", "ou": "side", "over_under": "side",
        "line": "alt_line", "threshold": "alt_line", "point": "alt_line",
        "prob": "q_proj", "q": "q_proj", "probability": "q_proj",
    }
    for src, dst in rename.items():
        if src in df.columns and dst not in df.columns:
            df.rename(columns={src: dst}, inplace=True)

    # Required/derived
    df["player_key"] = _series(df, "player_name", default="", dtype="object").apply(_player_key)
    if "team_abbr" in df.columns:
        df["team_abbr"] = _series(df, "team_abbr", default="", dtype="object").str.upper().str[:3]
    else:
        df["team_abbr"] = _series(df, "team_abbr", default="", dtype="object")

    if "market_type" in df.columns:
        df["market_type"] = _series(df, "market_type", default="", dtype="object").apply(_canonical_market)
    else:
        df["market_type"] = _series(df, "market_type", default="", dtype="object")

    if "side" in df.columns:
        df["side"] = _series(df, "side", default="", dtype="object").apply(_canonical_side)
    else:
        df["side"] = _series(df, "side", default="", dtype="object")

    if "alt_line" in df.columns:
        df["alt_line"] = _series(df, "alt_line").apply(_to_float)
    else:
        df["alt_line"] = _series(df, "alt_line")

    if "q_proj" in df.columns:
        df["q_proj"] = pd.to_numeric(df["q_proj"], errors="coerce").clip(0.01, 0.99)
    else:
        return pd.DataFrame()

    keep = ["player_key","team_abbr","market_type","side","alt_line","q_proj"]
    out = df[keep].dropna(subset=["q_proj"]).reset_index(drop=True)
    return out

# ===================== Attach feature projections ====================

def attach_projections(pool_df: pd.DataFrame, feat_norm: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Merge q_proj onto pool_df progressively:
      1) player_key + market + side + alt_line
      2) player_key + market + side
      3) team_abbr  + market + side + alt_line
      4) team_abbr  + market + side
    Returns (df, applied_count).
    """
    if pool_df is None or pool_df.empty:
        return pool_df.copy(), 0

    df = pool_df.copy().reset_index(drop=True)

    # p_market
    if "p_market" not in df.columns:
        if "decimal_odds" in df.columns:
            dec = pd.to_numeric(df["decimal_odds"], errors="coerce")
            df["p_market"] = _clamp_prob(1.0 / dec.replace(0, np.nan)).fillna(0.5)
        else:
            dec = _american_to_decimal(df.get("american_odds", np.nan))
            df["p_market"] = _clamp_prob(1.0 / dec.replace(0, np.nan)).fillna(0.5)
    else:
        df["p_market"] = _clamp_prob(df["p_market"]).fillna(0.5)

    if "q_model" not in df.columns:
        df["q_model"] = df["p_market"].copy()
    else:
        df["q_model"] = _clamp_prob(df["q_model"]).fillna(df["p_market"])

    df["old_q_model"] = df["q_model"].astype(float)

    if feat_norm is None or feat_norm.empty:
        if "decimal_odds" not in df.columns:
            df["decimal_odds"] = _american_to_decimal(df.get("american_odds", np.nan))
        df["edge"] = (df["q_model"] - df["p_market"]).fillna(0.0)
        df["ev"]   = (df["q_model"] * df["decimal_odds"] - 1.0).fillna(0.0)
        return df.drop(columns=["old_q_model"], errors="ignore"), 0

    if "player_key" not in df.columns:
        df["player_key"] = _series(df, "player_name", default="", dtype="object").apply(_player_key)

    df["q_proj_tmp"] = np.nan

    def _join(keys: Iterable[str]):
        nonlocal df
        if not set(keys).issubset(feat_norm.columns) or not set(keys).issubset(df.columns):
            return
        rhs = feat_norm[list(keys) + ["q_proj"]].dropna(subset=["q_proj"]).drop_duplicates()
        df = df.merge(rhs, on=list(keys), how="left")
        df["q_proj_tmp"] = df["q_proj_tmp"].combine_first(df["q_proj"])
        df.drop(columns=["q_proj"], inplace=True)

    _join(["player_key","market_type","side","alt_line"])
    _join(["player_key","market_type","side"])
    _join(["team_abbr","market_type","side","alt_line"])
    _join(["team_abbr","market_type","side"])

    df["q_proj_tmp"] = df["q_proj_tmp"].clip(0.01, 0.99)
    df["q_model"] = df["q_proj_tmp"].where(df["q_proj_tmp"].notna(), df["q_model"])

    old = df["old_q_model"].astype(float).to_numpy()
    new = df["q_model"].astype(float).to_numpy()
    applied = int(np.sum(~np.isclose(new, old, rtol=1e-05, atol=1e-08, equal_nan=True)))

    if "decimal_odds" not in df.columns:
        df["decimal_odds"] = _american_to_decimal(df.get("american_odds", np.nan))
    df.drop(columns=["q_proj_tmp","old_q_model"], inplace=True, errors="ignore")
    df["edge"] = (df["q_model"] - df["p_market"]).fillna(0.0)
    df["ev"]   = (df["q_model"] * df["decimal_odds"] - 1.0).fillna(0.0)

    return df, applied

# ================== Logistic enhancement (q_final) ===================

REFS: Dict[str, float] = {
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

DEFAULT_WEIGHTS: Dict[str, float] = {
    "intercept":        -0.10,
    "opp_k_pct":         1.50,
    "opp_bb_pct":       -1.20,
    "pitch_mix_fit":     0.50,
    "exp_ip":            0.80,
    "exp_pitches":       0.40,
    "park_k_factor":     0.30,
    "park_bb_factor":   -0.20,
    "weather_factor":    0.20,
    "bullpen_fatigue":  -0.30,
    "form_k9":           0.60,
    "form_bb9":         -0.60,
}

def _load_weights(path: str | None) -> Dict[str, float]:
    if not path: return DEFAULT_WEIGHTS
    try:
        with open(path, "r") as f:
            w = json.load(f)
        return {**DEFAULT_WEIGHTS, **w}
    except Exception:
        return DEFAULT_WEIGHTS

def _ensure_market_inputs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "p_market" not in out.columns:
        if "decimal_odds" in out.columns:
            dec = pd.to_numeric(out["decimal_odds"], errors="coerce")
            out["p_market"] = _clamp_prob(1.0 / dec.replace(0, np.nan)).fillna(0.5)
        else:
            dec = _american_to_decimal(out.get("american_odds", np.nan))
            out["p_market"] = _clamp_prob(1.0 / dec.replace(0, np.nan)).fillna(0.5)
    else:
        out["p_market"] = _clamp_prob(out["p_market"]).fillna(0.5)

    if "q_model" not in out.columns:
        out["q_model"] = out["p_market"].copy()
    else:
        out["q_model"] = _clamp_prob(out["q_model"]).fillna(out["p_market"])

    if "decimal_odds" not in out.columns:
        out["decimal_odds"] = _american_to_decimal(out.get("american_odds", np.nan))
    return out

def apply_enhanced_model(pool_df: pd.DataFrame, weights_path: str | None = "parlay/model_weights.json", blend: float = 0.35) -> pd.DataFrame:
    """
    Produces q_final from available feature columns:
    opp_k_pct, opp_bb_pct, pitch_mix_fit, exp_ip, exp_pitches,
    park_k_factor, park_bb_factor, weather_factor, bullpen_fatigue,
    form_k9, form_bb9
    """
    if pool_df is None or pool_df.empty:
        return pool_df
    df = _ensure_market_inputs(pool_df)
    W = _load_weights(weights_path)

    idx = df.index
    def f(name: str) -> pd.Series:
        base = REFS[name]
        s = pd.to_numeric(df.get(name, pd.Series(base, index=idx)), errors="coerce").fillna(base)
        return s - base

    z = pd.Series(W.get("intercept", 0.0), index=idx, dtype="float64")
    z += W["opp_k_pct"]       * f("opp_k_pct")
    z += W["opp_bb_pct"]      * f("opp_bb_pct")
    z += W["pitch_mix_fit"]   * f("pitch_mix_fit")
    z += W["exp_ip"]          * f("exp_ip")
    z += W["exp_pitches"]     * (f("exp_pitches") / 20.0)
    z += W["park_k_factor"]   * f("park_k_factor")
    z += W["park_bb_factor"]  * f("park_bb_factor")
    z += W["weather_factor"]  * f("weather_factor")
    z += W["bullpen_fatigue"] * f("bullpen_fatigue")
    z += W["form_k9"]         * (f("form_k9") / 2.0)
    z += W["form_bb9"]        * f("form_bb9")

    q_lr = _sigmoid(z)
    base_q = df["q_model"]
    logit_blend = (1.0 - blend) * _safe_logit(q_lr) + blend * _safe_logit(base_q)
    q_final = _sigmoid(logit_blend)

    df["q_lr"]    = _clamp_prob(q_lr)
    df["q_final"] = _clamp_prob(q_final)
    df["edge"]    = (df["q_final"] - df["p_market"]).fillna(0.0)
    df["ev"]      = (df["q_final"] * df["decimal_odds"] - 1.0).fillna(0.0)
    return df

def apply_projections_and_enhancements(pool_df: pd.DataFrame, feat_norm: pd.DataFrame | None, weights_path: str | None = "parlay/model_weights.json", blend: float = 0.35) -> Tuple[pd.DataFrame, int]:
    df, applied = attach_projections(pool_df, feat_norm)
    df = apply_enhanced_model(df, weights_path=weights_path, blend=blend)
    return df, applied