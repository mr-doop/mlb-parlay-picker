# parlay/feature_join.py
from __future__ import annotations
import numpy as np
import pandas as pd

# ---------------- Utilities ----------------

def _player_key(name) -> str:
    """First initial + last name, lowercase; robust to NaN."""
    if not isinstance(name, (str, np.str_)):
        return ""
    s = name.replace(".", "").strip()
    if not s or s.lower() == "nan":
        return ""
    parts = [p for p in s.split() if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return (parts[0][:1] + parts[0]).lower()
    return (parts[0][:1] + parts[-1]).lower()

def _canonical_market(s: str) -> str:
    s = (s or "").strip().lower().replace(" ", "").replace("-", "")
    if s in {"ks","k","strikeouts","pitcherks","pks"}: return "PITCHER_KS"
    if s in {"outs","pitcherouts","pouts"}:           return "PITCHER_OUTS"
    if s in {"bb","walks","pitcherbb","pbb"}:         return "PITCHER_WALKS"
    if s in {"win","pitchertowin","pitcherwin"}:      return "PITCHER_WIN"
    if s in {"ml","moneyline"}:                       return "MONEYLINE"
    if s in {"runline","spread","rl"}:                return "RUN_LINE"
    if s in {"altrunline","alternativerunline","arl"}:return "ALT_RUN_LINE"
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
    """Return df[name] if it exists; otherwise a Series of 'default' with same length/index."""
    if name in df.columns:
        s = df[name]
    else:
        fill = default
        if fill is None:
            fill = np.nan
        s = pd.Series([fill] * len(df), index=df.index)
    if dtype is not None:
        try:
            s = s.astype(dtype)
        except Exception:
            pass
    return s

# ---------------- Public API ----------------

def normalize_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize arbitrary user/auto features into the canonical schema:

        ['player_key','team_abbr','market_type','side','alt_line','q_proj']

    Notes:
      • Any missing columns are backfilled with safe defaults (empty strings / NaN).
      • 'q_proj' is required; clipped to [0.01, 0.99].
      • Accepts aliases: player/name, team/team_code, market/type/bet_type,
        side/ou/over_under, line/threshold/point, prob/q/probability.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    rename = {
        "player": "player_name",
        "name": "player_name",

        "team": "team_abbr",
        "team_code": "team_abbr",

        "market": "market_type",
        "type": "market_type",
        "bet_type": "market_type",

        "side": "side",
        "ou": "side",
        "over_under": "side",

        "line": "alt_line",
        "threshold": "alt_line",
        "point": "alt_line",

        "prob": "q_proj",
        "q": "q_proj",
        "probability": "q_proj",
    }
    # Only rename when target not already present
    for src, dst in rename.items():
        if src in df.columns and dst not in df.columns:
            df.rename(columns={src: dst}, inplace=True)

    # ---- Build normalized columns safely ----
    # player_key from player_name (fallback to empty-string Series so .apply works)
    names = _series(df, "player_name", default="", dtype="object")
    df["player_key"] = names.apply(_player_key)

    # team_abbr normalized to 3-letter uppercase
    if "team_abbr" in df.columns:
        df["team_abbr"] = _series(df, "team_abbr", default="", dtype="object").str.upper().str[:3]
    else:
        df["team_abbr"] = _series(df, "team_abbr", default="", dtype="object")

    # market, side, line
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
        df["alt_line"] = _series(df, "alt_line")  # NaNs

    # q_proj (REQUIRED)
    if "q_proj" in df.columns:
        df["q_proj"] = pd.to_numeric(df["q_proj"], errors="coerce").clip(0.01, 0.99)
    else:
        # No probabilities = cannot attach anything
        return pd.DataFrame()

    keep = ["player_key","team_abbr","market_type","side","alt_line","q_proj"]
    out = df[keep].copy()
    out = out.dropna(subset=["q_proj"]).reset_index(drop=True)
    return out

def attach_projections(pool_df: pd.DataFrame, feat_norm: pd.DataFrame):
    """
    Progressively attach q_proj to pool_df and recompute edge/EV.
    Join order:
      1) player_key + market + side + alt_line
      2) player_key + market + side
      3) team_abbr  + market + side + alt_line
      4) team_abbr  + market + side

    Robustness:
      - Keep 'old_q_model' as a COLUMN before merges → avoids Series misalignment.
      - Use np.isclose to count applied changes.
    """
    if pool_df is None or pool_df.empty:
        return pool_df.copy(), 0

    df = pool_df.copy().reset_index(drop=True)

    if "q_model" not in df.columns:
        base = df.get("p_market", pd.Series(0.50, index=df.index))
        df["q_model"] = pd.to_numeric(base, errors="coerce").fillna(0.50).clip(0.01, 0.99)

    df["old_q_model"] = df["q_model"].astype(float)

    if feat_norm is None or feat_norm.empty:
        df["edge"] = (df["q_model"] - df["p_market"]).fillna(0.0)
        df["ev"]   = (df["q_model"] * df["decimal_odds"] - 1.0).fillna(0.0)
        return df.drop(columns=["old_q_model"], errors="ignore"), 0

    if "player_key" not in df.columns:
        df["player_key"] = ""

    df["q_proj_tmp"] = np.nan

    def _join(keys):
        nonlocal df
        if not set(keys).issubset(feat_norm.columns) or not set(keys).issubset(df.columns):
            return
        rhs = feat_norm[keys + ["q_proj"]].dropna(subset=["q_proj"]).drop_duplicates()
        df = df.merge(rhs, on=keys, how="left")
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

    df.drop(columns=["q_proj_tmp","old_q_model"], inplace=True, errors="ignore")
    df["edge"] = (df["q_model"] - df["p_market"]).fillna(0.0)
    df["ev"]   = (df["q_model"] * df["decimal_odds"] - 1.0).fillna(0.0)

    return df, applied