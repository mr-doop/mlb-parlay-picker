# parlay/feature_join.py
from __future__ import annotations
import numpy as np
import pandas as pd

# ---------------- Utilities ----------------

def _player_key(name) -> str:
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

# ---------------- Public API ----------------

def normalize_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize arbitrary user/auto features into:
      ['player_key','team_abbr','market_type','side','alt_line','q_proj']
    q_proj is clipped to [0.01, 0.99].
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    rename = {
        "player":"player_name", "name":"player_name",
        "team":"team_abbr", "team_code":"team_abbr",
        "market":"market_type", "type":"market_type", "bet_type":"market_type",
        "side":"side", "ou":"side", "over_under":"side",
        "line":"alt_line", "threshold":"alt_line", "point":"alt_line",
        "prob":"q_proj", "q":"q_proj", "probability":"q_proj",
    }
    for k, v in rename.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)

    # Derive normalized fields
    df["player_key"] = df.get("player_name", "").apply(_player_key)
    if "team_abbr" in df.columns:
        df["team_abbr"] = df["team_abbr"].astype(str).str.upper().str[:3]
    if "market_type" in df.columns:
        df["market_type"] = df["market_type"].apply(_canonical_market)
    if "side" in df.columns:
        df["side"] = df["side"].apply(_canonical_side)
    if "alt_line" in df.columns:
        df["alt_line"] = df["alt_line"].apply(_to_float)
    if "q_proj" in df.columns:
        df["q_proj"] = pd.to_numeric(df["q_proj"], errors="coerce").clip(0.01, 0.99)

    keep = [c for c in ["player_key","team_abbr","market_type","side","alt_line","q_proj"] if c in df.columns]
    df = df[keep].dropna(how="all")

    # Must have q
    if "q_proj" not in df.columns:
        return pd.DataFrame()

    df = df[df["q_proj"].notna()].reset_index(drop=True)
    return df

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
        # Fall back gracefully if missing
        base = df.get("p_market", pd.Series(0.50, index=df.index))
        df["q_model"] = pd.to_numeric(base, errors="coerce").fillna(0.50).clip(0.01, 0.99)

    # Snapshot BEFORE any merges to preserve alignment
    df["old_q_model"] = df["q_model"].astype(float)

    if feat_norm is None or feat_norm.empty:
        df["edge"] = (df["q_model"] - df["p_market"]).fillna(0.0)
        df["ev"]   = (df["q_model"] * df["decimal_odds"] - 1.0).fillna(0.0)
        applied = 0
        return df.drop(columns=["old_q_model"], errors="ignore"), applied

    if "player_key" not in df.columns:
        df["player_key"] = ""

    df["q_proj_tmp"] = np.nan

    def _join(keys):
        nonlocal df
        if not set(keys).issubset(feat_norm.columns) or not set(keys).issubset(df.columns):
            return
        rhs = (
            feat_norm[keys + ["q_proj"]]
            .dropna(subset=["q_proj"])
            .drop_duplicates()
        )
        # Merge will duplicate rows when multiple matches exist; that's okay because
        # 'old_q_model' duplicates with them, keeping row-wise comparison consistent.
        df = df.merge(rhs, on=keys, how="left")
        df["q_proj_tmp"] = df["q_proj_tmp"].combine_first(df["q_proj"])
        df.drop(columns=["q_proj"], inplace=True)

    _join(["player_key","market_type","side","alt_line"])
    _join(["player_key","market_type","side"])
    _join(["team_abbr","market_type","side","alt_line"])
    _join(["team_abbr","market_type","side"])

    # Apply and clip
    df["q_proj_tmp"] = df["q_proj_tmp"].clip(0.01, 0.99)
    df["q_model"] = df["q_proj_tmp"].where(df["q_proj_tmp"].notna(), df["q_model"])

    # Count rows where q changed (handle float tolerance and NaNs)
    old = df["old_q_model"].astype(float).to_numpy()
    new = df["q_model"].astype(float).to_numpy()
    applied = int(np.sum(~np.isclose(new, old, rtol=1e-05, atol=1e-08, equal_nan=True)))

    # Cleanup + recompute economics
    df.drop(columns=["q_proj_tmp","old_q_model"], inplace=True, errors="ignore")
    df["edge"] = (df["q_model"] - df["p_market"]).fillna(0.0)
    df["ev"]   = (df["q_model"] * df["decimal_odds"] - 1.0).fillna(0.0)

    return df, applied