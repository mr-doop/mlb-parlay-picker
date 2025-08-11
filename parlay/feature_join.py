# parlay/feature_join.py
from __future__ import annotations
import numpy as np
import pandas as pd
import re

# --------- Helpers ---------
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
    s = (s or "").strip().lower()
    s = s.replace(" ", "").replace("-", "")
    if s in {"ks","k","strikeouts","pitcherks","pks"}:
        return "PITCHER_KS"
    if s in {"outs","pitcherouts","pouts"}:
        return "PITCHER_OUTS"
    if s in {"bb","walks","pitcherbb","pbb"}:
        return "PITCHER_WALKS"
    if s in {"win","pitchertowin","pitcherwin"}:
        return "PITCHER_WIN"
    if s in {"ml","moneyline"}:
        return "MONEYLINE"
    if s in {"runline","spread","rl"}:
        return "RUN_LINE"
    if s in {"altrunline","alternativerunline","arl"}:
        return "ALT_RUN_LINE"
    return s.upper()  # leave custom as-is

def _canonical_side(s: str) -> str:
    s = (s or "").strip().lower()
    if s in {"o","over"}: return "OVER"
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

# --------- Public API ---------
def normalize_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a user features CSV into canonical schema for robust matching."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    # lower-case columns for flexible renaming
    df.columns = [c.lower().strip() for c in df.columns]

    rename = {
        "player":"player_name",
        "name":"player_name",
        "team":"team_abbr",
        "team_code":"team_abbr",
        "market":"market_type",
        "type":"market_type",
        "bet_type":"market_type",
        "side":"side",
        "ou":"side",
        "over_under":"side",
        "line":"alt_line",
        "threshold":"alt_line",
        "prob":"q_proj",
        "q":"q_proj",
        "probability":"q_proj",
    }
    for k,v in rename.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k:v}, inplace=True)

    # Standardize core columns
    if "player_name" in df.columns:
        df["player_key"] = df["player_name"].apply(_player_key)
    else:
        df["player_key"] = ""

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
    # Drop rows without q_proj
    if "q_proj" in df.columns:
        df = df[df["q_proj"].notna()]
    else:
        df = pd.DataFrame()
    return df.reset_index(drop=True)

def attach_projections(pool_df: pd.DataFrame, feat_norm: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Overlay q_proj onto pool_df via progressive joins. Returns (df, applied_count)."""
    if feat_norm is None or feat_norm.empty:
        return pool_df.copy(), 0
    df = pool_df.copy()
    df["player_key"] = df.get("player_key", "").copy() if "player_key" in df.columns else ""
    old_q = df["q_model"].copy()
    df["q_proj_tmp"] = np.nan

    def _join(keys):
        nonlocal df
        if not set(keys).issubset(set(feat_norm.columns)) or not set(keys).issubset(set(df.columns)):
            return
        rhs = feat_norm[keys + ["q_proj"]].dropna(subset=["q_proj"]).drop_duplicates()
        df = df.merge(rhs, on=keys, how="left", suffixes=("",""))
        df["q_proj_tmp"] = df["q_proj_tmp"].combine_first(df["q_proj"])
        df.drop(columns=["q_proj"], inplace=True)

    # Priority: exact player + line -> player no line -> team + line -> team no line
    _join(["player_key","market_type","side","alt_line"])
    _join(["player_key","market_type","side"])
    _join(["team_abbr","market_type","side","alt_line"])
    _join(["team_abbr","market_type","side"])

    df["q_proj_tmp"] = df["q_proj_tmp"].clip(0.01, 0.99)
    df["q_model"] = df["q_proj_tmp"].where(df["q_proj_tmp"].notna(), df["q_model"])
    applied = int((df["q_model"] != old_q).sum())
    df.drop(columns=["q_proj_tmp"], inplace=True, errors="ignore")

    # Recompute economics with updated q_model
    df["edge"] = (df["q_model"] - df["p_market"]).fillna(0.0)
    df["ev"]   = (df["q_model"] * df["decimal_odds"] - 1.0).fillna(0.0)
    return df, applied