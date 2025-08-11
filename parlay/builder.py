# parlay/builder.py
from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

from .odds import american_to_decimal

# ---- helpers ---------------------------------------------------------

def _combo_decimal(legs: List[pd.Series]) -> float:
    dec = 1.0
    for r in legs:
        d = float(r.get("decimal_odds", american_to_decimal(int(r["american_odds"]))))
        dec *= d
    return dec

def _combo_q_independent(legs: List[pd.Series]) -> float:
    q = 1.0
    for r in legs:
        q *= float(r.get("q_model", r.get("p_market", 0.5)))
    return q

def _score_row(r: pd.Series, weight_ev: float = 0.5) -> float:
    # blend safety and value
    q = float(r.get("q_model", r.get("p_market", 0.5)))
    ev = float(r.get("ev", 0.0))
    return q + weight_ev * max(ev, 0.0)

def _diversify_mask(df: pd.DataFrame, chosen: List[pd.Series]) -> pd.Series:
    if not chosen: return pd.Series([True]*len(df), index=df.index)
    used_players = {str(r.get("player_id","")) for r in chosen if pd.notna(r.get("player_id"))}
    used_games   = {str(r.get("game_id","")) for r in chosen if pd.notna(r.get("game_id"))}
    used_types   = {str(r.get("market_type","")) for r in chosen if pd.notna(r.get("market_type"))}

    m = pd.Series(True, index=df.index)
    if used_players: m &= ~df["player_id"].astype(str).isin(used_players)
    if used_games:   m &= ~df["game_id"].astype(str).isin(used_games)
    # allow duplicates in type but not all same; cap to <= legs/2
    for t in used_types:
        cap = max(1, len(chosen)//2)
        if sum(1 for r in chosen if str(r.get("market_type",""))==t) >= cap:
            m &= (df["market_type"] != t)
    return m

def _pick_greedy(pool: pd.DataFrame, legs: int, weight_ev: float) -> List[pd.Series]:
    picks: List[pd.Series] = []
    df = pool.copy()
    while len(picks) < legs and not df.empty:
        mask = _diversify_mask(df, picks)
        cand = df[mask].copy()
        if cand.empty:
            cand = df.copy()  # relax diversification if needed
        cand["__score"] = cand.apply(_score_row, axis=1, weight_ev=weight_ev)
        best_idx = cand.sort_values(["__score","q_model","decimal_odds"], ascending=[False,False,False]).index[0]
        picks.append(df.loc[best_idx])
        df = df.drop(index=[best_idx])
    return picks

def _pool_for_tier(df: pd.DataFrame, tier: str) -> pd.DataFrame:
    d = df.copy()
    if tier == "Low":
        # safer: high q, usually negative odds
        return d[(d["q_model"] >= 0.70) & (d["american_odds"] <= -100)]
    if tier == "Medium":
        # balance: decent q, allow mix; ensure not all heavy favorites
        return d[(d["q_model"] >= 0.62)]
    if tier == "High":
        # value hunting: allow more +money, but keep q >= 0.55
        return d[(d["q_model"] >= 0.55)]
    return d

def build_presets(pool: pd.DataFrame, legs_list=(4,5,6,8), min_parlay_am="+600") -> Dict[Tuple[int,str], Dict]:
    """
    Returns dict keyed by (legs, tier) with:
      { "legs": [rows], "decimal": float, "q_est": float, "meets_min": bool, "label": str }
    """
    # compute decimal odds if missing
    if "decimal_odds" not in pool.columns:
        pool = pool.copy()
        pool["decimal_odds"] = pool["american_odds"].apply(lambda x: american_to_decimal(int(x)))

    # minimum parlay decimal given American +600:
    min_dec = 1 + (abs(int(min_parlay_am.replace("+",""))) / 100.0)  # 7.0

    out: Dict[Tuple[int,str], Dict] = {}
    # keep track of picks used so tiers differ
    used_global = set()

    for legs in legs_list:
        for tier, w in [("Low", 0.25), ("Medium", 0.50), ("High", 0.80)]:
            pool_t = _pool_for_tier(pool, tier).copy()
            if pool_t.empty:
                out[(legs,tier)] = {"legs": [], "decimal": 1.0, "q_est": 0.0, "meets_min": False, "label": f"{tier}"}
                continue

            # de-duplicate globally so Medium/High aren't identical to Low
            pool_t = pool_t[~pool_t.index.isin(list(used_global))].copy() if used_global else pool_t

            picks = _pick_greedy(pool_t, legs, weight_ev=w)
            if not picks:
                out[(legs,tier)] = {"legs": [], "decimal": 1.0, "q_est": 0.0, "meets_min": False, "label": f"{tier}"}
                continue

            for p in picks:
                used_global.add(p.name)

            dec = _combo_decimal(picks)
            q   = _combo_q_independent(picks)
            out[(legs,tier)] = {"legs": picks, "decimal": dec, "q_est": q, "meets_min": (dec >= min_dec), "label": tier}
    return out