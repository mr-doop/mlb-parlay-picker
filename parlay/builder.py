# parlay/builder.py
from __future__ import annotations
from typing import List, Dict, Tuple, Set
import numpy as np
import pandas as pd
from .odds import american_to_decimal

# ---------- odds helpers ----------
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
    q  = float(r.get("q_model", r.get("p_market", 0.5)))
    ev = float(r.get("ev", 0.0))
    return q + weight_ev * max(ev, 0.0)

def _sig(r: pd.Series) -> Tuple:
    return (str(r.get("player_id","")), str(r.get("market_type","")), str(r.get("side","")),
            float(r.get("alt_line", np.nan)), str(r.get("game_id","")))

def _diversify_mask(df: pd.DataFrame, chosen: List[pd.Series], used_global: Set[Tuple]) -> pd.Series:
    if not chosen and not used_global:
        return pd.Series(True, index=df.index)
    m = pd.Series(True, index=df.index)
    # Avoid reusing exact same leg (across tiers)
    if used_global:
        m &= ~df.apply(lambda r: _sig(r) in used_global, axis=1)
    # Avoid same player and same game inside a single card
    if chosen:
        used_players = {str(r.get("player_id","")) for r in chosen}
        used_games   = {str(r.get("game_id","")) for r in chosen}
        if "player_id" in df: m &= ~df["player_id"].astype(str).isin(used_players)
        if "game_id"   in df: m &= ~df["game_id"].astype(str).isin(used_games)
        # cap dominant market type
        used_types = {str(r.get("market_type","")) for r in chosen}
        for t in used_types:
            cap = max(1, len(chosen)//2)
            if sum(1 for r in chosen if str(r.get("market_type",""))==t) >= cap:
                m &= (df["market_type"] != t)
    return m

def _pick_greedy(pool: pd.DataFrame, legs: int, weight_ev: float, used_global: Set[Tuple]) -> List[pd.Series]:
    picks: List[pd.Series] = []
    df = pool.copy()
    while len(picks) < legs and not df.empty:
        mask = _diversify_mask(df, picks, used_global)
        cand = df[mask].copy()
        if cand.empty:
            cand = df.copy()  # relax if too strict
        cand["__score"] = cand.apply(_score_row, axis=1, weight_ev=weight_ev)
        best_idx = cand.sort_values(["__score","q_model","decimal_odds"], ascending=[False,False,False]).index[0]
        picks.append(df.loc[best_idx])
        used_global.add(_sig(df.loc[best_idx]))
        df = df.drop(index=[best_idx])
    return picks

def _pool_for_tier(df: pd.DataFrame, tier: str) -> pd.DataFrame:
    d = df.copy()
    if tier == "Low":    return d[(d["q_model"] >= 0.70) & (d["american_odds"] <= -100)]
    if tier == "Medium": return d[(d["q_model"] >= 0.62)]
    if tier == "High":   return d[(d["q_model"] >= 0.55)]
    return d

def build_presets(pool: pd.DataFrame, legs_list=(4,5,6,8), min_parlay_am="+600",
                  odds_min=None, odds_max=None) -> Dict[Tuple[int,str], Dict]:
    # odds range filter
    df = pool.copy()
    if odds_min is not None and odds_max is not None:
        df = df[(df["american_odds"] >= odds_min) & (df["american_odds"] <= odds_max)]
    if "decimal_odds" not in df.columns:
        df["decimal_odds"] = df["american_odds"].apply(lambda x: american_to_decimal(int(x)))

    min_dec = 1 + (abs(int(str(min_parlay_am).replace("+",""))) / 100.0)  # +600 => 7.0
    out: Dict[Tuple[int,str], Dict] = {}
    used_global: Set[Tuple] = set()

    for legs in legs_list:
        for tier, w in [("Low", 0.25), ("Medium", 0.50), ("High", 0.80)]:
            pool_t = _pool_for_tier(df, tier).copy()
            if pool_t.empty:
                out[(legs,tier)] = {"legs": [], "decimal": 1.0, "q_est": 0.0, "meets_min": False, "label": tier}
                continue
            picks = _pick_greedy(pool_t, legs, weight_ev=w, used_global=used_global)
            if not picks:
                out[(legs,tier)] = {"legs": [], "decimal": 1.0, "q_est": 0.0, "meets_min": False, "label": tier}
                continue
            dec = _combo_decimal(picks)
            q   = _combo_q_independent(picks)
            out[(legs,tier)] = {"legs": picks, "decimal": dec, "q_est": q, "meets_min": (dec >= min_dec), "label": tier}
    return out

def build_one_tap(pool: pd.DataFrame, legs: int, risk: str, odds_min=None, odds_max=None) -> Dict:
    # risk in {"Low","Medium","High"}
    return build_presets(pool, legs_list=(legs,), min_parlay_am="+600",
                         odds_min=odds_min, odds_max=odds_max)[(legs, risk)]