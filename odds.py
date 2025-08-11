from __future__ import annotations
import numpy as np
import pandas as pd

def american_to_decimal(a: int | float) -> float:
    a = float(a)
    if a > 0:
        return 1.0 + (a / 100.0)
    else:
        return 1.0 + (100.0 / abs(a))

def implied_prob_from_american(a: int | float) -> float:
    a = float(a)
    if a > 0:
        return 100.0 / (a + 100.0)
    else:
        return abs(a) / (abs(a) + 100.0)

def vig_free_binary(p1: float, p2: float) -> tuple[float, float]:
    s = p1 + p2
    if s <= 0:
        return p1, p2
    return p1 / s, p2 / s

def compute_vig_free_probs(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each binary market pair, compute vig-free implied probability for each side.
    Assumes df has columns: game_id, market_type, side, alt_line, team, player_id, american_odds.
    This function returns a copy with columns: implied_p, vigfree_p, decimal_odds.
    If a pair is missing, vigfree_p falls back to implied_p.
    """
    df = df.copy()
    df["implied_p"] = df["american_odds"].apply(implied_prob_from_american)
    df["decimal_odds"] = df["american_odds"].apply(american_to_decimal)
    df["vigfree_p"] = df["implied_p"]  # fallback

    # Build pair keys by market type
    def pair_key(row):
        mt = row["market_type"]
        gid = row["game_id"]
        if mt in {"PITCHER_KS", "PITCHER_OUTS", "PITCHER_WALKS"}:
            return ("PROP", gid, mt, row.get("player_id", None), row.get("alt_line", None))
        elif mt == "PITCHER_WIN":
            return ("WIN", gid, mt, row.get("player_id", None), None)
        elif mt in {"MONEYLINE"}:
            return ("ML", gid, mt, None, None)
        elif mt in {"RUN_LINE", "ALT_RUN_LINE"}:
            # pair by abs(spread)
            spread = row.get("alt_line", None)
            spread_abs = abs(float(spread)) if spread is not None else None
            return ("RL", gid, mt, None, spread_abs)
        else:
            return ("OTHER", gid, mt, None, row.get("alt_line", None))

    df["_pair_key"] = df.apply(pair_key, axis=1)

    # For each pair key, attempt vig removal using two sides
    grouped = df.groupby("_pair_key", dropna=False)
    rows = []
    for key, g in grouped:
        g = g.copy()
        if len(g) >= 2:
            # try to identify the two complementary sides
            # For ML: expect HOME / AWAY
            # For Props: OVER / UNDER
            # For Win: YES / NO
            # For Run Lines: two teams on same abs(spread)
            sides = g["side"].str.upper().tolist()
            if "HOME" in sides and "AWAY" in sides:
                p_home = g.loc[g["side"].str.upper()=="HOME", "implied_p"].iloc[0]
                p_away = g.loc[g["side"].str.upper()=="AWAY", "implied_p"].iloc[0]
                vf_home, vf_away = vig_free_binary(p_home, p_away)
                # assign
                g.loc[g["side"].str.upper()=="HOME", "vigfree_p"] = vf_home
                g.loc[g["side"].str.upper()=="AWAY", "vigfree_p"] = vf_away
            elif "OVER" in sides and "UNDER" in sides:
                p_over = g.loc[g["side"].str.upper()=="OVER", "implied_p"].iloc[0]
                p_under = g.loc[g["side"].str.upper()=="UNDER", "implied_p"].iloc[0]
                vf_over, vf_under = vig_free_binary(p_over, p_under)
                g.loc[g["side"].str.upper()=="OVER", "vigfree_p"] = vf_over
                g.loc[g["side"].str.upper()=="UNDER", "vigfree_p"] = vf_under
            elif "YES" in sides and "NO" in sides:
                p_yes = g.loc[g["side"].str.upper()=="YES", "implied_p"].iloc[0]
                p_no = g.loc[g["side"].str.upper()=="NO", "implied_p"].iloc[0]
                vf_yes, vf_no = vig_free_binary(p_yes, p_no)
                g.loc[g["side"].str.upper()=="YES", "vigfree_p"] = vf_yes
                g.loc[g["side"].str.upper()=="NO", "vigfree_p"] = vf_no
            else:
                # For run lines: two teams on same abs(spread) — just take top two
                if key[0] == "RL" and g["team"].nunique() >= 2:
                    teams = g["team"].unique()[:2]
                    p1 = g.loc[g["team"]==teams[0], "implied_p"].iloc[0]
                    p2 = g.loc[g["team"]==teams[1], "implied_p"].iloc[0]
                    vf1, vf2 = vig_free_binary(p1, p2)
                    g.loc[g["team"]==teams[0], "vigfree_p"] = vf1
                    g.loc[g["team"]==teams[1], "vigfree_p"] = vf2
                # else leave fallback
        rows.append(g)
    out = pd.concat(rows, ignore_index=True).drop(columns=["_pair_key"])
    return out
