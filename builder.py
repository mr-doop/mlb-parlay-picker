from __future__ import annotations
import numpy as np
import pandas as pd

# Greedy parlay builder (cross-game only, MVP)
def build_parlay_greedy(df: pd.DataFrame,
                        target_decimal_odds: float = 6.0,
                        min_legs: int = 4,
                        max_legs: int = 7,
                        mode: str = "SAFETY") -> pd.DataFrame:
    """
    df must include: game_id, description, decimal_odds, q (probability), market_type.
    mode: "SAFETY" (sort by q desc) or "VALUE" (sort by q*ln(odds)).
    Cross-game: at most 1 leg per game_id.
    """
    pool = df.copy()
    pool = pool[pool["decimal_odds"] > 1.0].dropna(subset=["q"])

    if mode.upper() == "VALUE":
        pool["score"] = pool["q"] * np.log(pool["decimal_odds"])
    else:
        # SAFETY
        pool["score"] = pool["q"] + 1e-6 * np.log(pool["decimal_odds"])

    pool = pool.sort_values(["score", "q", "decimal_odds"], ascending=[False, False, False])

    chosen = []
    used_games = set()
    total_odds = 1.0

    for _, row in pool.iterrows():
        gid = row["game_id"]
        if gid in used_games:
            continue
        # add leg
        chosen.append(row)
        used_games.add(gid)
        total_odds *= row["decimal_odds"]
        if total_odds >= target_decimal_odds and len(chosen) >= min_legs:
            break
        if len(chosen) >= max_legs:
            break

    if len(chosen) < min_legs:
        # try relaxing cross-game rule minimally (optional): here we keep MVP strict.
        pass

    picks = pd.DataFrame(chosen)
    if not picks.empty:
        est_hit_prob = picks["q"].prod()
    else:
        est_hit_prob = np.nan

    picks.attrs["total_decimal_odds"] = total_odds
    picks.attrs["est_hit_prob"] = est_hit_prob
    return picks
