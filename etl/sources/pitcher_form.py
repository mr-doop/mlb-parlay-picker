# etl/sources/pitcher_form.py
# Pull pitcher-level recent form using pybaseball: last 30 days Statcast.
# Provides: last5_pc_mean, last3_pc_mean, days_rest, pitcher_k_rate, pitcher_bb_rate, leash_bias (derived).

import pandas as pd
import numpy as np
from typing import Optional, Dict

def _clean_name(n: str) -> Dict[str, str]:
    n = (n or "").replace(".", "").replace(",", " ").strip()
    # remove suffixes
    for suf in [" Jr", " Jr.", " II", " III", " IV", " Sr", " Sr."]:
        if n.endswith(suf): n = n[: -len(suf)]
    parts = [p for p in n.split() if p]
    if not parts: return {"first":"", "last":""}
    if len(parts) == 1: return {"first":"", "last":parts[0]}
    return {"first":parts[0], "last":parts[-1]}

def _lookup_bamid(name: str) -> Optional[int]:
    try:
        from pybaseball import playerid_lookup
        nm = _clean_name(name)
        df = playerid_lookup(nm["last"], nm["first"])
        if df is None or df.empty: return None
        # choose the most recent debut / active if available
        df = df.sort_values(["mlb_played_last","mlb_played_first"], ascending=[False, False])
        return int(df.iloc[0]["key_mlbam"])
    except Exception:
        return None

def _recent_statcast_pitcher(pid: int, start: str, end: str) -> Optional[pd.DataFrame]:
    try:
        from pybaseball import statcast_pitcher
        df = statcast_pitcher(start_dt=start, end_dt=end, player_id=pid)
        if df is None or df.empty: return None
        # keep only columns we need
        keep = ["game_date","events","description","pitch_type"]
        for k in keep:
            if k not in df.columns:
                df[k] = np.nan
        return df[keep].copy()
    except Exception:
        return None

def _rates_from_events(df: pd.DataFrame) -> Dict[str, float]:
    # Plate appearances ≈ rows with a terminal event (walk/strikeout/hit/out/etc.)
    ev = df["events"].fillna("")
    is_pa = ev != ""
    pa = int(is_pa.sum()) if is_pa is not None else 0
    if pa == 0:
        return {"pitcher_k_rate": 0.24, "pitcher_bb_rate": 0.08}
    k = ev.str.contains("strikeout", case=False, na=False).sum()
    bb = ev.str.contains("walk", case=False, na=False).sum()
    return {
        "pitcher_k_rate": max(0.05, min(0.45, k / pa)),
        "pitcher_bb_rate": max(0.02, min(0.20, bb / pa)),
    }

def _days_between(d1: pd.Timestamp, d2: pd.Timestamp) -> float:
    return float((d1.normalize() - d2.normalize()).days)

def build(date: str, dk_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-player pitcher form features for all players in dk_df with props."""
    # Work off players who actually have props in the slate.
    players = dk_df[dk_df["player_id"].notna()][["player_id","player_name"]].drop_duplicates()

    rows = []
    end = pd.to_datetime(date)
    start = (end - pd.Timedelta(days=35)).strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")

    for _, r in players.iterrows():
        pid = str(r["player_id"])
        name = str(r["player_name"] or "")

        # try lookup
        bam = _lookup_bamid(name)
        if bam is None:
            # no match → safe defaults; you'll still get weather/park/etc.
            rows.append({
                "player_id": pid,
                "last5_pc_mean": 90.0,
                "last3_pc_mean": 90.0,
                "days_rest": 5.0,
                "pitcher_k_rate": 0.24,
                "pitcher_bb_rate": 0.08,
                "leash_bias": 0.0,
            })
            continue

        sc = _recent_statcast_pitcher(bam, start, end_s)
        if sc is None or sc.empty:
            rows.append({
                "player_id": pid,
                "last5_pc_mean": 90.0,
                "last3_pc_mean": 90.0,
                "days_rest": 5.0,
                "pitcher_k_rate": 0.24,
                "pitcher_bb_rate": 0.08,
                "leash_bias": 0.0,
            })
            continue

        # per-game pitch counts (count rows per game_date)
        pc_by_game = sc.groupby("game_date").size().sort_index()
        last3_pc = pc_by_game.tail(3).astype(float)
        last5_pc = pc_by_game.tail(5).astype(float)
        last3_pc_mean = float(last3_pc.mean()) if not last3_pc.empty else 90.0
        last5_pc_mean = float(last5_pc.mean()) if not last5_pc.empty else last3_pc_mean

        # days rest since most recent game
        try:
            last_game = pd.to_datetime(pc_by_game.index[-1])
            days_rest = _days_between(end, last_game)
            # if there was a start 4–6 days earlier, rest ~4–6; else clamp
            days_rest = max(2.0, min(9.0, days_rest))
        except Exception:
            days_rest = 5.0

        rates = _rates_from_events(sc)

        # derive leash bias: above/below 90 pitches → ± up to ~0.25 IP
        leash_bias = max(-0.3, min(0.3, (last5_pc_mean - 90.0) / 100.0))

        rows.append({
            "player_id": pid,
            "last5_pc_mean": last5_pc_mean,
            "last3_pc_mean": last3_pc_mean,
            "days_rest": days_rest,
            "pitcher_k_rate": rates["pitcher_k_rate"],
            "pitcher_bb_rate": rates["pitcher_bb_rate"],
            "leash_bias": leash_bias,
        })

    return pd.DataFrame(rows)