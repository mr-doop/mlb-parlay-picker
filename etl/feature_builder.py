# etl/feature_builder.py
# Stitches all feature sources and returns a tidy per-player feature frame.

from functools import reduce
import pandas as pd

from etl.sources import weather, park_factors, opp_rates, bullpen
from etl.sources import pitcher_form

DEFAULTS = {
    "pitcher_k_rate": 0.24,
    "pitcher_bb_rate": 0.08,
    "opp_k_rate": 0.22,
    "opp_bb_rate": 0.08,
    "last5_pitch_ct_mean": 90.0,
    "days_rest": 5.0,
    "leash_bias": 0.0,
    "favorite_flag": 0,
    "bullpen_freshness": 6.0,
    "park_k_factor": 1.00,
    "park_run_factor": 1.00,
    "ump_k_bias": 0.00,
    "team_ml_vigfree": 0.50,
}

def _left_join(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    if a is None or a.empty: return b
    if b is None or b.empty: return a
    return a.merge(b, on="player_id", how="left")

def fill_defaults(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c, v in DEFAULTS.items():
        if c not in out.columns: out[c] = v
        out[c] = out[c].fillna(v)
    out["favorite_flag"] = out["favorite_flag"].astype(int)
    return out

def build_features(date: str, dk_df: pd.DataFrame) -> pd.DataFrame:
    # base: list of all player_ids that appear in any prop market
    base = dk_df[dk_df["player_id"].notna()][["player_id"]].drop_duplicates()

    frames = [base]
    try: frames.append(weather.build(date, dk_df))
    except Exception: pass
    try: frames.append(park_factors.build(date, dk_df))
    except Exception: pass
    try: frames.append(opp_rates.build(date, dk_df))
    except Exception: pass
    try: frames.append(bullpen.build(date, dk_df))
    except Exception: pass
    try: frames.append(pitcher_form.build(date, dk_df))   # <-- add this line
    except Exception: pass

    feats = reduce(_left_join, frames)
    feats = fill_defaults(feats)
    return feats