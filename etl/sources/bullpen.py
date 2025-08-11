# etl/sources/bullpen.py
# Placeholder bullpen freshness. Ready to be wired to a real source later.

import pandas as pd

def build(date: str, dk_df: pd.DataFrame) -> pd.DataFrame:
    # Broadcast neutral bullpen values to all player_ids in the slate.
    m = dk_df[dk_df["player_id"].notna()][["player_id"]].drop_duplicates()
    if m.empty:
        return pd.DataFrame(columns=["player_id","bullpen_freshness","bullpen_ip_1d","bullpen_ip_3d"])
    m["bullpen_freshness"] = 6.0   # lower = fresher pen
    m["bullpen_ip_1d"] = 2.0
    m["bullpen_ip_3d"] = 6.0
    return m