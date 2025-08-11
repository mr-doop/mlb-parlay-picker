# etl/sources/opp_rates.py
from __future__ import annotations
import pandas as pd
import numpy as np
import datetime as dt

# Fallback (varies by team; values are season-level approximations)
FALLBACK = {
    "ARI": (0.205, 0.085), "ATL": (0.214, 0.089), "BAL": (0.215, 0.082), "BOS": (0.236, 0.078),
    "CHC": (0.243, 0.090), "CWS": (0.230, 0.066), "CIN": (0.245, 0.093), "CLE": (0.199, 0.082),
    "COL": (0.251, 0.073), "DET": (0.249, 0.077), "HOU": (0.188, 0.083), "KC":  (0.215, 0.072),
    "LAA": (0.246, 0.084), "LAD": (0.212, 0.095), "MIA": (0.238, 0.077), "MIL": (0.238, 0.095),
    "MIN": (0.261, 0.092), "NYM": (0.210, 0.090), "NYY": (0.225, 0.101), "OAK": (0.257, 0.082),
    "PHI": (0.221, 0.090), "PIT": (0.235, 0.099), "SDP": (0.203, 0.089), "SFG": (0.231, 0.092),
    "SEA": (0.268, 0.095), "STL": (0.214, 0.093), "TBR": (0.247, 0.086), "TEX": (0.218, 0.092),
    "TOR": (0.208, 0.093), "WSH": (0.216, 0.081)
}

def _try_statcast_window(days: int = 21) -> pd.DataFrame | None:
    try:
        from pybaseball import statcast
        end = dt.date.today()
        start = end - dt.timedelta(days=days)
        df = statcast(start_dt=start.strftime("%Y-%m-%d"), end_dt=end.strftime("%Y-%m-%d"))
        if df is None or df.empty:
            return None
        df = df.assign(events=df["events"].fillna(""))
        # Determine batting team using inning_topbot
        top = df["inning_topbot"].fillna("").str.lower()
        bat_team = np.where(top.str.startswith("top"), df["away_team"], df["home_team"])
        dpa = df.assign(bat_team=bat_team)
        dpa = dpa[dpa["events"] != ""]
        if dpa.empty:
            return None
        g = dpa.groupby("bat_team").agg(
            PA=("events","size"),
            K=("events", lambda s: (s.str.contains("strikeout", case=False, na=False)).sum()),
            BB=("events", lambda s: (s.str.contains("walk", case=False, na=False)).sum())
        ).reset_index().rename(columns={"bat_team":"team_abbr"})
        g = g[g["PA"] > 0]
        g["opp_k_rate"] = g["K"]/g["PA"]
        g["opp_bb_rate"] = g["BB"]/g["PA"]
        return g[["team_abbr","opp_k_rate","opp_bb_rate"]]
    except Exception:
        return None

def load_opp_rates() -> pd.DataFrame:
    stat = _try_statcast_window()
    if stat is not None and not stat.empty:
        # normalize 'SD' to 'SDP' if needed
        stat["team_abbr"] = stat["team_abbr"].replace({"SD":"SDP"})
        return stat
    # fallback
    rows = [{"team_abbr": k, "opp_k_rate": v[0], "opp_bb_rate": v[1]} for k,v in FALLBACK.items()]
    return pd.DataFrame(rows)