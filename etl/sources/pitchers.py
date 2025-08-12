# etl/sources/pitchers.py
from __future__ import annotations
import requests
import numpy as np
import pandas as pd

def _player_key(name: str) -> str:
    if not isinstance(name, str): return ""
    s = name.replace(".", "").strip()
    if not s: return ""
    parts = [p for p in s.split() if p]
    if len(parts) == 1: return (parts[0][:1] + parts[0]).lower()
    return (parts[0][:1] + parts[-1]).lower()

def load_probable_pitchers(date_str: str) -> pd.DataFrame:
    """
    Use MLB Stats API to fetch probable pitchers for the date.
    Returns columns: game_id (filled later via join), player_key, team_abbr, player_name, home_abbr, away_abbr
    """
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=probablePitcher,team"
    try:
        j = requests.get(url, timeout=20).json()
    except Exception:
        return pd.DataFrame(columns=["game_id","player_key","team_abbr","player_name","home_abbr","away_abbr"])

    rows = []
    for d in j.get("dates", []) or []:
        for g in d.get("games", []) or []:
            home_team = g.get("teams", {}).get("home", {}).get("team", {}) or {}
            away_team = g.get("teams", {}).get("away", {}).get("team", {}) or {}
            home_abbr = (home_team.get("abbreviation") or home_team.get("teamCode") or "").upper()
            away_abbr = (away_team.get("abbreviation") or away_team.get("teamCode") or "").upper()

            # home probable
            hp = g.get("teams", {}).get("home", {}).get("probablePitcher", {}) or {}
            if hp.get("fullName"):
                rows.append({
                    "game_id": np.nan,  # joined later against DK schedule via abbrs
                    "player_key": _player_key(hp["fullName"]),
                    "team_abbr": home_abbr,
                    "player_name": hp["fullName"],
                    "home_abbr": home_abbr,
                    "away_abbr": away_abbr,
                })
            # away probable
            ap = g.get("teams", {}).get("away", {}).get("probablePitcher", {}) or {}
            if ap.get("fullName"):
                rows.append({
                    "game_id": np.nan,
                    "player_key": _player_key(ap["fullName"]),
                    "team_abbr": away_abbr,
                    "player_name": ap["fullName"],
                    "home_abbr": home_abbr,
                    "away_abbr": away_abbr,
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        # De-dupe by name/team
        df = df.drop_duplicates(subset=["player_key","team_abbr"]).reset_index(drop=True)
    return df