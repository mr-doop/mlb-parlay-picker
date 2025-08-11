# etl/sources/mlb_schedule.py
from __future__ import annotations
import datetime as dt
import requests
import pandas as pd
import re

def _tok(s: str) -> str:
    return re.sub(r'[^A-Za-z]', '', s or '')

def _gid(date_str: str, home_name: str, away_name: str) -> str:
    # Consistent, human-readable game_id similar to prior format
    return f"{date_str}-{_tok(home_name)}-{_tok(away_name)}"

def fetch_schedule(date_str: str) -> pd.DataFrame:
    """Return all MLB games for date with abbreviations and derived game_id."""
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {"sportId": 1, "date": date_str, "hydrate": "probablePitcher,team,linescore"}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            home = g["teams"]["home"]["team"]
            away = g["teams"]["away"]["team"]
            home_nm = home.get("name", "")
            away_nm = away.get("name", "")
            home_ab = home.get("abbreviation", "") or home.get("teamName", "")
            away_ab = away.get("abbreviation", "") or away.get("teamName", "")
            gid = _gid(date_str, home_nm, away_nm)
            games.append({
                "date": date_str,
                "game_id": gid,
                "home_name": home_nm,
                "away_name": away_nm,
                "home_abbr": home_ab,
                "away_abbr": away_ab,
                "matchup": f"{away_ab}@{home_ab}",
            })
    df = pd.DataFrame(games)
    return df