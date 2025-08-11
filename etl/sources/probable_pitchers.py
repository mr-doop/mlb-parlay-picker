# etl/sources/probable_pitchers.py
from __future__ import annotations
import datetime as dt
import requests
import pandas as pd
import re

def _tok(s: str) -> str:
    return re.sub(r'[^A-Za-z]', '', s or '')

def _player_key(full_name: str) -> str:
    # "Taijuan Walker" -> "twalker"; "Jose A. Quintana" -> "jquintana"
    name = (full_name or "").strip().replace(".", "")
    if not name:
        return ""
    parts = [p for p in name.split() if p]
    if len(parts) == 1:
        return parts[0][:1].lower() + parts[0].lower()
    return (parts[0][:1] + parts[-1]).lower()

def fetch_probable_pitchers(date_str: str) -> pd.DataFrame:
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {"sportId": 1, "date": date_str, "hydrate": "probablePitcher"}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    rows = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            home = g["teams"]["home"]; away = g["teams"]["away"]
            home_team = home["team"].get("name",""); home_ab = home["team"].get("abbreviation","")
            away_team = away["team"].get("name",""); away_ab = away["team"].get("abbreviation","")
            date = date_str
            gid = f"{date}-{_tok(home_team)}-{_tok(away_team)}"
            # away pitcher
            ap = away.get("probablePitcher")
            if ap and ap.get("fullName"):
                rows.append({
                    "game_id": gid, "team_abbr": away_ab, "player_name": ap["fullName"],
                    "player_key": _player_key(ap["fullName"])
                })
            # home pitcher
            hp = home.get("probablePitcher")
            if hp and hp.get("fullName"):
                rows.append({
                    "game_id": gid, "team_abbr": home_ab, "player_name": hp["fullName"],
                    "player_key": _player_key(hp["fullName"])
                })
    return pd.DataFrame(rows)