# etl/sources/schedule.py
from __future__ import annotations
import requests
import pandas as pd
from datetime import datetime, timezone

DK_EVENTGROUP_MLB = 84240

TEAM_ABBR = {
    "arizona diamondbacks":"ARI","atlanta braves":"ATL","baltimore orioles":"BAL",
    "boston red sox":"BOS","chicago cubs":"CHC","chicago white sox":"CWS",
    "cincinnati reds":"CIN","cleveland guardians":"CLE","colorado rockies":"COL",
    "detroit tigers":"DET","houston astros":"HOU","kansas city royals":"KC",
    "los angeles angels":"LAA","los angeles dodgers":"LAD","miami marlins":"MIA",
    "milwaukee brewers":"MIL","minnesota twins":"MIN","new york mets":"NYM",
    "new york yankees":"NYY","oakland athletics":"OAK","philadelphia phillies":"PHI",
    "pittsburgh pirates":"PIT","san diego padres":"SDP","san francisco giants":"SFG",
    "seattle mariners":"SEA","st. louis cardinals":"STL","tampa bay rays":"TBR",
    "texas rangers":"TEX","toronto blue jays":"TOR","washington nationals":"WSH",
    "st louis cardinals":"STL","la angels":"LAA","la dodgers":"LAD","tampa bay":"TBR",
    "san diego":"SDP","san francisco":"SFG"
}

def _abbr(name: str) -> str:
    n = (name or "").strip().lower()
    return TEAM_ABBR.get(n, (n[:3] or "").upper())

def _matchup(away: str, home: str) -> str:
    a = _abbr(away); h = _abbr(home)
    return f"{a}@{h}"

def load_schedule_for_date(date_str: str) -> pd.DataFrame:
    """
    Pull schedule for the given date from DraftKings event group JSON.
    Returns columns: date, game_id, home_abbr, away_abbr, matchup
    """
    url = f"https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/{DK_EVENTGROUP_MLB}?format=json"
    headers = {"User-Agent":"Mozilla/5.0"}
    try:
        j = requests.get(url, headers=headers, timeout=25).json()
    except Exception:
        return pd.DataFrame(columns=["date","game_id","home_abbr","away_abbr","matchup"])

    evs = (j.get("eventGroup", {}) or {}).get("events", []) or []
    rows = []
    for e in evs:
        start = (e.get("startDate") or "")[:10]
        if start != date_str:
            continue
        home = e.get("homeTeamName") or e.get("homeTeam", {}).get("name")
        away = e.get("awayTeamName") or e.get("awayTeam", {}).get("name")
        rows.append({
            "date": date_str,
            "game_id": e.get("eventId"),
            "home_abbr": _abbr(home),
            "away_abbr": _abbr(away),
            "matchup": _matchup(away, home),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates().sort_values("matchup").reset_index(drop=True)
    return df