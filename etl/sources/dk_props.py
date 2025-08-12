# etl/sources/dk_props.py
from __future__ import annotations
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# MLB event group used by DraftKings site APIs
DK_EVENTGROUP_MLB = 84240

# Map common DK team names -> 3-letter codes we use everywhere
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
    # fallbacks DK sometimes uses
    "st louis cardinals":"STL","la angels":"LAA","la dodgers":"LAD","tampa bay":"TBR",
    "san diego":"SDP","san francisco":"SFG"
}

def _abbr(name: str) -> str:
    n = (name or "").strip().lower()
    return TEAM_ABBR.get(n, (n[:3] or "").upper())

def _player_key(name: str) -> str:
    if not isinstance(name, str): return ""
    s = name.replace(".", "").strip()
    if not s: return ""
    parts = [p for p in s.split() if p]
    if len(parts) == 1: return (parts[0][:1] + parts[0]).lower()
    return (parts[0][:1] + parts[-1]).lower()

def _to_int(x):
    try: return int(str(x).replace("+", "").strip())
    except Exception: return np.nan

def _to_float(x):
    try: return float(x)
    except Exception: return np.nan

def pull_dk_pitcher_props(date_str: str, schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch MLB pitcher props (Ks, Outs, Walks, To Record a Win) directly from DK's public JSON.
    Returns a normalized DataFrame with columns compatible with the app board.
    Note: DK may change this structure at any time; wrapped in try/except to fail safe.
    """
    url = f"https://sportsbook.draftkings.com//sites/US-SB/api/v5/eventgroups/{DK_EVENTGROUP_MLB}?format=json"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        j = requests.get(url, headers=headers, timeout=25).json()
    except Exception:
        return pd.DataFrame()

    eg = j.get("eventGroup", {})
    events = eg.get("events", []) or []

    # eventId -> (home_abbr, away_abbr)
    ev_map = {}
    for e in events:
        home = e.get("homeTeamName") or e.get("homeTeam", {}).get("name")
        away = e.get("awayTeamName") or e.get("awayTeam", {}).get("name")
        ev_map[e.get("eventId")] = dict(home_abbr=_abbr(home), away_abbr=_abbr(away))

    def _market_for(sub_name: str):
        s = (sub_name or "").lower()
        if "strikeout" in s: return "PITCHER_KS"
        if "total outs" in s or "outs " in s or s.endswith(" outs"): return "PITCHER_OUTS"
        if "walk" in s: return "PITCHER_WALKS"
        if "record a win" in s or "to record a win" in s: return "PITCHER_WIN"
        return None

    rows = []
    for cat in eg.get("offerCategories", []):
        for d in cat.get("offerSubcategoryDescriptors", []):
            mkt = _market_for(d.get("name"))
            if not mkt: 
                continue
            offersets = d.get("offerSubcategory", {}).get("offers", []) or []
            for offerset in offersets:
                for offer in (offerset or []):
                    ev_id = offer.get("eventId")
                    ev_info = ev_map.get(ev_id, {})
                    outcomes = offer.get("outcomes", []) or []
                    for oc in outcomes:
                        label = (oc.get("label") or "").strip().lower()
                        if label.startswith("over"): side = "OVER"
                        elif label.startswith("under"): side = "UNDER"
                        elif label == "yes": side = "YES"
                        elif label == "no": side = "NO"
                        else: side = label.upper() or ""
                        player = oc.get("participant") or oc.get("participantName") or offer.get("label") or ""
                        line = oc.get("line") if oc.get("line") is not None else offer.get("line")
                        rows.append({
                            "date": date_str,
                            "game_id": np.nan,  # will fill via schedule join below
                            "home_abbr": ev_info.get("home_abbr"),
                            "away_abbr": ev_info.get("away_abbr"),
                            "market_type": mkt,
                            "side": side,
                            "alt_line": _to_float(line),
                            "american_odds": _to_int(oc.get("oddsAmerican")),
                            "player_name": player,
                            "player_id": oc.get("participantId"),
                            "player_key": _player_key(player),
                        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Attach game_id by matching home/away abbreviations from schedule for today
    if not schedule_df.empty and {"home_abbr","away_abbr"}.issubset(df.columns):
        df = df.merge(
            schedule_df[["game_id","home_abbr","away_abbr"]],
            on=["home_abbr","away_abbr"], how="left"
        )

    # Ensure required columns exist
    needed = ["date","game_id","market_type","side","alt_line","american_odds",
              "player_id","player_name","player_key","home_abbr","away_abbr"]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    # DraftKings sometimes shows alt line as None for "Win" markets; keep NaN.
    return df