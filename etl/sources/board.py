# etl/sources/board.py
from __future__ import annotations
import requests
import numpy as np
import pandas as pd

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

def _american_to_decimal(a):
    try:
        a = int(a)
    except Exception:
        return np.nan
    if a < 0: return 1 + 100/abs(a)
    return 1 + a/100.0

def load_dk_board(date_str: str) -> pd.DataFrame:
    """
    Pulls Moneyline and Run Line (incl. alternates) from DraftKings.
    Returns normalized columns used by the app.
    """
    url = f"https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/{DK_EVENTGROUP_MLB}?format=json"
    headers = {"User-Agent":"Mozilla/5.0"}
    try:
        j = requests.get(url, headers=headers, timeout=25).json()
    except Exception:
        return pd.DataFrame()

    eg = j.get("eventGroup", {}) or {}
    events = eg.get("events", []) or []

    # Map eventId -> (home, away)
    ev_map = {}
    for e in events:
        if (e.get("startDate") or "")[:10] != date_str:
            continue
        home = e.get("homeTeamName") or e.get("homeTeam", {}).get("name")
        away = e.get("awayTeamName") or e.get("awayTeam", {}).get("name")
        ev_map[e.get("eventId")] = dict(home=_abbr(home), away=_abbr(away))

    def _market_kind(name: str):
        s = (name or "").lower()
        if "moneyline" in s: return "MONEYLINE"
        if "run line" in s: return "RUN_LINE"
        if "alternate run line" in s or "alt run line" in s: return "ALT_RUN_LINE"
        return None

    rows = []
    for cat in eg.get("offerCategories", []) or []:
        for d in cat.get("offerSubcategoryDescriptors", []) or []:
            kind = _market_kind(d.get("name",""))
            if kind not in {"MONEYLINE","RUN_LINE","ALT_RUN_LINE"}:
                continue
            offersets = (d.get("offerSubcategory", {}) or {}).get("offers", []) or []
            for offerset in offersets:
                for offer in (offerset or []):
                    ev_id = offer.get("eventId")
                    teams = ev_map.get(ev_id)
                    if not teams:
                        continue
                    outcomes = offer.get("outcomes", []) or []
                    for oc in outcomes:
                        # Determine which team this outcome belongs to
                        label = (oc.get("label") or oc.get("name") or "").strip().lower()
                        team_abbr = None
                        # direct team label
                        if label in TEAM_ABBR:
                            team_abbr = TEAM_ABBR[label]
                        # home/away label
                        if not team_abbr and label in {"home","home team"}:
                            team_abbr = teams["home"]
                        if not team_abbr and label in {"away","away team"}:
                            team_abbr = teams["away"]
                        # or try participant/teamAbbreviation if present
                        if not team_abbr:
                            t_abbr = (oc.get("teamAbbreviation") or "").strip().upper()
                            if len(t_abbr) in (2,3):
                                team_abbr = t_abbr

                        american = oc.get("oddsAmerican")
                        alt_line = oc.get("line") if oc.get("line") is not None else offer.get("line")
                        rows.append({
                            "date": date_str,
                            "game_id": ev_id,
                            "home_abbr": teams["home"],
                            "away_abbr": teams["away"],
                            "market_type": kind,
                            "side": "HOME" if team_abbr == teams["home"] else ("AWAY" if team_abbr == teams["away"] else ""),
                            "alt_line": float(alt_line) if alt_line is not None else np.nan,
                            "american_odds": int(str(american).replace("+","")) if american is not None else np.nan,
                            "decimal_odds": _american_to_decimal(american),
                            "team_abbr": team_abbr or "",
                            "player_id": np.nan,
                            "player_name": "",
                            "player_key": "",
                        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Normalize RL alt_line if missing
    mask = df["market_type"].isin(["RUN_LINE","ALT_RUN_LINE"]) & df["alt_line"].isna()
    df.loc[mask, "alt_line"] = 1.5
    # Keep only requested date events
    df = df[df["date"] == date_str].drop_duplicates().reset_index(drop=True)
    return df