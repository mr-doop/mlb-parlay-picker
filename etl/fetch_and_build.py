# etl/fetch_and_build.py
# ---------------------------------------------------------------------
# Fetch DraftKings MLB markets + pitcher props from The Odds API,
# normalize into a single CSV, then build a features CSV using
# public sources (weather, park factors, opponent rates, bullpen).
#
# Requires a Streamlit Secret or env var:
#   ODDS_API_KEY = "<your_the_odds_api_key>"
#
# Output:
#   dk_markets_YYYY-MM-DD.csv
#   features_YYYY-MM-DD.csv
# ---------------------------------------------------------------------

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import csv
import os
import time
import datetime as dt

import requests
import pandas as pd
import streamlit as st

# ------------------------- API constants ------------------------------
SPORT = "baseball_mlb"
ROOT = "https://api.the-odds-api.com/v4"

# Game-level markets (pull once for the slate)
GAME_MARKETS = "h2h,spreads"

# Event-level player prop markets (must be fetched per event)
# Valid MLB keys (no 'player_' prefix here)
PROP_MARKETS = ",".join([
    "pitcher_strikeouts",
    "pitcher_strikeouts_alternate",
    "pitcher_outs",
    "pitcher_walks",
    "pitcher_walks_alternate",
    "pitcher_record_a_win",
])

# ------------------------- CSV row type -------------------------------
@dataclass
class DKRow:
    date: str
    game_id: str
    market_type: str         # MONEYLINE / RUN_LINE / ALT_RUN_LINE / PITCHER_KS / PITCHER_OUTS / PITCHER_WALKS / PITCHER_WIN
    side: str                # e.g. HOME/AWAY for ML, Over/Under/Yes/No for props, or team name for run lines
    team: str                # team name (for spreads/ML), blank for most props
    player_id: str           # stable id from player_name (lowercase last_firstInitial)
    player_name: str         # display name from Odds API
    alt_line: str            # numeric line for alt/props (e.g., 4.5 Ks, 17.5 outs, +1.5 RL); empty if NA
    american_odds: int       # DK price

# ------------------------- helpers -----------------------------------
def _key() -> str:
    """Get API key from Streamlit Secrets or environment."""
    try:
        return str(st.secrets["ODDS_API_KEY"])
    except Exception:
        k = os.getenv("ODDS_API_KEY")
        if not k:
            raise RuntimeError("ODDS_API_KEY missing -- add it in Streamlit → Manage app → Settings → Secrets")
        return str(k)

def _get(url: str, params: Dict[str, Any]) -> Any:
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def _norm_game_id(date: str, home: str, away: str) -> str:
    # "YYYY-MM-DD-HomeTeamNoSpaces-AwayTeamNoSpaces"
    return f"{date}-{home.replace(' ', '')}-{away.replace(' ', '')}"

def _pid_from_name(name: str) -> str:
    # crude stable id: last_firstInitial
    parts = [p for p in name.replace(".", "").split(" ") if p]
    if not parts:
        return name.lower()
    first = parts[0]
    last = parts[-1]
    return f"{last.lower()}_{first[0].lower()}"

# ------------------------- API calls ---------------------------------
def fetch_games(date: str) -> List[Dict[str, Any]]:
    """
    Get slate with DK game markets (h2h, spreads).
    The Odds API /odds returns upcoming events; we filter client-side by date.
    """
    url = f"{ROOT}/sports/{SPORT}/odds"
    params = {
        "apiKey": _key(),
        "regions": "us",
        "markets": GAME_MARKETS,
        "oddsFormat": "american",
        "bookmakers": "draftkings",
    }
    j = _get(url, params)
    # Keep only games whose commence_time date matches `date` (UTC vs local caveat--good enough for MVP)
    keep = []
    d_req = pd.to_datetime(date).date()
    for g in j:
        try:
            d = pd.to_datetime(g.get("commence_time")).date()
            if d == d_req:
                keep.append(g)
        except Exception:
            keep.append(g)
    return keep

def fetch_props_for_event(event_id: str) -> Any:
    """Fetch DK props for a single event via event-level ODDS endpoint."""
    url = f"{ROOT}/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": _key(),
        "regions": "us",
        "markets": PROP_MARKETS,
        "oddsFormat": "american",
        "bookmakers": "draftkings",
    }
    return _get(url, params)

def _iter_dk_markets_from_event_response(event_odds: Any):
    """
    Normalize The Odds API event-level response into (market_key, outcomes) for DraftKings only.
    Supports both possible shapes we encounter.
    """
    # Shape A: list[ { key, bookmakers: [ { key, markets: [ { key, outcomes } ] } ] } ]
    if isinstance(event_odds, list):
        for m in event_odds:
            base_key = str(m.get("key", "")).lower()
            for bm in m.get("bookmakers", []):
                if bm.get("key") != "draftkings":
                    continue
                for mk in bm.get("markets", []):
                    mkey = str(mk.get("key", base_key)).lower()
                    outs = mk.get("outcomes", [])
                    if outs:
                        yield mkey, outs
        return
    # Shape B: dict { bookmakers: [ { key, markets: [ ... ] } ] }
    if isinstance(event_odds, dict):
        for bm in event_odds.get("bookmakers", []):
            if bm.get("key") != "draftkings":
                continue
            for mk in bm.get("markets", []):
                mkey = str(mk.get("key", "")).lower()
                outs = mk.get("outcomes", [])
                if outs:
                    yield mkey, outs

# ------------------------- row builder -------------------------------
def build_rows(date: str, games: List[Dict[str, Any]]) -> List[DKRow]:
    rows: List[DKRow] = []

    for g in games:
        event_id = g.get("id") or g.get("event_id")
        home = g.get("home_team", "HOME")
        away = g.get("away_team", "AWAY")
        gid = _norm_game_id(date, home, away)

        # 1) Game markets (ML + spreads)
        for bm in g.get("bookmakers", []):
            if bm.get("key") != "draftkings":
                continue
            for mk in bm.get("markets", []):
                mkey = str(mk.get("key", "")).lower()
                if mkey == "h2h":
                    for out in mk.get("outcomes", []):
                        side = "HOME" if out.get("name") == home else "AWAY"
                        rows.append(DKRow(
                            date, gid, "MONEYLINE", side,
                            out.get("name", ""), "", "", "",
                            int(out.get("price", 0))
                        ))
                elif mkey == "spreads":
                    for out in mk.get("outcomes", []):
                        spread = out.get("point")
                        team = out.get("name", "")
                        mt = "RUN_LINE" if (spread is not None and abs(float(spread)) == 1.5) else "ALT_RUN_LINE"
                        rows.append(DKRow(
                            date, gid, mt, team, team,
                            "", "", str(spread) if spread is not None else "",
                            int(out.get("price", 0))
                        ))

        # 2) Event-level props (Ks/Outs/Walks/Win)
        if not event_id:
            continue
        try:
            props = fetch_props_for_event(event_id)
        except requests.HTTPError as e:
            # 422 often means market missing for this event -- warn and continue
            st.warning(f"Props fetch failed for {gid}: {e}")
            time.sleep(0.15)
            continue

        for mkey, outcomes in _iter_dk_markets_from_event_response(props):
            lk = mkey.lower()
            for out in outcomes:
                name = str(out.get("name", ""))            # 'Over'/'Under'/'Yes'/'No'
                desc = str(out.get("description", ""))     # player name
                label = desc or name
                price = int(out.get("price", 0))
                point = out.get("point", None)

                if "strikeout" in lk:
                    side = "OVER" if "over" in name.lower() else ("UNDER" if "under" in name.lower() else "")
                    rows.append(DKRow(
                        date, gid, "PITCHER_KS", side, "",
                        _pid_from_name(label), label,
                        str(point) if point is not None else "", price
                    ))
                elif "outs" in lk:
                    side = "OVER" if "over" in name.lower() else ("UNDER" if "under" in name.lower() else "")
                    rows.append(DKRow(
                        date, gid, "PITCHER_OUTS", side, "",
                        _pid_from_name(label), label,
                        str(point) if point is not None else "", price
                    ))
                elif "walks" in lk:
                    side = "OVER" if "over" in name.lower() else ("UNDER" if "under" in name.lower() else "")
                    rows.append(DKRow(
                        date, gid, "PITCHER_WALKS", side, "",
                        _pid_from_name(label), label,
                        str(point) if point is not None else "", price
                    ))
                elif "record_a_win" in lk or "to_record_a_win" in lk or "pitcher_record_a_win" in lk:
                    side = "YES" if "yes" in name.lower() else ("NO" if "no" in name.lower() else "")
                    rows.append(DKRow(
                        date, gid, "PITCHER_WIN", side, "",
                        _pid_from_name(label), label,
                        "", price
                    ))
        time.sleep(0.10)  # be gentle on rate limits

    return rows

# ------------------------- IO helpers --------------------------------
def write_csv(path: str, rows: List[DKRow]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date","game_id","market_type","side","team","player_id","player_name","alt_line","american_odds"])
        for r in rows:
            w.writerow([r.date, r.game_id, r.market_type, r.side, r.team, r.player_id, r.player_name, r.alt_line, r.american_odds])

# ------------------------- public entrypoint --------------------------
def run(date: str) -> Tuple[str, str]:
    """
    Build both CSVs for the given date.
    Returns: (odds_csv_path, features_csv_path)
    """
    # 1) DK odds CSV
    games = fetch_games(date)
    rows = build_rows(date, games)
    odds_path = f"dk_markets_{date}.csv"
    write_csv(odds_path, rows)

    # 2) Features CSV (weather, park, opp, bullpen)
    from etl.feature_builder import build_features
    try:
        dk_df = pd.read_csv(odds_path)
        feats = build_features(date, dk_df)
        feat_path = f"features_{date}.csv"
        feats.to_csv(feat_path, index=False)
    except Exception as e:
        # Fail-safe: write an empty skeleton so the app can proceed
        st.warning(f"Feature build failed, writing skeleton features: {e}")
        feat_path = f"features_{date}.csv"
        pd.DataFrame(columns=["player_id"]).to_csv(feat_path, index=False)

    return odds_path, feat_path

# Allow running from CLI for quick tests on Streamlit Cloud shell
if __name__ == "__main__":
    today = dt.date.today().strftime("%Y-%m-%d")
    o, f = run(today)
    print("Wrote:", o, f)