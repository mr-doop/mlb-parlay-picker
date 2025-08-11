# etl/fetch_and_build.py
from __future__ import annotations
import os, io, requests, pandas as pd
from .sources.mlb_schedule import fetch_schedule
from .sources.probable_pitchers import fetch_probable_pitchers
from .sources.opp_rates import load_opp_rates

ODDS_BASE = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"

def _get_api_key() -> str:
    k = os.getenv("ODDS_API_KEY", "")
    if not k:
        raise RuntimeError("Missing ODDS_API_KEY in environment/secrets")
    return k

def _fetch_dk_board(date_str: str) -> pd.DataFrame:
    """Fetch DK board (H2H + spreads). Fill run-line alt_line from 'point'."""
    params = {
        "apiKey": _get_api_key(),
        "regions": "us",
        "markets": "h2h,spreads",
        "oddsFormat": "american",
        "bookmakers": "draftkings",
        "dateFormat": "iso",
    }
    r = requests.get(ODDS_BASE, params=params, timeout=20)
    if r.status_code == 401:
        raise RuntimeError("Unauthorized from The Odds API (check key/plan)")
    r.raise_for_status()
    data = r.json()

    rows = []
    for ev in data:
        home = ev.get("home_team","")
        away = ev.get("away_team","")
        gid = f"{date_str}-{home.replace(' ','')}-{away.replace(' ','')}"
        for bm in ev.get("bookmakers", []):
            if bm.get("key") != "draftkings":
                continue
            for mk in bm.get("markets", []):
                key = mk.get("key")
                for out in mk.get("outcomes", []):
                    price = int(out.get("price", 0))
                    team  = out.get("name","")
                    if key == "h2h":
                        rows.append({
                            "date": date_str, "game_id": gid, "market_type": "MONEYLINE",
                            "team": team, "side": "YES", "alt_line": None, "american_odds": price,
                            "player_id": None, "player_name": None
                        })
                    elif key == "spreads":
                        # FIX: set alt_line from 'point'
                        point = out.get("point", None)
                        rows.append({
                            "date": date_str, "game_id": gid, "market_type": "RUN_LINE",
                            "team": team, "side": "YES", "alt_line": point, "american_odds": price,
                            "player_id": None, "player_name": None
                        })
    return pd.DataFrame(rows)

def run(date_str: str):
    """Return (odds_df, feat_df, odds_csv, feat_csv, schedule_df, pitchers_df)."""
    schedule_df = fetch_schedule(date_str)
    try:
        odds_base = _fetch_dk_board(date_str)
    except Exception:
        odds_base = pd.DataFrame(columns=["date","game_id","market_type","team","side","alt_line","american_odds","player_id","player_name"])
    pitchers_df = fetch_probable_pitchers(date_str)
    _ = load_opp_rates()  # ensure cache warm; app will call again

    feat_df = pd.DataFrame({"player_id": [], "q_proj": []})  # placeholder (user upload)
    odds_csv = odds_base.to_csv(index=False).encode("utf-8")
    feat_csv = feat_df.to_csv(index=False).encode("utf-8")
    return odds_base, feat_df, odds_csv, feat_csv, schedule_df, pitchers_df