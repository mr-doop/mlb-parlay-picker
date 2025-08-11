# etl/fetch_and_build.py
from __future__ import annotations
import os, io, datetime as dt, requests, pandas as pd
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
    """Fetch DK board for H2H + spreads to guarantee games are present. Player props
       may require a paid plan; we keep event coverage robust even without them."""
    params = {
        "apiKey": _get_api_key(),
        "regions": "us",
        "markets": "h2h,spreads",  # conservative to avoid INVALID_MARKET/401 on free plans
        "oddsFormat": "american",
        "bookmakers": "draftkings",
        "dateFormat": "iso"
    }
    r = requests.get(ODDS_BASE, params=params, timeout=20)
    if r.status_code == 401:
        raise RuntimeError("Unauthorized from The Odds API (check key/plan)")
    r.raise_for_status()
    data = r.json()

    rows = []
    for ev in data:
        event_id = ev.get("id","")
        home = ev.get("home_team",""); away = ev.get("away_team","")
        # market odds may be multiple; pick DK
        for bm in ev.get("bookmakers", []):
            if bm.get("key") != "draftkings":
                continue
            # H2H -> moneylines
            for mk in bm.get("markets", []):
                mkt_key = mk.get("key")
                for out in mk.get("outcomes", []):
                    team = out.get("name","")
                    price = int(out.get("price", 0))
                    rows.append({
                        "date": date_str,
                        "game_id": f"{date_str}-{home.replace(' ','')}-{away.replace(' ','')}",
                        "market_type": "MONEYLINE" if mkt_key == "h2h" else "RUN_LINE",
                        "team": team,
                        "side": "YES",
                        "alt_line": None,
                        "american_odds": price,
                        "player_id": None,
                        "player_name": None
                    })
    return pd.DataFrame(rows)

def run(date_str: str) -> tuple[pd.DataFrame, pd.DataFrame, bytes, bytes, pd.DataFrame, pd.DataFrame]:
    """Return (odds_df, feat_df, odds_csv_bytes, feat_csv_bytes, schedule_df, pitchers_df)."""
    # 1) schedule (authoritative game coverage)
    schedule_df = fetch_schedule(date_str)

    # 2) odds (DK) for base markets (ensures DK games we do see have odds)
    try:
        odds_base = _fetch_dk_board(date_str)
    except Exception:
        odds_base = pd.DataFrame(columns=["date","game_id","market_type","team","side","alt_line","american_odds","player_id","player_name"])

    # 3) probable pitchers (drives team mapping for props and helps with display)
    pitchers_df = fetch_probable_pitchers(date_str)

    # 4) baseline features from opponent rates (global; detailed per leg happens in app)
    opp_rates = load_opp_rates()

    # Minimal features DF: we’ll fill more in app once legs are known
    feat_df = pd.DataFrame({"player_id": [], "q_proj": []})  # placeholder

    # CSV bytes for download buttons
    odds_csv = odds_base.to_csv(index=False).encode("utf-8")
    feat_csv = feat_df.to_csv(index=False).encode("utf-8")

    return odds_base, feat_df, odds_csv, feat_csv, schedule_df, pitchers_df