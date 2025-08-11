\

import os, sys, csv, time, math, argparse, datetime, requests
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import streamlit as st

# Optional analytics: pybaseball for starters + rates
from pybaseball import schedule_and_record, pitching_game_logs, playerid_lookup
from pybaseball import team_batting

DATE_FMT = "%Y-%m-%d"
TODAY = datetime.date.today().strftime(DATE_FMT)

ODDS_API = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
SPORT = "baseball_mlb"

@dataclass
class DKRow:
    date: str
    game_id: str
    market_type: str
    side: str
    team: str
    player_id: str
    player_name: str
    alt_line: str
    american_odds: int

def american(team_name: str) -> str:
    return team_name

def fetch_dk_odds(date: str) -> List[Dict[str,Any]]:
    load_dotenv()
    key = st.secrets.get("ODDS_API_KEY") or os.getenv("ODDS_API_KEY")
    if not key:
        raise RuntimeError("ODDS_API_KEY missing. Put it in .env")
    params = {
        "apiKey": key,
        "regions": "us",
        "markets": "h2h,spreads,player_props",
        "oddsFormat": "american",
        "bookmakers": "draftkings"
    }
    url = ODDS_API.format(sport=SPORT)
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def normalize_game_id(date: str, home: str, away: str) -> str:
    # Simplify into DATE-HOME-AWAY
    home_code = home.replace(" ", "")
    away_code = away.replace(" ", "")
    return f"{date}-{home_code}-{away_code}"

def map_to_rows(date: str, data: List[Dict[str,Any]]) -> List[DKRow]:
    rows: List[DKRow] = []
    for game in data:
        home = game.get("home_team","HOME")
        away = game.get("away_team","AWAY")
        gid = normalize_game_id(date, home, away)
        for bm in game.get("bookmakers", []):
            if bm.get("key") != "draftkings": 
                continue
            for market in bm.get("markets", []):
                key = market.get("key")
                if key == "h2h":
                    for out in market.get("outcomes", []):
                        side = "HOME" if out.get("name")==home else "AWAY"
                        rows.append(DKRow(date, gid, "MONEYLINE", side, out.get("name",""), "", "", "", int(out.get("price",0))))
                elif key == "spreads":
                    for out in market.get("outcomes", []):
                        spread = out.get("point")
                        team = out.get("name","")
                        mt = "RUN_LINE" if abs(spread)==1.5 else "ALT_RUN_LINE"
                        rows.append(DKRow(date, gid, mt, team, team, "", "", str(spread), int(out.get("price",0))))
                elif key == "player_props":
                    # NOTE: The Odds API encodes many sub-markets; mapping may vary.
                    # We attempt to map some common pitcher props.
                    for out in market.get("outcomes", []):
                        desc = out.get("description","").lower()
                        name = out.get("name","")
                        price = int(out.get("price",0))
                        point = out.get("point", None)
                        # crude parsing hints
                        if "strikeouts" in desc or "k" in desc:
                            # expect name to carry Over/Under, but API shapes vary
                            side = "OVER" if "over" in name.lower() else ("UNDER" if "under" in name.lower() else "")
                            rows.append(DKRow(date, gid, "PITCHER_KS", side, "", name, name, str(point) if point is not None else "", price))
                        elif "outs" in desc:
                            side = "OVER" if "over" in name.lower() else ("UNDER" if "under" in name.lower() else "")
                            rows.append(DKRow(date, gid, "PITCHER_OUTS", side, "", name, name, str(point) if point is not None else "", price))
                        elif "walks" in desc:
                            side = "OVER" if "over" in name.lower() else ("UNDER" if "under" in name.lower() else "")
                            rows.append(DKRow(date, gid, "PITCHER_WALKS", side, "", name, name, str(point) if point is not None else "", price))
                        elif "to record a win" in desc or "to get the win" in desc or "pitcher win" in desc:
                            side = "YES" if "yes" in name.lower() else ("NO" if "no" in name.lower() else "")
                            rows.append(DKRow(date, gid, "PITCHER_WIN", side, "", name, name, "", price))
                # else ignore
    return rows

def write_csv(path: str, rows: List[DKRow]):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date","game_id","market_type","side","team","player_id","player_name","alt_line","american_odds"])
        for r in rows:
            w.writerow([r.date, r.game_id, r.market_type, r.side, r.team, r.player_id, r.player_name, r.alt_line, r.american_odds])

# --- Pitcher features (quick build) ---

def build_pitcher_features_for_date(date: str, starters_hint: Optional[Dict[str,str]]=None) -> str:
    """
    Build a minimal features CSV for the day's primary starters.
    This function uses pybaseball to pull recent logs and rough opponent rates.
    Returns output path.
    """
    # For MVP simplicity, assume you fill player_id manually = last name + first initial lower (e.g., cole_g).
    # Here we can't reliably resolve all IDs without robust mapping.
    # We'll output a template with columns and leave numeric defaults; you can fill a few starters quickly.
    out_path = f"features_{date}.csv"
    cols = ["player_id","pitcher_k_rate","pitcher_bb_rate","opp_k_rate","opp_bb_rate",
            "last5_pitch_ct_mean","days_rest","leash_bias","favorite_flag","bullpen_freshness",
            "park_k_factor","ump_k_bias","team_ml_vigfree"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        # Seed a few template rows (user can fill values)
        w.writerow(["cole_g",0.30,0.07,0.25,0.08,95,5,0.2,1,4.0,1.05,1.02,0.58])
        w.writerow(["burnes_c",0.31,0.07,0.24,0.08,98,5,0.1,1,5.5,1.02,1.01,0.60])
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=TODAY, help="YYYY-MM-DD")
    ap.add_argument("--odds_out", default=None, help="Output CSV path for DK odds")
    ap.add_argument("--features_out", default=None, help="Output CSV path for features")
    args = ap.parse_args()

    date = args.date
    odds_rows = map_to_rows(date, fetch_dk_odds(date))
    odds_out = args.odds_out or f"dk_markets_{date}.csv"
    write_csv(odds_out, odds_rows)
    print(f"Wrote {odds_out}")

    feat_out = args.features_out or build_pitcher_features_for_date(date)
    print(f"Seeded features at {feat_out} (edit values for the pitchers you’ll use).")

if __name__ == "__main__":
    main()
