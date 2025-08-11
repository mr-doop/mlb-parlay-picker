
import os, csv, argparse, datetime, time, requests, streamlit as st
from dataclasses import dataclass
from typing import Any, Dict, List

DATE_FMT = "%Y-%m-%d"
SPORT = "baseball_mlb"
ROOT = "https://api.the-odds-api.com/v4"

GAME_MARKETS = "h2h,spreads"
PROP_MARKETS = ",".join([
    "player_pitcher_strikeouts",
    "player_pitcher_outs",
    "player_pitcher_walks",
    "player_pitcher_to_record_a_win"
])

@dataclass
class DKRow:
    date: str; game_id: str; market_type: str; side: str; team: str
    player_id: str; player_name: str; alt_line: str; american_odds: int

def _key() -> str:
    try:
        # If st.secrets is a dict-like object
        if hasattr(st.secrets, "__getitem__"):
            return st.secrets["ODDS_API_KEY"]
    except Exception:
        pass
    # Fallback to environment variable
    k = os.getenv("ODDS_API_KEY")
    if not k:
        raise RuntimeError("ODDS_API_KEY missing (add it in Streamlit Secrets or environment).")
    return k

def _get(url: str, params: Dict[str, Any]) -> Any:
    r = requests.get(url, params=params, timeout=60)
    if r.status_code >= 400:
        raise requests.HTTPError(f"{r.status_code} {r.reason}: {r.text}")
    return r.json()

def _norm_game_id(date: str, home: str, away: str) -> str:
    return f"{date}-{home.replace(' ','')}-{away.replace(' ','')}"

def _pid_from_name(name: str) -> str:
    # crude: last name + first initial
    parts = [p for p in name.replace('.','').split(' ') if p]
    if not parts: return name.lower()
    first = parts[0]; last = parts[-1]
    return f"{last.lower()}_{first[0].lower()}"

def fetch_games(date: str):
    url = f"{ROOT}/sports/{SPORT}/odds"
    params = {
        "apiKey": _key(),
        "regions": "us",
        "markets": GAME_MARKETS,
        "oddsFormat": "american",
        "bookmakers": "draftkings"
    }
    return _get(url, params)

def fetch_props_for_event(event_id: str):
    url = f"{ROOT}/sports/{SPORT}/events/{event_id}/markets"
    params = {
        "apiKey": _key(),
        "regions": "us",
        "markets": PROP_MARKETS,
        "oddsFormat": "american",
        "bookmakers": "draftkings"
    }
    return _get(url, params)

def build_rows(date: str, games: List[Dict[str,Any]]) -> List[DKRow]:
    rows: List[DKRow] = []
    for g in games:
        event_id = g.get("id") or g.get("event_id")
        home = g.get("home_team","HOME"); away = g.get("away_team","AWAY")
        gid = _norm_game_id(date, home, away)
        # Game markets from the parent call
        for bm in g.get("bookmakers", []):
            if bm.get("key") != "draftkings": continue
            for m in bm.get("markets", []):
                key = str(m.get("key","")).lower()
                if key == "h2h":
                    for out in m.get("outcomes", []):
                        side = "HOME" if out.get("name")==home else "AWAY"
                        rows.append(DKRow(date,gid,"MONEYLINE",side,out.get("name",""),"","", "", int(out.get("price",0))))
                elif key == "spreads":
                    for out in m.get("outcomes", []):
                        spread = out.get("point")
                        team = out.get("name","")
                        mt = "RUN_LINE" if spread and abs(spread)==1.5 else "ALT_RUN_LINE"
                        rows.append(DKRow(date,gid,mt,team,team,"","", str(spread) if spread is not None else "", int(out.get("price",0))))
        # Now props via /events/{id}/markets
        if not event_id: 
            continue
        try:
            props = fetch_props_for_event(event_id)
            # props is a list of market dicts
            for m in props:
                key = str(m.get("key","")).lower()
                for bm in m.get("bookmakers", []):
                    if bm.get("key")!="draftkings": continue
                    for mk in bm.get("markets", []):
                        # The Odds API may nest market info differently inside event markets.
                        mkey = str(mk.get("key", key)).lower()
                        for out in mk.get("outcomes", []):
                            name = str(out.get("name",""))
                            desc = str(out.get("description",""))
                            price = int(out.get("price",0))
                            point = out.get("point", None)
                            # Map market type
                            if "strikeout" in mkey:
                                side = "OVER" if "over" in name.lower() else ("UNDER" if "under" in name.lower() else "")
                                player = desc or name
                                rows.append(DKRow(date,gid,"PITCHER_KS",side,"",_pid_from_name(player),player, str(point) if point is not None else "", price))
                            elif "outs" in mkey:
                                side = "OVER" if "over" in name.lower() else ("UNDER" if "under" in name.lower() else "")
                                player = desc or name
                                rows.append(DKRow(date,gid,"PITCHER_OUTS",side,"",_pid_from_name(player),player, str(point) if point is not None else "", price))
                            elif "walks" in mkey:
                                side = "OVER" if "over" in name.lower() else ("UNDER" if "under" in name.lower() else "")
                                player = desc or name
                                rows.append(DKRow(date,gid,"PITCHER_WALKS",side,"",_pid_from_name(player),player, str(point) if point is not None else "", price))
                            elif "record_a_win" in mkey or "to_record_a_win" in mkey or "pitcher_to_record_a_win" in mkey:
                                side = "YES" if "yes" in name.lower() else ("NO" if "no" in name.lower() else "")
                                player = desc or name.replace("Yes","").replace("No","").strip()
                                rows.append(DKRow(date,gid,"PITCHER_WIN",side,"",_pid_from_name(player),player,"", price))
        except requests.HTTPError as e:
            # Skip props for this event; still keep game markets
            st.warning(f"Props fetch failed for {gid}: {e}")
            time.sleep(0.2)
        time.sleep(0.15)  # be gentle on rate limits
    return rows

def write_csv(path: str, rows: List[DKRow]):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date","game_id","market_type","side","team","player_id","player_name","alt_line","american_odds"])
        for r in rows:
            w.writerow([r.date,r.game_id,r.market_type,r.side,r.team,r.player_id,r.player_name,r.alt_line,r.american_odds])

def run(date: str):
    games = fetch_games(date)
    rows = build_rows(date, games)
    out_odds = f"dk_markets_{date}.csv"
    write_csv(out_odds, rows)
    # seed a basic features file
    feat = f"features_{date}.csv"
    with open(feat, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["player_id","pitcher_k_rate","pitcher_bb_rate","opp_k_rate","opp_bb_rate","last5_pitch_ct_mean","days_rest","leash_bias","favorite_flag","bullpen_freshness","park_k_factor","ump_k_bias","team_ml_vigfree"])
    return out_odds, feat

if __name__ == "__main__":
    import sys
    d = datetime.date.today().strftime(DATE_FMT)
    odds, feat = run(d)
    print("Wrote:", odds, feat)
