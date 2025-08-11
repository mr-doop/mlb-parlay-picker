import os, csv, datetime, time, requests, streamlit as st
from dataclasses import dataclass
from typing import Any, Dict, List

DATE_FMT = "%Y-%m-%d"
SPORT = "baseball_mlb"
ROOT = "https://api.the-odds-api.com/v4"

# We pull game odds first (h2h, spreads), then per-event props (Ks/Outs/Walks/Win)
GAME_MARKETS = "h2h,spreads"
PROP_MARKETS = ",".join([
    "pitcher_strikeouts",
    "pitcher_strikeouts_alternate",  # alt lines (X+)
    "pitcher_outs",
    "pitcher_walks",
    "pitcher_walks_alternate",       # alt lines (X+)
    "pitcher_record_a_win"
])

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

def _key() -> str:
    # Streamlit Cloud exposes secrets like a dict; no `.get`
    try:
        return str(st.secrets["ODDS_API_KEY"])
    except Exception:
        k = os.getenv("ODDS_API_KEY")
        if not k:
            raise RuntimeError("ODDS_API_KEY missing (add it in Streamlit → Manage app → Settings → Secrets)")
        return str(k)

def _get(url: str, params: Dict[str, Any]) -> Any:
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def _norm_game_id(date: str, home: str, away: str) -> str:
    return f"{date}-{home.replace(' ','')}-{away.replace(' ','')}"

def _pid_from_name(name: str) -> str:
    # crude ID: lastName_firstInitial (cole_g)
    parts = [p for p in name.replace(".", "").split(" ") if p]
    if not parts:
        return name.lower()
    first = parts[0]
    last = parts[-1]
    return f"{last.lower()}_{first[0].lower()}"

def fetch_games(date: str):
    """Get games (includes DraftKings h2h & spreads) and event IDs."""
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
    """Get DraftKings props for a single event via the event-level ODDS endpoint."""
    url = f"{ROOT}/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": _key(),
        "regions": "us",
        "markets": PROP_MARKETS,
        "oddsFormat": "american",
        "bookmakers": "draftkings"
    }
    return _get(url, params)

def _iter_dk_markets_from_event_response(event_odds: Any):
    """
    The Odds API can nest things two ways at event-level. Normalize to
    (market_key, outcomes_list) tuples for DraftKings only.
    """
    # Shape A: list[ { key, bookmakers: [ { key, markets: [ { key, outcomes: [...] } ] } ] } ]
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

    # Shape B: dict with 'bookmakers' directly
    if isinstance(event_odds, dict):
        for bm in event_odds.get("bookmakers", []):
            if bm.get("key") != "draftkings":
                continue
            for mk in bm.get("markets", []):
                mkey = str(mk.get("key", "")).lower()
                outs = mk.get("outcomes", [])
                if outs:
                    yield mkey, outs

def build_rows(date: str, games: List[Dict[str, Any]]) -> List[DKRow]:
    rows: List[DKRow] = []
    for g in games:
        event_id = g.get("id") or g.get("event_id")
        home = g.get("home_team", "HOME")
        away = g.get("away_team", "AWAY")
        gid = _norm_game_id(date, home, away)

        # 1) Game markets (from the /odds call above)
        for bm in g.get("bookmakers", []):
            if bm.get("key") != "draftkings":
                continue
            for mk in bm.get("markets", []):
                mkey = str(mk.get("key", "")).lower()
                if mkey == "h2h":
                    for out in mk.get("outcomes", []):
                        side = "HOME" if out.get("name") == home else "AWAY"
                        rows.append(DKRow(
                            date, gid, "MONEYLINE", side, out.get("name", ""),
                            "", "", "", int(out.get("price", 0))
                        ))
                elif mkey == "spreads":
                    for out in mk.get("outcomes", []):
                        spread = out.get("point")
                        team = out.get("name", "")
                        mt = "RUN_LINE" if (spread is not None and abs(float(spread)) == 1.5) else "ALT_RUN_LINE"
                        rows.append(DKRow(
                            date, gid, mt, team, team, "", "", 
                            str(spread) if spread is not None else "", int(out.get("price", 0))
                        ))

        # 2) Props per event (Ks/Outs/Walks/Win) via /events/{id}/odds
        if not event_id:
            continue
        try:
            props = fetch_props_for_event(event_id)
        except requests.HTTPError as e:
            st.warning(f"Props fetch failed for {gid}: {e}")
            time.sleep(0.15)
            continue

        for mkey, outcomes in _iter_dk_markets_from_event_response(props):
            lk = mkey.lower()
            for out in outcomes:
                name = str(out.get("name", ""))      # usually 'Over'/'Under'/'Yes'/'No'
                desc = str(out.get("description", ""))  # usually player name
                label = desc or name
                price = int(out.get("price", 0))
                point = out.get("point", None)

                if "strikeout" in lk:
                    side = "OVER" if "over" in name.lower() else ("UNDER" if "under" in name.lower() else "")
                    rows.append(DKRow(date, gid, "PITCHER_KS", side, "", _pid_from_name(label), label,
                                      str(point) if point is not None else "", price))
                elif "outs" in lk:
                    side = "OVER" if "over" in name.lower() else ("UNDER" if "under" in name.lower() else "")
                    rows.append(DKRow(date, gid, "PITCHER_OUTS", side, "", _pid_from_name(label), label,
                                      str(point) if point is not None else "", price))
                elif "walks" in lk:
                    side = "OVER" if "over" in name.lower() else ("UNDER" if "under" in name.lower() else "")
                    rows.append(DKRow(date, gid, "PITCHER_WALKS", side, "", _pid_from_name(label), label,
                                      str(point) if point is not None else "", price))
                elif "record_a_win" in lk or "to_record_a_win" in lk or "pitcher_to_record_a_win" in lk:
                    side = "YES" if "yes" in name.lower() else ("NO" if "no" in name.lower() else "")
                    rows.append(DKRow(date, gid, "PITCHER_WIN", side, "", _pid_from_name(label), label, "", price))

        time.sleep(0.10)  # be gentle on rate limits
    return rows

def write_csv(path: str, rows: List[DKRow]):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date","game_id","market_type","side","team","player_id","player_name","alt_line","american_odds"])
        for r in rows:
            w.writerow([r.date, r.game_id, r.market_type, r.side, r.team, r.player_id, r.player_name, r.alt_line, r.american_odds])

def run(date: str):
    games = fetch_games(date)
    rows = build_rows(date, games)
    out_odds = f"dk_markets_{date}.csv"
    write_csv(out_odds, rows)
    # Seed empty features file so you can download/edit quickly
    feat = f"features_{date}.csv"
    with open(feat, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["player_id","pitcher_k_rate","pitcher_bb_rate","opp_k_rate","opp_bb_rate","last5_pitch_ct_mean","days_rest","leash_bias","favorite_flag","bullpen_freshness","park_k_factor","ump_k_bias","team_ml_vigfree"])
    return out_odds, feat

if __name__ == "__main__":
    d = datetime.date.today().strftime(DATE_FMT)
    odds, feat = run(d)
    print("Wrote:", odds, feat)