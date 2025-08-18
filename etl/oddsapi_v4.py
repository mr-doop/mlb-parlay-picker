# etl/oddsapi_v4.py
from __future__ import annotations
import time
from typing import Iterable, List, Dict, Any, Optional
import requests
import pandas as pd

BASE = "https://api.the-odds-api.com/v4"

# -------------------- helpers --------------------

def american_to_decimal(price: Optional[float]) -> Optional[float]:
    if price is None:
        return None
    p = float(price)
    return 1.0 + (100.0/abs(p) if p < 0 else p/100.0)

def _norm_side(description: str) -> str:
    d = (description or "").strip().lower()
    if d.startswith("over"):
        return "Over"
    if d.startswith("under"):
        return "Under"
    if d in ("yes", "no"):
        return d.title()
    # some books use just 'Over' / 'Under'
    if d == "over":  return "Over"
    if d == "under": return "Under"
    return ""

def _safe_get(d: Dict[str, Any], *path, default=None):
    cur = d
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

# -------------------- public API --------------------

def fetch_events(api_key: str) -> pd.DataFrame:
    """
    List today’s MLB events (used to drive event-scoped prop queries).
    """
    url = f"{BASE}/sports/baseball_mlb/events"
    r = requests.get(url, params={"apiKey": api_key})
    r.raise_for_status()
    js = r.json() or []
    if not js:
        return pd.DataFrame()
    df = pd.json_normalize(js)
    # standardize columns we’ll use downstream
    df = df.rename(columns={
        "id": "event_id",
        "commence_time": "commence_time_iso",
        "home_team": "home_team",
        "away_team": "away_team",
    })
    return df[["event_id", "commence_time_iso", "home_team", "away_team"]]

def fetch_props_for_events(
    api_key: str,
    event_ids: Iterable[str],
    markets: List[str],
    bookmaker: str = "draftkings",
    region: str = "us",
    sleep_between: float = 0.15,
    chunk_size: int = 3,
) -> pd.DataFrame:
    """
    Fetch player props from the *event* endpoint.
    - Calls /events/{id}/odds for each event.
    - Queries markets in small chunks to avoid long URLs / edge cases.
    - Returns a tidy DataFrame with one row per outcome (prop leg).
    """
    out_rows: List[Dict[str, Any]] = []
    ev_ids = list(event_ids)

    # break markets into chunks (shorter query strings & less 422 risk)
    def _chunks(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i+n]

    for ev in ev_ids:
        for mchunk in _chunks(markets, chunk_size):
            params = {
                "apiKey": api_key,
                "regions": region,
                "oddsFormat": "american",
                "bookmakers": bookmaker,
                "markets": ",".join(mchunk),
            }
            url = f"{BASE}/sports/baseball_mlb/events/{ev}/odds"
            r = requests.get(url, params=params)
            if r.status_code == 404:
                # event not found yet at book
                time.sleep(sleep_between)
                continue
            try:
                r.raise_for_status()
            except requests.HTTPError as e:
                # bubble up useful message but keep going
                try:
                    msg = r.json()
                except Exception:
                    msg = r.text
                print(f"Props fetch failed for event {ev}, markets {mchunk}: {r.status_code} {msg}")
                time.sleep(sleep_between)
                continue

            js = r.json() or []
            if not js:
                time.sleep(sleep_between)
                continue

            # Response is a list with a single event element
            ev_obj = js[0] if isinstance(js, list) else js
            event_id = _safe_get(ev_obj, "id", default=ev)
            commence_iso = _safe_get(ev_obj, "commence_time")
            home_team = _safe_get(ev_obj, "home_team")
            away_team = _safe_get(ev_obj, "away_team")

            for bk in ev_obj.get("bookmakers", []):
                if bk.get("key") != bookmaker:
                    continue
                last_update = bk.get("last_update")
                for mk in bk.get("markets", []):
                    mkey = mk.get("key")
                    for oc in mk.get("outcomes", []):
                        # For player props: outcomes carry 'name' (player) and
                        # 'description' ("Over 6.5", "Yes", etc.), 'point' (line), 'price' (american)
                        player_name = oc.get("name")
                        side = _norm_side(oc.get("description", ""))
                        line = oc.get("point")
                        american = oc.get("price")
                        decimal = american_to_decimal(american)

                        out_rows.append({
                            "event_id": event_id,
                            "commence_time_iso": commence_iso,
                            "home_team": home_team,
                            "away_team": away_team,
                            "bookmaker": bookmaker,
                            "market_key": mkey,
                            "player_name": player_name,
                            "description_raw": oc.get("description"),
                            "side": side,
                            "line": line,
                            "american_odds": american,
                            "decimal_odds": decimal,
                            "last_update": last_update,
                        })
            time.sleep(sleep_between)

    if not out_rows:
        return pd.DataFrame()

    df = pd.DataFrame(out_rows)
    # Standardize some types
    for col in ["line", "american_odds", "decimal_odds"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Create a draft game_id from teams if you use one elsewhere (home@away)
    df["game_id"] = (
        df["away_team"].fillna("").str.replace(" ", "", regex=False) + "@" +
        df["home_team"].fillna("").str.replace(" ", "", regex=False)
    )
    return df