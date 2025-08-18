# app.py -- MLB Parlay Picker (clean UI, fixed props, real projections)
# Fixes:
#  - Filters: de-duplicate games list by event_id
#  - Props parser: prefer `participant` for player_name (avoid name="Over")
#  - Team mapping: robust attach to probables (full key + last-name fallback)
#  - Team filter: don't drop rows with missing team when no team filter selected
#  - Projections: auto-apply toggle (default ON) so q_model != p_market

from __future__ import annotations
import os, math, html
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Dict, Any, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

APP_TITLE = "MLB Parlay Picker -- MVP"
BOOKMAKER = "draftkings"
TZ = "US/Eastern"
BASE = "https://api.the-odds-api.com/v4"

ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", os.getenv("ODDS_API_KEY", ""))

GAME_MARKETS = ["h2h", "spreads"]  # Moneyline + Run Line

# Only your pitcher markets (+ alts where available)
PROPS_MARKETS = [
    "pitcher_strikeouts",
    "pitcher_strikeouts_alternate",
    "pitcher_walks",
    "pitcher_walks_alternate",
    "pitcher_outs",
    "pitcher_record_a_win",
]

TEAM_ABBR = {
    "Arizona Diamondbacks":"ARI","Atlanta Braves":"ATL","Baltimore Orioles":"BAL","Boston Red Sox":"BOS",
    "Chicago Cubs":"CHC","Chicago White Sox":"CWS","Cincinnati Reds":"CIN","Cleveland Guardians":"CLE",
    "Colorado Rockies":"COL","Detroit Tigers":"DET","Houston Astros":"HOU","Kansas City Royals":"KC",
    "Los Angeles Angels":"LAA","Los Angeles Dodgers":"LAD","Miami Marlins":"MIA","Milwaukee Brewers":"MIL",
    "Minnesota Twins":"MIN","New York Mets":"NYM","New York Yankees":"NYY","Oakland Athletics":"OAK",
    "Philadelphia Phillies":"PHI","Pittsburgh Pirates":"PIT","San Diego Padres":"SDP","San Francisco Giants":"SFG",
    "Seattle Mariners":"SEA","St. Louis Cardinals":"STL","Tampa Bay Rays":"TBR","Texas Rangers":"TEX",
    "Toronto Blue Jays":"TOR","Washington Nationals":"WSH"
}

# -----------------------------
# Utilities
# -----------------------------

def get_session_df(key: str) -> pd.DataFrame:
    val = st.session_state.get(key, None)
    return val if isinstance(val, pd.DataFrame) else pd.DataFrame()

def american_to_decimal(a) -> float | None:
    try:
        a = float(a)
    except Exception:
        return None
    if a == 0:
        return None
    return 1 + (100/abs(a) if a < 0 else a/100)

def implied_prob_from_american(a) -> float | None:
    try:
        a = float(a)
    except Exception:
        return None
    if a == 0:
        return None
    return (abs(a)/(abs(a)+100)) if a < 0 else (100/(a+100))

def pct(x, nd=1):
    if x is None or pd.isna(x):
        return "--"
    return f"{100*float(x):.{nd}f}%"

def safe_str(x) -> str:
    try:
        return str(x)
    except Exception:
        return ""

def norm_key(name: str) -> str:
    return (name or "").lower().replace(".","").strip()

def last_name_key(name: str) -> str:
    n = norm_key(name).split()
    return n[-1] if n else ""

def first_initial_last(full_name: str) -> str:
    s = (full_name or "").replace(".", "").strip()
    if not s: return ""
    parts = s.split()
    if len(parts) == 1: return parts[0]
    return f"{parts[0][0]}. {parts[-1]}"

def fmt_team(abbr: str | None) -> str:
    return safe_str(abbr).upper()[:3] if abbr else ""

def short_market_label(market_key: str) -> str:
    M = {
        "h2h":"ML",
        "spreads":"Run Line",
        "pitcher_strikeouts":"Ks",
        "pitcher_strikeouts_alternate":"Ks",
        "pitcher_walks":"BB",
        "pitcher_walks_alternate":"BB",
        "pitcher_outs":"Outs",
        "pitcher_record_a_win":"Win",
    }
    return M.get(market_key, market_key)

def clean_side_text(text: str) -> str:
    s = (text or "").strip().lower()
    if s.startswith("over"):  return "Over"
    if s.startswith("under"): return "Under"
    if s == "yes":  return "Yes"
    if s == "no":   return "No"
    return text

def build_bet_title(r: pd.Series) -> str:
    market = r.get("market_key","")
    cat = r.get("category","")
    side = r.get("side","")
    team = fmt_team(r.get("team_abbr",""))
    opp = fmt_team(r.get("opp_abbr",""))
    is_home = r.get("is_home", None)
    matchup = f"{opp}@{team}" if is_home is True else (f"{team}@{opp}" if is_home is False else f"{opp}@{team}")
    label = short_market_label(market)
    line = r.get("line", None)
    player = r.get("player_name","")
    name = first_initial_last(player)

    if cat in ("Moneyline","Run Line"):
        if cat == "Run Line":
            rl = r.get("line_run", "")
            if rl is None or pd.isna(rl): return f"{team} {label} ({matchup})"
            sign = "-" if float(rl) < 0 else "+"
            return f"{team} {sign}{abs(float(rl)):.1f} ({matchup})"
        return f"{team} {label} ({matchup})"

    if side in ("Over","Under") and line is not None and not pd.isna(line):
        side_short = "O" if side == "Over" else "U"
        return f"{name} ({team}) {side_short}{float(line):g} {label}"
    if side in ("Yes","No"):
        return f"{name} ({team}) Win {'YES' if side=='Yes' else 'NO'}"
    return f"{name} ({team}) {label}"

def chips_row(r: pd.Series) -> str:
    odds = r.get("american_odds")
    q = r.get("q_model")
    p = r.get("p_market")
    edge = (q - p) if (pd.notna(q) and pd.notna(p)) else None
    pills = []
    pills.append(f"<span class='chip'>Odds {int(odds) if pd.notna(odds) else '--'}</span>")
    pills.append(f"<span class='chip'>q {pct(q)}</span>")
    pills.append(f"<span class='chip'>Market {pct(p)}</span>")
    if edge is not None:
        pills.append(f"<span class='chip'>Edge {edge*100:+.1f}%</span>")
    oppK = r.get("opp_k_rate"); oppBB = r.get("opp_bb_rate")
    if pd.notna(oppK): pills.append(f"<span class='chip'>OppK {oppK*100:.1f}%</span>")
    if pd.notna(oppBB): pills.append(f"<span class='chip'>OppBB {oppBB*100:.1f}%</span>")
    return " ".join(pills)

def render_card(r: pd.Series):
    title = build_bet_title(r)
    chips = chips_row(r)
    st.markdown(
        f"""
        <div class="card">
          <div class="card-title">{html.escape(title)}</div>
          <div class="card-sub">{chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Fetchers (OddsAPI v4)
# -----------------------------

def oddsapi_events(api_key: str) -> pd.DataFrame:
    url = f"{BASE}/sports/baseball_mlb/events"
    r = requests.get(url, params={"apiKey": api_key}, timeout=30)
    r.raise_for_status()
    js = r.json() or []
    rows=[]
    for e in js:
        rows.append({
            "event_id": e.get("id"),
            "commence_time": e.get("commence_time"),
            "home_team": e.get("home_team"),
            "away_team": e.get("away_team"),
            "home_abbr": TEAM_ABBR.get(e.get("home_team",""), None),
            "away_abbr": TEAM_ABBR.get(e.get("away_team",""), None),
        })
    return pd.DataFrame(rows)

def oddsapi_odds_board(api_key: str, markets: list[str]) -> list[dict]:
    url = f"{BASE}/sports/baseball_mlb/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "oddsFormat": "american",
        "markets": ",".join(markets),
        "bookmakers": BOOKMAKER,
    }
    r = requests.get(url, params=params, timeout=45)
    r.raise_for_status()
    return r.json()

def oddsapi_props_for_events(
    api_key: str,
    event_ids: Iterable[str],
    markets: List[str],
    bookmaker: str = "draftkings",
    region: str = "us",
    sleep_between: float = 0.12,
    chunk_size: int = 3,
) -> pd.DataFrame:
    out_rows: List[Dict[str, Any]] = []
    ev_ids = [e for e in event_ids if e]

    def _chunks(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i+n]

    for ev in ev_ids:
        for mchunk in _chunks(markets, chunk_size):
            params = {"apiKey": api_key, "regions": region, "oddsFormat": "american",
                      "bookmakers": bookmaker, "markets": ",".join(mchunk)}
            url = f"{BASE}/sports/baseball_mlb/events/{ev}/odds"
            r = requests.get(url, params=params, timeout=45)
            if r.status_code == 404:
                continue
            try:
                r.raise_for_status()
            except requests.HTTPError:
                # log but continue
                continue

            js = r.json() or []
            if not js: continue
            ev_obj = js[0] if isinstance(js, list) else js
            event_id = ev_obj.get("id", ev)
            commence_iso = ev_obj.get("commence_time")
            home_team = ev_obj.get("home_team")
            away_team = ev_obj.get("away_team")

            for bk in ev_obj.get("bookmakers", []):
                if bk.get("key") != bookmaker:
                    continue
                last_update = bk.get("last_update")
                for mk in bk.get("markets", []):
                    mkey = mk.get("key")
                    for oc in mk.get("outcomes", []):
                        # IMPORTANT: use participant first; name is often "Over"/"Under"
                        player_name = oc.get("participant") or oc.get("name") or ""
                        desc = oc.get("description", "")
                        side = clean_side_text(desc)
                        line = oc.get("point")
                        american = oc.get("price")
                        dec = american_to_decimal(american)
                        out_rows.append({
                            "event_id": event_id,
                            "commence_time": commence_iso,
                            "home_team": home_team,
                            "away_team": away_team,
                            "bookmaker": bookmaker,
                            "market_key": mkey,
                            "player_name": player_name,
                            "description_raw": desc,
                            "side": side,
                            "line": line,
                            "american_odds": american,
                            "decimal_odds": dec,
                            "last_update": last_update,
                        })
    df = pd.DataFrame(out_rows)
    for col in ("line","american_odds","decimal_odds"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if not df.empty:
        df["game_id"] = (
            df["away_team"].fillna("").str.replace(" ","", regex=False) + "@" +
            df["home_team"].fillna("").str.replace(" ","", regex=False)
        )
    return df

def normalize_board(board_json: list[dict]) -> pd.DataFrame:
    rows=[]
    for ev in board_json:
        eid = ev.get("id")
        home = ev.get("home_team"); away = ev.get("away_team")
        home_abbr = TEAM_ABBR.get(home,""); away_abbr = TEAM_ABBR.get(away,"")
        for bm in ev.get("bookmakers", []):
            if bm.get("key") != BOOKMAKER: 
                continue
            for mk in bm.get("markets", []):
                key = mk.get("key","")
                if key == "h2h":
                    for oc in mk.get("outcomes", []):
                        name = oc.get("name","")
                        price = oc.get("price")
                        is_home = (name == home)
                        team_abbr = TEAM_ABBR.get(name, home_abbr if is_home else away_abbr)
                        opp_abbr = away_abbr if is_home else home_abbr
                        rows.append({
                            "event_id": eid,"category":"Moneyline","market_key":"h2h","side":"",
                            "team_abbr":team_abbr,"opp_abbr":opp_abbr,"is_home":is_home,
                            "player_name":"","line":None,"line_run":None,"american_odds":price
                        })
                elif key == "spreads":
                    for oc in mk.get("outcomes", []):
                        name = oc.get("name",""); price = oc.get("price"); point = oc.get("point")
                        is_home = (name == home)
                        team_abbr = TEAM_ABBR.get(name, home_abbr if is_home else away_abbr)
                        opp_abbr = away_abbr if is_home else home_abbr
                        rows.append({
                            "event_id": eid,"category":"Run Line","market_key":"spreads","side":"",
                            "team_abbr":team_abbr,"opp_abbr":opp_abbr,"is_home":is_home,
                            "player_name":"","line":None,"line_run":point,"american_odds":price
                        })
    return pd.DataFrame(rows)

# -----------------------------
# Probables, Opponent rates, Projections
# -----------------------------

def mlb_probables(date: datetime) -> pd.DataFrame:
    d = date.strftime("%Y-%m-%d")
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={d}&hydrate=probablePitcher,team"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    js = r.json()
    rows=[]
    for blk in js.get("dates", []):
        for g in blk.get("games", []):
            hm = g.get("teams",{}).get("home",{})
            aw = g.get("teams",{}).get("away",{})
            hp = (hm.get("probablePitcher") or {}).get("fullName")
            ap = (aw.get("probablePitcher") or {}).get("fullName")
            home_name = hm.get("team",{}).get("name","")
            away_name = aw.get("team",{}).get("name","")
            rows.append({
                "home_abbr": TEAM_ABBR.get(home_name,""),
                "away_abbr": TEAM_ABBR.get(away_name,""),
                "home_pitcher": hp, "away_pitcher": ap
            })
    # Build normalized keys
    df = pd.DataFrame(rows)
    if not df.empty:
        df["home_pitch_key"] = df["home_pitcher"].apply(norm_key)
        df["away_pitch_key"] = df["away_pitcher"].apply(norm_key)
        df["home_last"] = df["home_pitcher"].apply(last_name_key)
        df["away_last"] = df["away_pitcher"].apply(last_name_key)
    return df

def attach_probables_to_props(props_df: pd.DataFrame, prob_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    if props_df.empty or prob_df.empty or events_df.empty:
        return props_df
    props_df = props_df.copy()
    # Map event_id → home/away abbr (from events_df)
    pair_map = events_df.drop_duplicates("event_id")[["event_id","home_abbr","away_abbr"]].set_index("event_id").to_dict("index")

    # Build a lookup per (home_abbr,away_abbr) to probable keys
    prob_df = prob_df.copy()
    prob_df["key"] = prob_df["home_abbr"].fillna("") + "@" + prob_df["away_abbr"].fillna("")
    prob_map = prob_df.set_index("key").to_dict("index")

    for i, r in props_df.iterrows():
        eid = r.get("event_id"); p = norm_key(r.get("player_name",""))
        if not eid or not p: 
            continue
        pair = pair_map.get(eid, None)
        if not pair: 
            continue
        home_abbr = pair.get("home_abbr"); away_abbr = pair.get("away_abbr")
        k = f"{home_abbr}@{away_abbr}"
        pr = prob_map.get(k, None)
        if not pr:
            # try reverse
            k2 = f"{away_abbr}@{home_abbr}"
            pr = prob_map.get(k2, None)
            if pr:
                home_abbr, away_abbr = away_abbr, home_abbr
        if not pr:
            continue

        hp = pr.get("home_pitch_key"); ap = pr.get("away_pitch_key")
        hl = pr.get("home_last");     al = pr.get("away_last")
        pl = last_name_key(p)

        match_home = (p == hp) or (pl and pl == hl)
        match_away = (p == ap) or (pl and pl == al)

        if match_home and not match_away:
            props_df.at[i,"team_abbr"] = home_abbr; props_df.at[i,"opp_abbr"] = away_abbr; props_df.at[i,"is_home"]=True
        elif match_away and not match_home:
            props_df.at[i,"team_abbr"] = away_abbr; props_df.at[i,"opp_abbr"] = home_abbr; props_df.at[i,"is_home"]=False
        # if ambiguous (both or neither), leave as-is; filter will handle if user selects specific teams
    return props_df

def mlb_team_rates_last_days(days=21) -> pd.DataFrame:
    end = datetime.now(timezone.utc); start = end - timedelta(days=days)
    start_str, end_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    teams_url = "https://statsapi.mlb.com/api/v1/teams?sportId=1&activeStatus=Y"
    tr = requests.get(teams_url, timeout=30); tr.raise_for_status()
    teams = tr.json().get("teams", [])
    rows=[]
    for t in teams:
        name = t.get("name",""); abbr = TEAM_ABBR.get(name)
        if not abbr: continue
        tid = t.get("id")
        url = f"https://statsapi.mlb.com/api/v1/teams/{tid}/stats?stats=byDateRange&group=hitting&startDate={start_str}&endDate={end_str}"
        r = requests.get(url, timeout=30)
        if r.status_code != 200: continue
        stats = r.json().get("stats", [])
        stat = (stats[0].get("splits",[]) or [{}])[0].get("stat",{})
        so = float(stat.get("strikeOuts",0)); bb = float(stat.get("baseOnBalls",0))
        pa = float(stat.get("plateAppearances",0))
        if pa == 0:
            ab = float(stat.get("atBats",0)); hbp = float(stat.get("hitByPitch",0))
            sb = float(stat.get("sacBunts",0)); sf = float(stat.get("sacFlies",0))
            pa = ab + bb + hbp + sb + sf
        k_rate = so/pa if pa else np.nan; bb_rate = bb/pa if pa else np.nan
        rows.append({"team_abbr": abbr, "opp_k_rate": k_rate, "opp_bb_rate": bb_rate})
    return pd.DataFrame(rows)

def vig_free_pair(a1, a2):
    p1 = implied_prob_from_american(a1); p2 = implied_prob_from_american(a2)
    if p1 is None or p2 is None: return (p1, p2)
    s = p1 + p2
    if s == 0: return (p1, p2)
    return (p1/s, p2/s)

def compute_probs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["p_market"] = np.nan
    # Moneyline vig removal
    ml = df[df["category"]=="Moneyline"]
    for eid, g in ml.groupby("event_id"):
        idx = list(g.index)
        if len(idx) >= 2:
            p1 = df.at[idx[0],"american_odds"]; p2 = df.at[idx[1],"american_odds"]
            vf1, vf2 = vig_free_pair(p1, p2)
            df.at[idx[0], "p_market"] = vf1; df.at[idx[1], "p_market"] = vf2
    # Others: implied directly
    mask = df["p_market"].isna()
    df.loc[mask, "p_market"] = df.loc[mask, "american_odds"].apply(implied_prob_from_american)
    return df

# ---- Free projections (public data) ----
try:
    from pybaseball import pitching_stats as _fg_pitching_stats
except Exception:
    _fg_pitching_stats = None

@st.cache_data(ttl=60*60, show_spinner=False)
def _load_pitcher_table(season: int) -> pd.DataFrame:
    if _fg_pitching_stats is None:
        return pd.DataFrame()
    try:
        df = _fg_pitching_stats(season=season, qual=0)
    except Exception:
        return pd.DataFrame()
    df = df.rename(columns={"Name":"player_name","K%":"k_pct","BB%":"bb_pct","IP":"ip","GS":"gs"})
    for c in ["k_pct","bb_pct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")/100.0
    for c in ["ip","gs"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["player_key"] = df["player_name"].apply(norm_key)
    df["last_key"] = df["player_name"].apply(last_name_key)
    return df[["player_name","player_key","last_key","k_pct","bb_pct","ip","gs"]]

def _bf_from_ip(ip: float) -> float:
    return float(ip) * 4.3 if pd.notna(ip) else np.nan

def _ip_per_start(ip: float, gs: float) -> float:
    if pd.isna(ip) or pd.isna(gs) or gs <= 0: return np.nan
    return float(ip)/float(gs)

def _adj_rate(p_rate: float, opp_rate: float, weight=0.45) -> float:
    if pd.isna(p_rate) and pd.isna(opp_rate): return np.nan
    if pd.isna(p_rate): return float(opp_rate)
    if pd.isna(opp_rate): return float(p_rate)
    return (1.0 - weight)*float(p_rate) + weight*float(opp_rate)

def _poisson_over(mu: float, line: float) -> float:
    if mu is None or not np.isfinite(mu) or mu <= 0: return np.nan
    k = int(math.floor(line))
    cdf = 0.0
    term = math.exp(-mu)
    cdf += term
    for i in range(1, k+1):
        term *= mu / i
        cdf += term
    return float(max(0.0, min(1.0, 1.0 - cdf)))

def build_free_q(pool: pd.DataFrame, slate_date: datetime, opp_rates: pd.DataFrame) -> pd.DataFrame:
    if pool is None or pool.empty: return pool
    season = slate_date.year
    fg = _load_pitcher_table(season)
    if fg.empty: return pool
    df = pool.copy()
    # match using full key and last name
    df["player_key"] = df["player_name"].apply(norm_key)
    df["last_key"] = df["player_name"].apply(last_name_key)
    fg = fg.drop_duplicates(subset=["player_key"], keep="first")
    df = df.merge(fg, on="player_key", how="left", suffixes=("","_fg"))
    # if no full key match, try last-name fallback
    mask_missing = df["k_pct"].isna() & df["last_key"].notna()
    if mask_missing.any():
        fallback = fg.drop_duplicates(subset=["last_key"])[["last_key","k_pct","bb_pct","ip","gs"]]
        df = df.merge(fallback, on="last_key", how="left", suffixes=("","_ln"))
        for col in ["k_pct","bb_pct","ip","gs"]:
            df[col] = df[col].fillna(df[f"{col}_ln"])
        df.drop(columns=[c for c in df.columns if c.endswith("_ln")], inplace=True, errors="ignore")

    rates = (opp_rates or pd.DataFrame()).copy().rename(columns={"team_abbr":"opp_abbr_join"})
    df = df.merge(rates, left_on="opp_abbr", right_on="opp_abbr_join", how="left")
    df.drop(columns=["opp_abbr_join"], inplace=True, errors="ignore")

    df["ip_ps"]  = _ip_per_start(df["ip"], df["gs"])
    df["bf_mu"]  = _bf_from_ip(df["ip_ps"].fillna(5.5))
    df["k_rate"] = _adj_rate(df["k_pct"], df["opp_k_rate"], weight=0.45)
    df["bb_rate"]= _adj_rate(df["bb_pct"], df["opp_bb_rate"], weight=0.45)

    df["mu_k"]   = df["bf_mu"] * df["k_rate"]
    df["mu_bb"]  = df["bf_mu"] * df["bb_rate"]
    df["mu_out"] = (df["ip_ps"].fillna(5.5)) * 3.0

    # map ML probability for Win calc
    ml_map = {}
    ml_rows = df[df["category"]=="Moneyline"]
    for _, r in ml_rows.iterrows():
        ml_map[(r.get("event_id"), r.get("team_abbr"))] = r.get("p_market")

    if "q_model" not in df.columns:
        df["q_model"] = df.get("p_market")

    def _q_for_row(row) -> float | np.nan:
        mk = row.get("market_key",""); side = (row.get("side") or "").title(); line = row.get("line")
        if pd.isna(line): return np.nan
        if "strikeouts" in mk: mu = row.get("mu_k")
        elif "walks" in mk:   mu = row.get("mu_bb")
        elif "outs" in mk:    mu = row.get("mu_out")
        else: return np.nan
        if pd.isna(mu): return np.nan
        if side == "Over":  return _poisson_over(mu, float(line))
        if side == "Under": return 1.0 - _poisson_over(mu, float(line))
        return np.nan

    df["q_free"] = df.apply(_q_for_row, axis=1)

    def _q_win(row) -> float | np.nan:
        mk = row.get("market_key",""); side = (row.get("side") or "").title()
        if "record_a_win" not in mk or side not in ("Yes","No"): return np.nan
        ml_prob = ml_map.get((row.get("event_id"), row.get("team_abbr")))
        if pd.isna(ml_prob): ml_prob = row.get("p_market")
        p_outs15 = 1.0 - _poisson_over(row.get("mu_out", np.nan), 14.5) if pd.notna(row.get("mu_out")) else np.nan
        if pd.isna(ml_prob) or pd.isna(p_outs15): return np.nan
        yes = float(ml_prob) * float(p_outs15)
        return yes if side == "Yes" else (1.0 - yes)

    df["q_win"] = df.apply(_q_win, axis=1)
    df["q_model"] = np.where(df["q_free"].notna(), df["q_free"], df["q_model"])
    df["q_model"] = np.where(df["q_win"].notna(),  df["q_win"],  df["q_model"])

    df.drop(columns=[c for c in ["player_key","last_key","k_pct","bb_pct","ip","gs","ip_ps","bf_mu","k_rate","bb_rate","mu_k","mu_bb","mu_out","q_free","q_win"] if c in df.columns], inplace=True, errors="ignore")
    return df

def join_opponent_rates(df: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    if rates is None or rates.empty: return df
    return df.merge(rates, left_on="opp_abbr", right_on="team_abbr", how="left") \
             .drop(columns=["team_abbr_y"], errors="ignore") \
             .rename(columns={"team_abbr_x":"team_abbr"})

def attach_features(df: pd.DataFrame, feat: pd.DataFrame | None) -> pd.DataFrame:
    df = df.copy()
    df["q_model"] = df["p_market"]  # fallback
    if feat is None or feat.empty:
        return df
    f = feat.copy(); f.columns = [c.lower() for c in f.columns]
    if "q_proj" not in f.columns: return df
    if "player_name" in f.columns:
        f["player_key"] = f["player_name"].apply(norm_key)
        df["player_key"] = df["player_name"].apply(norm_key)
        df = df.merge(f[["player_key","q_proj"]], on="player_key", how="left")
        df["q_model"] = df["q_proj"].where(df["q_proj"].notna(), df["q_model"])
        df.drop(columns=["player_key","q_proj"], inplace=True, errors="ignore")
    return df

def build_pool(board_df: pd.DataFrame, props_df: pd.DataFrame, events_df: pd.DataFrame,
               prob_df: pd.DataFrame, rates_df: pd.DataFrame, feat_df: pd.DataFrame | None) -> pd.DataFrame:
    pieces = []
    if isinstance(board_df, pd.DataFrame) and not board_df.empty:
        pieces.append(board_df)
    if isinstance(props_df, pd.DataFrame) and not props_df.empty:
        props_df = attach_probables_to_props(props_df, prob_df, events_df)
        pieces.append(props_df)
    if not pieces:
        return pd.DataFrame()
    df = pd.concat(pieces, ignore_index=True, sort=False)

    df = compute_probs(df)
    df = join_opponent_rates(df, rates_df)
    df = attach_features(df, feat_df)

    for c in ["team_abbr","opp_abbr","player_name","market_key","category","side","line","line_run","american_odds","p_market","q_model","is_home","event_id"]:
        if c not in df.columns: df[c] = np.nan

    df["category"] = np.where(df["market_key"].isin(["h2h","spreads"]), df["category"],
                       np.where(df["market_key"].str.contains("strikeouts"), "Pitcher Ks",
                       np.where(df["market_key"].str.contains("walks"), "Pitcher Walks",
                       np.where(df["market_key"].str.contains("outs"), "Pitcher Outs",
                       np.where(df["market_key"].str.contains("record_a_win"), "Win", df["category"])))))
    df["description"] = df.apply(build_bet_title, axis=1)
    df["edge"] = df["q_model"] - df["p_market"]
    df["q_pct"] = (df["q_model"]*100).round(1)
    df["p_pct"] = (df["p_market"]*100).round(1)

    keys = ["event_id","team_abbr","player_name","market_key","side","line","line_run","american_odds"]
    df = df.drop_duplicates(subset=keys).reset_index(drop=True)
    return df

# -----------------------------
# UI
# -----------------------------

APP_CSS = """
<style>
  :root { --ink:#111827; --muted:#6b7280; --chip:#f3f4f6; }
  .card{background:#fff;border:1px solid #eee;border-radius:14px;padding:12px 14px;margin:10px 0;box-shadow:0 1px 3px rgba(0,0,0,.05);}
  .card-title{font-weight:600;color:var(--ink);margin-bottom:6px;}
  .card-sub{color:var(--muted);font-size:12px;}
  .chip{display:inline-block;background:var(--chip);padding:3px 8px;border-radius:999px;margin-right:6px;border:1px solid #e5e7eb}
  .pill{display:inline-block;padding:3px 8px;border-radius:999px;border:1px solid #e5e7eb;background:#fff;font-size:12px;margin-right:6px;}
</style>
"""

st.set_page_config(page_title=APP_TITLE, page_icon="⚾", layout="centered")
st.markdown(APP_CSS, unsafe_allow_html=True)
st.title(APP_TITLE)
st.caption("Manual fetch • DraftKings only • Apple‑like cards • Market % + free projections (q)")

# Init state
for k in ["events_df","board_df","props_df","prob_df","rates_df","features_df"]:
    st.session_state.setdefault(k, None)

with st.expander("How to use", expanded=False):
    st.write("1) **Fetch Events** and **Fetch Board**. 2) **Fetch Props** (credits). 3) *(Optional)* upload features with `player_name,q_proj`. 4) Ensure **Auto‑apply projections** is on. 5) Filter & use tabs. 6) Download CSVs.")

# Controls
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    slate_date = st.date_input("Slate date", value=datetime.now().date())

with c2:
    if st.button("Fetch Events"):
        if not ODDS_API_KEY:
            st.error("Set ODDS_API_KEY in Secrets.")
        else:
            try:
                events = oddsapi_events(ODDS_API_KEY)
                # De-duplicate by event_id to avoid duplicate games in filters
                events = events.drop_duplicates("event_id").reset_index(drop=True)
                st.session_state["events_df"] = events
                st.session_state["prob_df"] = mlb_probables(datetime.combine(slate_date, datetime.min.time()))
                st.success(f"Fetched {len(events)} events.")
            except Exception as e:
                st.error(f"Events fetch failed: {e}")

with c3:
    if st.button("Fetch Board (ML/RL)"):
        if not ODDS_API_KEY:
            st.error("Set ODDS_API_KEY in Secrets.")
        else:
            try:
                board_json = oddsapi_odds_board(ODDS_API_KEY, GAME_MARKETS)
                board_df = normalize_board(board_json)
                st.session_state["board_df"] = board_df
                st.success(f"Fetched board: {len(board_df)} lines.")
            except Exception as e:
                st.error(f"Board fetch failed: {e}")

with c4:
    if st.button("Fetch Props (uses credits)"):
        if not ODDS_API_KEY:
            st.error("Set ODDS_API_KEY in Secrets.")
        else:
            try:
                evdf = get_session_df("events_df")
                if evdf.empty:
                    evdf = oddsapi_events(ODDS_API_KEY).drop_duplicates("event_id")
                    st.session_state["events_df"] = evdf
                ev_ids = list(evdf["event_id"].dropna().unique())
                props_df = oddsapi_props_for_events(
                    api_key=ODDS_API_KEY,
                    event_ids=ev_ids,
                    markets=PROPS_MARKETS,
                    bookmaker=BOOKMAKER,
                    region="us",
                    sleep_between=0.12,
                    chunk_size=3
                )
                st.session_state["props_df"] = props_df
                st.success(f"Fetched props: {len(props_df)} rows.")
            except Exception as e:
                st.error(f"Props fetch failed: {e}")

with c5:
    auto_proj = st.toggle("Auto‑apply projections", value=True)

# Uploads
u1, u2 = st.columns(2)
with u1:
    up_board = st.file_uploader("Upload Board CSV (optional)", type=["csv"])
    if up_board:
        st.session_state["board_df"] = pd.read_csv(up_board)
        st.success(f"Loaded board CSV: {len(st.session_state['board_df'])} rows.")
with u2:
    up_feat = st.file_uploader("Upload Features CSV (player_name,q_proj)", type=["csv"])
    if up_feat:
        st.session_state["features_df"] = pd.read_csv(up_feat)
        st.success(f"Loaded features CSV: {len(st.session_state['features_df'])} rows.")

# Downloads
dl1, dl2 = st.columns(2)
bd = get_session_df("board_df")
if not bd.empty:
    dl1.download_button("Download Board CSV", bd.to_csv(index=False).encode(), file_name=f"board_{slate_date}.csv", mime="text/csv")
pdn = get_session_df("props_df")
if not pdn.empty:
    dl2.download_button("Download Props CSV", pdn.to_csv(index=False).encode(), file_name=f"props_{slate_date}.csv", mime="text/csv")

# Opponent rates cache
if st.session_state.get("rates_df") is None:
    try:
        st.session_state["rates_df"] = mlb_team_rates_last_days(days=21)
    except Exception:
        st.session_state["rates_df"] = pd.DataFrame()
rates_df = get_session_df("rates_df")

# Build pool
events_df = get_session_df("events_df")
board_df  = get_session_df("board_df")
props_df  = get_session_df("props_df")
prob_df   = get_session_df("prob_df")
feat_df   = get_session_df("features_df")

pool_base = build_pool(board_df, props_df, events_df, prob_df, rates_df, feat_df)

# Optional free projections
if auto_proj and not pool_base.empty:
    try:
        pool_base = build_free_q(pool_base, datetime.combine(slate_date, datetime.min.time()), rates_df)
    except Exception as e:
        st.warning(f"Free projections skipped: {e}")

# -----------------------------
# Filters
# -----------------------------

st.markdown("### Filters")

# Dedup games list by event
games_det = []
if not events_df.empty:
    for _, r in events_df.drop_duplicates("event_id").iterrows():
        games_det.append(f"{fmt_team(r['away_abbr'])}@{fmt_team(r['home_abbr'])}")

select_all_games = st.checkbox("Select all games", value=True)
games_pick = st.multiselect("Games", sorted(set(games_det)), default=(sorted(set(games_det)) if select_all_games else []))

teams_all = sorted(set([fmt_team(x) for x in pool_base["team_abbr"].dropna().unique().tolist()]))

# We won't drop rows with missing team unless the user selects specific teams
select_all_teams = st.checkbox("Select all teams", value=True)
teams_pick = st.multiselect("Teams", teams_all, default=(teams_all if select_all_teams else []))

cat_options = ["Moneyline","Run Line","Pitcher Ks","Pitcher Walks","Pitcher Outs","Win"]
def cat_from_row(r):
    if r["category"] in ("Moneyline","Run Line"): return r["category"]
    mk = r["market_key"]
    if "strikeouts" in mk: return "Pitcher Ks"
    if "walks" in mk: return "Pitcher Walks"
    if "outs" in mk: return "Pitcher Outs"
    if "record_a_win" in mk: return "Win"
    return r["category"]

pool = pool_base.copy()
if not pool.empty:
    pool["ui_cat"] = pool.apply(cat_from_row, axis=1)

cat_pick = st.multiselect("Categories", cat_options, default=cat_options)
odds_min, odds_max = st.slider("American odds", -700, 700, (-700, 700), step=5)

def row_game(r):
    a = fmt_team(r.get("opp_abbr")); h = fmt_team(r.get("team_abbr"))
    if r.get("is_home") is True:  return f"{a}@{h}"
    if r.get("is_home") is False: return f"{h}@{a}"
    return f"{a}@{h}"

if not pool.empty:
    pool["game_key"] = pool.apply(row_game, axis=1)

    # Team filter logic: if user selected teams, require membership; else, allow all (including None)
    def team_pass(x):
        t = fmt_team(x)
        if teams_pick:  # user chose specific teams
            return t in teams_pick
        return True  # pass-through if no team filter selected

    pool = pool[
        pool["american_odds"].between(odds_min, odds_max, inclusive="both") &
        (pool["game_key"].isin(games_pick) if games_pick else True) &
        (pool["ui_cat"].isin(cat_pick) if cat_pick else True) &
        pool["team_abbr"].apply(team_pass)
    ].reset_index(drop=True)

st.markdown("#### Coverage (MLB schedule) -- games detected")
st.caption(", ".join(sorted(set(games_det))) if games_det else "No games found for this date.")

# -----------------------------
# Tabs
# -----------------------------

tabs = st.tabs(["Candidates", "Top 20", "Parlay Presets", "Alt Line Safety Board", "One‑Tap Ticket", "ML Winners & RL Locks", "Debug"])

with tabs[0]:
    if pool.empty:
        st.info("No legs yet. Fetch board/props or relax filters.")
    else:
        show = pool.rename(columns={"american_odds":"Odds","q_pct":"q %","p_pct":"Market %"})
        st.dataframe(
            show[["description","category","Odds","q %","Market %","edge"]].sort_values(["edge","q %"], ascending=[False,False]),
            use_container_width=True, height=480
        )

with tabs[1]:
    if pool.empty:
        st.info("No legs.")
    else:
        top = pool.sort_values(["edge","q_model","p_market"], ascending=[False,False,True]).head(20)
        for _, r in top.iterrows():
            render_card(r)

def preset_bucket(df: pd.DataFrame, legs: int, risk: str) -> pd.DataFrame:
    if df.empty: return df
    d = df.copy()
    if risk == "Low":
        d = d[(d["american_odds"] <= -150) & (d["q_model"] >= 0.60)]
    elif risk == "Medium":
        d = d[(d["american_odds"] > -150) & (d["american_odds"] <= +150) & (d["q_model"] >= 0.55)]
    else:
        d = d[(d["american_odds"] > +150) & (d["q_model"] >= 0.50)]
    d = d.sort_values(["edge","q_model"], ascending=[False,False])
    picks=[]; used=set()
    for _, r in d.iterrows():
        if len(picks) >= legs: break
        g = r.get("game_key")
        if g in used: continue
        picks.append(r); used.add(g)
    return pd.DataFrame(picks)

with tabs[2]:
    if pool.empty:
        st.info("No legs.")
    else:
        st.markdown("**Parlay Presets (4 · 5 · 6 · 8 legs)**")
        for L in [4,5,6,8]:
            st.subheader(f"{L}-Leg")
            for risk in ["Low","Medium","High"]:
                dfp = preset_bucket(pool, L, risk)
                dec = np.prod([american_to_decimal(x) or 1.0 for x in dfp["american_odds"]]) if not dfp.empty else 1.0
                hit = np.prod([x for x in dfp["q_model"].fillna(0.0)]) if not dfp.empty else 0.0
                st.markdown(
                    f"<span class='pill'>Dec {dec:.2f}</span>"
                    f"<span class='pill'>~Hit {hit*100:.1f}%</span>"
                    f"<span class='pill'>Meets +600? {'✅' if dec>=7 else '❌'}</span>",
                    unsafe_allow_html=True
                )
                if dfp.empty:
                    st.caption("No picks meet this bucket.")
                else:
                    for _, r in dfp.iterrows():
                        render_card(r)

with tabs[3]:
    if pool.empty:
        st.info("No legs.")
    else:
        safety = pool[
            (pool["category"].isin(["Pitcher Ks","Pitcher Walks","Pitcher Outs","Win"])) &
            (pool["american_odds"] <= -180) & (pool["q_model"] >= 0.62)
        ].sort_values(["q_model","edge"], ascending=[False,False]).head(30)
        show = safety.rename(columns={"american_odds":"Odds","q_pct":"q %","p_pct":"Market %"})
        st.dataframe(show[["description","Odds","q %","Market %"]], use_container_width=True, height=420)

with tabs[4]:
    if pool.empty:
        st.info("No legs.")
    else:
        one = pool[(pool["american_odds"] <= -150) & (pool["q_model"] >= 0.62)].sort_values(["edge","q_model"], ascending=[False,False])
        one = preset_bucket(one, 4, "Low")
        dec = np.prod([american_to_decimal(x) or 1.0 for x in one["american_odds"]]) if not one.empty else 1.0
        hit = np.prod([x for x in one["q_model"].fillna(0.0)]) if not one.empty else 0.0
        st.subheader("One‑Tap: 4 legs • Low")
        st.caption(f"≈Hit {hit*100:.1f}% • Dec {dec:.2f} • Meets +600? {'✅' if dec>=7 else '❌'}")
        for _, r in one.iterrows():
            render_card(r)

with tabs[5]:
    ml = pool[pool["category"]=="Moneyline"].copy()
    if ml.empty:
        st.info("No moneyline legs.")
    else:
        ml["Lock?"] = (ml["q_model"] >= 0.65)
        show = ml.rename(columns={"american_odds":"Odds","q_pct":"q %","p_pct":"Market %"})
        st.dataframe(show[["description","Odds","q %","Market %","Lock?"]].sort_values(["Lock?","q %"], ascending=[False,False]),
                     use_container_width=True, height=420)

with tabs[6]:
    st.json({
        "events_df rows": len(get_session_df("events_df")),
        "board_df rows": len(get_session_df("board_df")),
        "props_df rows": len(get_session_df("props_df")),
        "prob_df rows": len(get_session_df("prob_df")),
        "rates_df rows": len(get_session_df("rates_df")),
        "features_df rows": len(get_session_df("features_df")),
        "has_api_key": bool(ODDS_API_KEY),
        "markets": PROPS_MARKETS,
        "auto_proj": auto_proj if 'auto_proj' in globals() else None,
    })