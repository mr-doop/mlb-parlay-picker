# app.py -- MLB Parlay Picker (manual fetch, Apple‑like UI)
# Fixes:
#  - No boolean use of DataFrames in `or` (uses get_session_df)
#  - Props 422: remove dateFormat from /odds props; chunk markets
#  - Robust empty-state handling

import os
import io
import math
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

APP_TITLE = "MLB Parlay Picker -- MVP"
BOOKMAKER = "draftkings"
TZ = "US/Eastern"

ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", os.getenv("ODDS_API_KEY", ""))

GAME_MARKETS = ["h2h", "spreads"]  # ML + Run Line
# Only the 4 pitcher markets (incl. alternates where available)
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

def first_initial_last(full_name: str) -> str:
    s = (full_name or "").replace(".", "").strip()
    if not s:
        return ""
    parts = s.split()
    if len(parts) == 1:
        return parts[0]
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
        "pitcher_hits_allowed":"Hits Allowed",
        "pitcher_hits_allowed_alternate":"Hits Allowed",
        "pitcher_earned_runs":"ER",
        "pitcher_outs":"Outs",
        "pitcher_record_a_win":"Win",
    }
    return M.get(market_key, market_key)

def clean_side(side_text: str) -> str:
    s = (side_text or "").strip().lower()
    if "over" in s: return "Over"
    if "under" in s: return "Under"
    if "yes" in s: return "Yes"
    if "no" in s: return "No"
    return side_text

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
            return f"{team} -{abs(rl):g}" if rl and rl < 0 else f"{team} +{abs(rl):g}" if rl else f"{team} {label} ({matchup})"
        return f"{team} {label} ({matchup})"

    if side in ("Over","Under") and line is not None and not pd.isna(line):
        side_short = "O" if side == "Over" else "U"
        return f"{name} ({team}) {side_short}{line:g} {label}"
    if side in ("Yes","No"):
        return f"{name} ({team}) Win {'YES' if side=='Yes' else 'NO'}"
    return f"{name} ({team}) {label}"

def chips_row(r: pd.Series) -> str:
    odds = r.get("american_odds")
    q = r.get("q_model")
    p = r.get("p_market")
    edge = (q - p) if (pd.notna(q) and pd.notna(p)) else None
    oppK = r.get("opp_k_rate")
    oppBB = r.get("opp_bb_rate")
    pills = []
    pills.append(f"<span class='chip'>Odds {int(odds) if pd.notna(odds) else '--'}</span>")
    pills.append(f"<span class='chip'>q {pct(q)}</span>")
    pills.append(f"<span class='chip'>Market {pct(p)}</span>")
    if edge is not None:
        pills.append(f"<span class='chip'>Edge {edge*100:+.1f}%</span>")
    if pd.notna(oppK): pills.append(f"<span class='chip'>OppK {oppK*100:.1f}%</span>")
    if pd.notna(oppBB): pills.append(f"<span class='chip'>OppBB {oppBB*100:.1f}%</span>")
    return " ".join(pills)

def render_card(r: pd.Series):
    title = build_bet_title(r)
    chips = chips_row(r)
    st.markdown(f"""
    <div class="card">
      <div class="card-title">{st.escape_markdown(title)}</div>
      <div class="card-sub">{chips}</div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Data fetchers (manual)
# -----------------------------

def oddsapi_events(date_str: str) -> pd.DataFrame:
    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/events"
    params = {"apiKey": ODDS_API_KEY, "dateFormat":"iso"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows=[]
    for e in data:
        rows.append({
            "event_id": e.get("id"),
            "commence_time": e.get("commence_time"),
            "home_team": e.get("home_team"),
            "away_team": e.get("away_team"),
            "home_abbr": TEAM_ABBR.get(e.get("home_team",""), None),
            "away_abbr": TEAM_ABBR.get(e.get("away_team",""), None),
        })
    return pd.DataFrame(rows)

def oddsapi_odds_board(markets: list[str]) -> list[dict]:
    """Board (ML/RL). Keep dateFormat off or on? It's fine here but not required."""
    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "oddsFormat": "american",
        "markets": ",".join(markets),
        "bookmakers": BOOKMAKER,
    }
    r = requests.get(url, params=params, timeout=45)
    r.raise_for_status()
    return r.json()

def oddsapi_odds_props_chunked(all_markets: list[str], chunk_size=3) -> list[dict]:
    """Fetch props in small chunks to avoid 422; omit dateFormat entirely."""
    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
    out = []
    for i in range(0, len(all_markets), chunk_size):
        mk = all_markets[i:i+chunk_size]
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "oddsFormat": "american",
            "markets": ",".join(mk),
            "bookmakers": BOOKMAKER,
        }
        r = requests.get(url, params=params, timeout=45)
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            # Surface the server message to the UI
            try:
                st.error(f"Props chunk {mk} fetch failed: {r.status_code} {r.text[:250]}")
            except Exception:
                st.error(f"Props chunk {mk} fetch failed: {e}")
            continue
        js = r.json()
        # merge per-event nodes
        out.extend(js)
    return out

def normalize_board(data_json: list[dict]) -> pd.DataFrame:
    rows=[]
    for ev in data_json:
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

def normalize_props(data_json: list[dict]) -> pd.DataFrame:
    rows=[]
    for ev in data_json:
        eid = ev.get("id")
        home = ev.get("home_team"); away = ev.get("away_team")
        for bm in ev.get("bookmakers", []):
            if bm.get("key") != BOOKMAKER: 
                continue
            for mk in bm.get("markets", []):
                key = mk.get("key","")
                if key not in PROPS_MARKETS:
                    continue
                for oc in mk.get("outcomes", []):
                    side = clean_side(oc.get("name",""))
                    point = oc.get("point")
                    price = oc.get("price")
                    player = oc.get("participant") or oc.get("description") or ""
                    rows.append({
                        "event_id":eid,"category":"Pitcher "+short_market_label(key) if "pitcher_" in key else short_market_label(key),
                        "market_key":key,"side":side,
                        "team_abbr":None,"opp_abbr":None,"is_home":None,
                        "player_name":player,"line":point,"line_run":None,"american_odds":price
                    })
    return pd.DataFrame(rows)

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
                "event_home": TEAM_ABBR.get(home_name,""),
                "event_away": TEAM_ABBR.get(away_name,""),
                "home_pitcher": hp, "away_pitcher": ap
            })
    return pd.DataFrame(rows)

def attach_probables_to_props(props_df: pd.DataFrame, prob_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    if props_df.empty or prob_df.empty or events_df.empty:
        return props_df
    props_df = props_df.copy()

    pairs = events_df[["event_id","home_abbr","away_abbr"]].drop_duplicates()
    pair_map = {r.event_id:(r.home_abbr,r.away_abbr) for _, r in pairs.iterrows()}

    for i, r in props_df.iterrows():
        eid = r.get("event_id"); player = (r.get("player_name") or "").replace(".","").lower().strip()
        if not eid or not player: 
            continue
        home_abbr, away_abbr = pair_map.get(eid, (None,None))
        if not home_abbr or not away_abbr:
            continue
        cand = prob_df[(prob_df["event_home"]==home_abbr) & (prob_df["event_away"]==away_abbr)]
        if cand.empty:
            cand = prob_df[(prob_df["event_home"]==away_abbr) & (prob_df["event_away"]==home_abbr)]
        if not cand.empty:
            row = cand.iloc[0]
            hp = (row.get("home_pitcher") or "").replace(".","").lower()
            ap = (row.get("away_pitcher") or "").replace(".","").lower()
            if hp and player in hp:
                props_df.at[i,"team_abbr"] = row.get("event_home"); props_df.at[i,"opp_abbr"] = row.get("event_away"); props_df.at[i,"is_home"]=True
            elif ap and player in ap:
                props_df.at[i,"team_abbr"] = row.get("event_away"); props_df.at[i,"opp_abbr"] = row.get("event_home"); props_df.at[i,"is_home"]=False
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
    if a1 is None or a2 is None: return (implied_prob_from_american(a1), implied_prob_from_american(a2))
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
    # Others: single leg implied
    mask = df["p_market"].isna()
    df.loc[mask, "p_market"] = df.loc[mask, "american_odds"].apply(implied_prob_from_american)
    return df

def join_opponent_rates(df: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    if rates is None or rates.empty: return df
    return df.merge(rates, left_on="opp_abbr", right_on="team_abbr", how="left").drop(columns=["team_abbr_y"], errors="ignore").rename(columns={"team_abbr_x":"team_abbr"})

def attach_features(df: pd.DataFrame, feat: pd.DataFrame | None) -> pd.DataFrame:
    df = df.copy()
    df["q_model"] = df["p_market"]  # fallback
    if feat is None or feat.empty:
        return df
    f = feat.copy(); f.columns = [c.lower() for c in f.columns]
    if "q_proj" not in f.columns: return df
    if "player_name" in f.columns:
        f["player_key"] = f["player_name"].fillna("").str.replace(".","", regex=False).str.strip().str.lower()
        df["player_key"] = df["player_name"].fillna("").str.replace(".","", regex=False).str.strip().str.lower()
        df = df.merge(f[["player_key","q_proj"]], on="player_key", how="left")
        df["q_model"] = df["q_proj"].where(df["q_proj"].notna(), df["q_model"])
        df.drop(columns=["player_key","q_proj"], inplace=True, errors="ignore")
    return df

def build_pool(board_df: pd.DataFrame, props_df: pd.DataFrame, events_df: pd.DataFrame,
               prob_df: pd.DataFrame, rates_df: pd.DataFrame, feat_df: pd.DataFrame | None) -> pd.DataFrame:
    pieces = []
    if isinstance(board_df, pd.DataFrame) and not board_df.empty: pieces.append(board_df)
    if isinstance(props_df, pd.DataFrame) and not props_df.empty:
        # attach teams to props via probables + events
        props_df = attach_probables_to_props(props_df, prob_df, events_df)
        pieces.append(props_df)
    if not pieces:
        return pd.DataFrame()
    df = pd.concat(pieces, ignore_index=True, sort=False)

    df = compute_probs(df)
    df = join_opponent_rates(df, rates_df)
    df = attach_features(df, feat_df)

    # Ensure columns exist
    for c in ["team_abbr","opp_abbr","player_name","market_key","category","side","line","line_run","american_odds","p_market","q_model"]:
        if c not in df.columns: df[c] = np.nan

    # Title/edge
    df["description"] = df.apply(build_bet_title, axis=1)
    df["edge"] = df["q_model"] - df["p_market"]
    df["q_pct"] = (df["q_model"]*100).round(1)
    df["p_pct"] = (df["p_market"]*100).round(1)

    # Dedupe
    keys = ["event_id","team_abbr","player_name","market_key","side","line","line_run","american_odds"]
    df = df.drop_duplicates(subset=keys).reset_index(drop=True)

    return df

# -----------------------------
# UI
# -----------------------------

APP_CSS = """
<style>
  :root { --ink:#111827; --muted:#6b7280; --chip:#f3f4f6; }
  .metric-wrap{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:8px 0 18px;}
  .metric{background:#fff;border:1px solid #eee;border-radius:14px;padding:12px 10px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.04);}
  .metric .big{font-weight:700;font-size:20px;color:var(--ink);}
  .metric .sub{font-size:12px;color:var(--muted);}
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
st.caption("Manual fetch (save credits) • Apple‑like cards • Vig‑free market % + optional projections (q)")

# State init
for k in ["events_df","board_df","props_df","prob_df","rates_df","features_df"]:
    st.session_state.setdefault(k, None)

with st.expander("How to use", expanded=False):
    st.write("1) Pick date → **Fetch Board**. 2) **Fetch Props** when ready (credits). 3) *(Optional)* Upload features with `player_name,q_proj`. 4) Filter & use tabs. 5) Download CSVs.")

# Controls
cols = st.columns(3)
with cols[0]:
    slate_date = st.date_input("Slate date", value=datetime.now().date())
with cols[1]:
    if st.button("Fetch Board (games + ML/RL)"):
        try:
            events = oddsapi_events(slate_date.strftime("%Y-%m-%d"))
            board_json = oddsapi_odds_board(GAME_MARKETS)
            board_df = normalize_board(board_json)
            prob_df = mlb_probables(datetime.combine(slate_date, datetime.min.time()))
            st.session_state["events_df"] = events
            st.session_state["board_df"] = board_df
            st.session_state["prob_df"] = prob_df
            st.success(f"Fetched board: {len(board_df)} lines across {events['event_id'].nunique()} games.")
        except Exception as e:
            st.error(f"Board fetch failed: {e}")
with cols[2]:
    if st.button("Fetch Props (uses credits)"):
        try:
            props_json = oddsapi_odds_props_chunked(PROPS_MARKETS, chunk_size=3)
            props_df = normalize_props(props_json)
            if props_df.empty:
                st.warning("Props response was empty (check plan/markets/time).")
            st.session_state["props_df"] = props_df
            st.success(f"Fetched props: {len(props_df)} rows.")
        except Exception as e:
            st.error(f"Props fetch error: {e}")

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

# Downloads after fetch
dl1, dl2 = st.columns(2)
bd = get_session_df("board_df")
if not bd.empty:
    dl1.download_button("Download Board CSV", bd.to_csv(index=False).encode(), file_name=f"board_{slate_date}.csv", mime="text/csv")
pdn = get_session_df("props_df")
if not pdn.empty:
    dl2.download_button("Download Props CSV", pdn.to_csv(index=False).encode(), file_name=f"props_{slate_date}.csv", mime="text/csv")

# Build pool
events_df = get_session_df("events_df")
board_df  = get_session_df("board_df")
props_df  = get_session_df("props_df")
prob_df   = get_session_df("prob_df")

# Opponent rates (cached)
if st.session_state.get("rates_df") is None:
    try:
        st.session_state["rates_df"] = mlb_team_rates_last_days(days=21)
    except Exception:
        st.session_state["rates_df"] = pd.DataFrame()
rates_df = get_session_df("rates_df")
feat_df  = get_session_df("features_df")

pool_base = build_pool(board_df, props_df, events_df, prob_df, rates_df, feat_df)

# Filters
st.markdown("### Filters")

games_det = []
if not events_df.empty:
    for _, r in events_df.iterrows():
        games_det.append(f"{fmt_team(r['away_abbr'])}@{fmt_team(r['home_abbr'])}")

select_all_games = st.checkbox("Select all games", value=True)
games_pick = st.multiselect("Games", games_det, default=(games_det if select_all_games else []))

teams_all = sorted(set([fmt_team(x) for x in pool_base["team_abbr"].dropna().unique().tolist()]))
select_all_teams = st.checkbox("Select all teams", value=True)
teams_pick = st.multiselect("Teams", teams_all, default=(teams_all if select_all_teams else []))

cat_options = ["Moneyline","Run Line","Pitcher Ks","Pitcher Walks","Pitcher Hits Allowed","Pitcher ER","Pitcher Outs","Win"]
def cat_from_row(r):
    if r["category"] in ("Moneyline","Run Line"): return r["category"]
    mk = r["market_key"]
    if "strikeouts" in mk: return "Pitcher Ks"
    if "walks" in mk: return "Pitcher Walks"
    if "hits_allowed" in mk: return "Pitcher Hits Allowed"
    if "earned_runs" in mk: return "Pitcher ER"
    if "outs" in mk: return "Pitcher Outs"
    if "record_a_win" in mk: return "Win"
    return r["category"]

pool = pool_base.copy()
if not pool.empty:
    pool["ui_cat"] = pool.apply(cat_from_row, axis=1)

cat_pick = st.multiselect("Categories", cat_options, default=["Moneyline","Run Line"])
odds_min, odds_max = st.slider("American odds", -700, 700, (-700, 700), step=5)

def row_game(r):
    a = fmt_team(r.get("opp_abbr")); h = fmt_team(r.get("team_abbr"))
    if r.get("is_home") is True:  return f"{a}@{h}"
    if r.get("is_home") is False: return f"{h}@{a}"
    return f"{a}@{h}"

if not pool.empty:
    pool["game_key"] = pool.apply(row_game, axis=1)
    pool = pool[
        pool["american_odds"].between(odds_min, odds_max, inclusive="both") &
        pool["game_key"].isin(games_pick if games_pick else pool["game_key"].unique()) &
        pool["team_abbr"].apply(lambda x: fmt_team(x) in (teams_pick if teams_pick else teams_all)) &
        pool["ui_cat"].isin(cat_pick if cat_pick else cat_options)
    ].reset_index(drop=True)

st.markdown("#### Coverage (from MLB schedule) -- games detected")
st.caption(", ".join(games_det) if games_det else "No games found for this date.")

# Tabs
tabs = st.tabs(["Candidates", "Top 20", "Parlay Presets", "Alt Line Safety Board", "One‑Tap Ticket", "ML Winners & RL Locks", "Debug"])

with tabs[0]:
    if pool.empty:
        st.info("No legs. Fetch board/props or relax filters.")
    else:
        show = pool.rename(columns={"american_odds":"Odds","q_pct":"q %","p_pct":"Market %"})
        st.dataframe(show[["description","category","Odds","q %","Market %","edge"]].sort_values(["edge","q %"], ascending=[False,False]),
                     use_container_width=True, height=480)

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
                st.markdown(f"<span class='pill'>Dec {dec:.2f}</span><span class='pill'>~Hit {hit*100:.1f}%</span><span class='pill'>Meets +600? {'✅' if dec>=7 else '❌'}</span>", unsafe_allow_html=True)
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
            (pool["ui_cat"].isin(["Pitcher Ks","Pitcher Walks","Pitcher Hits Allowed","Pitcher ER","Pitcher Outs"])) &
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
        "events_df": len(get_session_df("events_df")),
        "board_df": len(get_session_df("board_df")),
        "props_df": len(get_session_df("props_df")),
        "prob_df": len(get_session_df("prob_df")),
        "rates_df": len(get_session_df("rates_df")),
        "features_df": len(get_session_df("features_df")),
        "has_api_key": bool(ODDS_API_KEY),
    })