# app.py — MLB Parlay Picker (clean UI + grouped cards + persistence + enhanced model)
# -----------------------------------------------------------------------------
# - Manual fetch buttons (protect OddsAPI credits)
# - Event/board from OddsAPI; props via /events/{id}/odds
# - Alternates (Ks, BB) included
# - Feature enrichment (opp K/BB, park, weather, pitcher form)
# - Enhanced logistic model -> q_final; EV & Edge
# - Grouped player-prop cards: one primary line + Alt Lines dropdown
# - Top 20 remains (but grouped view is default for props); ML/RL separate
# - Auto-snapshot to server (zip) and download/upload

import os, re, json, math, glob, zipfile
from io import BytesIO
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

from parlay.feature_join import (
    apply_enhanced_model,  # uses parlay/model_weights.json
)

# ---------------------------- Config ---------------------------------
APP_TTL_MIN = 30
BOOK   = "draftkings"
REGION = "us"
SPORT  = "baseball_mlb"

PROP_MARKETS = [
    "pitcher_strikeouts",
    "pitcher_strikeouts_alternate",
    "pitcher_walks",
    "pitcher_walks_alternate",
    "pitcher_outs",
    "pitcher_record_a_win",
]

SNAP_DIR = "snapshots"

# --------------------------- Utilities --------------------------------
def _s(x): return "" if x is None else str(x)
def _cap3(s): return _s(s).upper()[:3]
def american_to_decimal(a):
    try:
        a = int(a);  return (1 + a/100) if a > 0 else (1 + 100/(-a))
    except: return None
def american_to_prob(a):
    try:
        a = int(a);  return 100/(a+100) if a>=0 else (-a)/((-a)+100)
    except: return None
def pct(x, d=1):
    try: return f"{float(x):.{d}%}"
    except: return "--"
def first_init_last(full_name:str) -> str:
    s = _s(full_name).strip()
    if not s: return ""
    parts = s.split()
    if len(parts) == 1: return parts[0].title()
    return f"{parts[0][0].upper()}. {' '.join(p.title() for p in parts[1:])}"

TEAM_ABBR = {
    "Baltimore Orioles":"BAL","Boston Red Sox":"BOS","New York Yankees":"NYY","Tampa Bay Rays":"TBR","Toronto Blue Jays":"TOR",
    "Chicago White Sox":"CWS","Cleveland Guardians":"CLE","Detroit Tigers":"DET","Kansas City Royals":"KC","Minnesota Twins":"MIN",
    "Houston Astros":"HOU","Los Angeles Angels":"LAA","Oakland Athletics":"OAK","Seattle Mariners":"SEA","Texas Rangers":"TEX",
    "Atlanta Braves":"ATL","Miami Marlins":"MIA","New York Mets":"NYM","Philadelphia Phillies":"PHI","Washington Nationals":"WSH",
    "Chicago Cubs":"CHC","Cincinnati Reds":"CIN","Milwaukee Brewers":"MIL","Pittsburgh Pirates":"PIT","St. Louis Cardinals":"STL",
    "Arizona Diamondbacks":"ARI","Colorado Rockies":"COL","Los Angeles Dodgers":"LAD","San Diego Padres":"SDP","San Francisco Giants":"SFG",
}
ABBR_SET = set(TEAM_ABBR.values())

# Coarse park knobs (K/BB factors)
PARK = {
    "ARI":{"k":0.99,"bb":1.00},"ATL":{"k":0.99,"bb":0.98},"BAL":{"k":1.01,"bb":1.00},
    "BOS":{"k":0.98,"bb":1.01},"CHC":{"k":1.00,"bb":1.00},"CWS":{"k":1.00,"bb":0.99},
    "CIN":{"k":0.99,"bb":1.00},"CLE":{"k":1.01,"bb":1.01},"COL":{"k":0.95,"bb":1.02},
    "DET":{"k":1.02,"bb":1.01},"HOU":{"k":0.99,"bb":0.99},"KC":{"k":1.00,"bb":1.00},
    "LAA":{"k":1.00,"bb":0.99},"LAD":{"k":1.00,"bb":0.99},"MIA":{"k":1.02,"bb":1.01},
    "MIL":{"k":0.99,"bb":1.00},"MIN":{"k":1.01,"bb":1.00},"NYM":{"k":1.00,"bb":1.01},
    "NYY":{"k":0.99,"bb":0.99},"OAK":{"k":1.01,"bb":1.02},"PHI":{"k":0.99,"bb":0.99},
    "PIT":{"k":1.01,"bb":1.01},"SDP":{"k":1.02,"bb":1.00},"SFG":{"k":1.02,"bb":1.01},
    "SEA":{"k":1.01,"bb":1.00},"STL":{"k":1.00,"bb":1.00},"TBR":{"k":1.00,"bb":1.00},
    "TEX":{"k":0.99,"bb":0.99},"TOR":{"k":1.00,"bb":0.99},"WSH":{"k":1.00,"bb":1.01},
}

BALLPARK_COORDS = {
    "ARI": (33.4455, -112.0667),"ATL":(33.8907,-84.4677),"BAL":(39.2839,-76.6217),
    "BOS": (42.3467, -71.0972),"CHC":(41.9484,-87.6553),"CWS":(41.8300,-87.6339),
    "CIN": (39.0979, -84.5074),"CLE":(41.4962,-81.6852),"COL":(39.7569,-104.9669),
    "DET": (42.3390, -83.0485),"HOU":(29.7572,-95.3558),"KC": (39.0516,-94.4803),
    "LAA": (33.8003,-117.8827),"LAD":(34.0739,-118.2400),"MIA":(25.7781,-80.2197),
    "MIL": (43.0280, -87.9712),"MIN":(44.9817,-93.2776),"NYM":(40.7571,-73.8458),
    "NYY": (40.8296, -73.9262),"OAK":(37.7516,-122.2005),"PHI":(39.9057,-75.1665),
    "PIT": (40.4469, -80.0057),"SDP":(32.7073,-117.1566),"SFG":(37.7786,-122.3893),
    "SEA": (47.5914, -122.3325),"STL":(38.6226,-90.1928),"TBR":(27.7683,-82.6533),
    "TEX": (32.7473, - 97.0842),"TOR":(43.6414,-79.3894),"WSH":(38.8730,-77.0074),
}

def api_key():
    try: return st.secrets["ODDS_API_KEY"]
    except Exception: return ""

def _get(url, params, timeout=15):
    r = requests.get(url, params=params, timeout=timeout)
    status = r.status_code
    try:
        r.raise_for_status()
        return status, r.json(), None
    except Exception as e:
        return status, None, str(e)

# -------------------- External context (weather/opp rates) ----------
@st.cache_data(show_spinner=False, ttl=900)
def get_weather_factor(team_abbr: str) -> float:
    ab = _cap3(team_abbr)
    latlon = BALLPARK_COORDS.get(ab)
    if not latlon: return 1.0
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params=dict(latitude=latlon[0], longitude=latlon[1],
                        hourly="wind_speed_10m,wind_direction_10m", forecast_days=1),
            timeout=10
        )
        r.raise_for_status()
        js = r.json()
        wind = float(js["hourly"]["wind_speed_10m"][0])
        wdir = float(js["hourly"]["wind_direction_10m"][0])
        factor = 1.0 + max(0.0, wind/20.0) * (1.0 if 200 <= wdir <= 340 else -0.5)
        return max(0.85, min(1.15, factor))
    except Exception:
        return 1.0

@st.cache_data(show_spinner=False, ttl=6*3600)
def get_opponent_rates() -> pd.DataFrame:
    # Fallback-safe (returns league-average if pybaseball unavailable)
    try:
        from pybaseball import team_batting
        yr = datetime.now().year
        tb = team_batting(yr)
        tb["PA"] = tb["AB"] + tb["BB"] + tb["HBP"].fillna(0) + tb["SF"].fillna(0)
        tb["opp_k_pct"] = tb["SO"] / tb["PA"]
        tb["opp_bb_pct"] = tb["BB"] / tb["PA"]
        rows = []
        for _, r in tb.iterrows():
            name = _s(r.get("Team"))
            abbr = TEAM_ABBR.get(name)
            if not abbr:
                for k,v in TEAM_ABBR.items():
                    if name in k or k in name: abbr = v; break
            if abbr:
                rows.append(dict(team_abbr=abbr,
                                 opp_k_pct=float(r["opp_k_pct"]),
                                 opp_bb_pct=float(r["opp_bb_pct"])))
        df = pd.DataFrame(rows)
        if df.empty: raise RuntimeError("fallback")
        return df
    except Exception:
        return pd.DataFrame([dict(team_abbr=ab, opp_k_pct=0.22, opp_bb_pct=0.08)
                             for ab in ABBR_SET])

@st.cache_data(show_spinner=False, ttl=6*3600)
def pitcher_last5(name: str) -> dict:
    """Tiny helper for pitcher form; robust to API errors."""
    try:
        from pybaseball import playerid_lookup, statcast_pitcher
        nm = _s(name)
        if not nm: return {}
        fs = nm.split()
        first, last = fs[0], fs[-1]
        ids = playerid_lookup(last=last, first=first)
        if ids is None or ids.empty: return {}
        mlbam = int(ids.iloc[0]["key_mlbam"])
        end = datetime.now().date()
        start = end - timedelta(days=60)
        df = statcast_pitcher(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), mlbam)
        if df is None or df.empty: return {}
        df["isK"] = df["description"].str.contains("strikeout", na=False)
        df["isBB"] = df["description"].str.contains("walk", na=False)
        g = df.groupby("game_date").agg(pitches=("pitch_number","max"),
                                        Ks=("isK","sum"), BB=("isBB","sum"),
                                        outs=("outs_when_up","max"))
        g = g.reset_index().sort_values("game_date", ascending=False).head(5)
        if g.empty: return {}
        ip = (g["outs"].fillna(0)/3.0).mean()
        pc = g["pitches"].mean()
        k9 = (g["Ks"].sum() / max(1.0, (g["outs"].fillna(0).sum()/3.0))) * 9.0
        bb9 = (g["BB"].sum() / max(1.0, (g["outs"].fillna(0).sum()/3.0))) * 9.0
        return {"exp_ip":float(ip), "exp_pitches":float(pc),
                "form_k9":float(k9), "form_bb9":float(bb9)}
    except Exception:
        return {}

# --------------------------- Persistence ------------------------------
def save_snapshot_to_zip(events_df, board_df, props_df, rates_df) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        if isinstance(events_df, pd.DataFrame) and not events_df.empty:
            z.writestr("events.csv", events_df.to_csv(index=False))
        if isinstance(board_df, pd.DataFrame) and not board_df.empty:
            z.writestr("board.csv", board_df.to_csv(index=False))
        if isinstance(props_df, pd.DataFrame) and not props_df.empty:
            z.writestr("props.csv", props_df.to_csv(index=False))
        if isinstance(rates_df, pd.DataFrame) and not rates_df.empty:
            z.writestr("rates.csv", rates_df.to_csv(index=False))
    buf.seek(0)
    return buf.getvalue()

def load_snapshot_from_zip(file) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        with zipfile.ZipFile(file) as z:
            def _rd(name):
                try:
                    with z.open(name) as f:
                        return pd.read_csv(f)
                except: return pd.DataFrame()
            return _rd("events.csv"), _rd("board.csv"), _rd("props.csv"), _rd("rates.csv")
    except:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def write_server_snapshot(bytes_data: bytes) -> str:
    os.makedirs(SNAP_DIR, exist_ok=True)
    fname = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    path = os.path.join(SNAP_DIR, fname)
    with open(path, "wb") as f:
        f.write(bytes_data)
    return path

def last_server_snapshot_path() -> str|None:
    files = sorted(glob.glob(os.path.join(SNAP_DIR, "snapshot_*.zip")))
    return files[-1] if files else None

def autosave_snapshot(ss) -> str|None:
    try:
        bytes_zip = save_snapshot_to_zip(ss.get("events_df"), ss.get("board_df"),
                                         ss.get("props_df"), ss.get("rates_df"))
        if bytes_zip and len(bytes_zip) > 0:
            ss["last_snapshot_bytes"] = bytes_zip
            path = write_server_snapshot(bytes_zip)
            ss["last_snapshot_path"] = path
            ss["last_snapshot_ts"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return path
    except Exception as e:
        ss["last_snapshot_error"] = str(e)
    return None

# --------------------------- Fetchers --------------------------------
def _pretty_category(mkey:str) -> str:
    mkey = _s(mkey)
    if "strikeout" in mkey: return "Pitcher Ks"
    if "walks" in mkey and "pitcher" in mkey: return "Pitcher BB"
    if "outs" in mkey and "pitcher" in mkey: return "Pitcher Outs"
    if "record_a_win" in mkey: return "Pitcher Win"
    return mkey

@st.cache_data(show_spinner=False, ttl=APP_TTL_MIN*60)
def fetch_events() -> pd.DataFrame:
    key = api_key()
    if not key: return pd.DataFrame()
    u = f"https://api.the-odds-api.com/v4/sports/{SPORT}/events"
    status, js, err = _get(u, dict(apiKey=key, dateFormat="iso"))
    if err or not isinstance(js, list): return pd.DataFrame()
    rows = []
    for ev in js:
        home = TEAM_ABBR.get(ev.get("home_team"), _cap3(ev.get("home_team")))
        away = TEAM_ABBR.get(ev.get("away_team"), _cap3(ev.get("away_team")))
        rows.append(dict(
            event_id=ev.get("id"),
            start=ev.get("commence_time"),
            home_abbr=home, away_abbr=away,
            matchup=f"{away}@{home}"
        ))
    return pd.DataFrame(rows).drop_duplicates("event_id")

@st.cache_data(show_spinner=False, ttl=APP_TTL_MIN*60)
def fetch_board() -> pd.DataFrame:
    key = api_key()
    if not key: return pd.DataFrame()
    u = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
    status, js, err = _get(u, dict(apiKey=key, regions=REGION, bookmakers=BOOK,
                                   oddsFormat="american", dateFormat="iso", markets="h2h,spreads"))
    if err or not isinstance(js, list): return pd.DataFrame()
    rows = []
    for g in js:
        eid = g.get("id")
        home = TEAM_ABBR.get(g.get("home_team"), _cap3(g.get("home_team")))
        away = TEAM_ABBR.get(g.get("away_team"), _cap3(g.get("away_team")))
        matchup = f"{away}@{home}"
        for bk in g.get("bookmakers", []):
            if bk.get("key") != BOOK: continue
            for m in bk.get("markets", []):
                mkey = m.get("key")
                if mkey not in ("h2h","spreads"): continue
                for o in m.get("outcomes", []):
                    price = o.get("price")
                    line = o.get("point")
                    name = o.get("name")
                    category = "Moneyline" if mkey=="h2h" else "Run Line"
                    team = TEAM_ABBR.get(name, _cap3(name))
                    rows.append(dict(
                        event_id=eid, market_type=category, team_abbr=team, player_name=None,
                        side=None, line=line, american_odds=price, decimal_odds=american_to_decimal(price),
                        p_market=american_to_prob(price), game_id=eid, home_abbr=home, away_abbr=away,
                        description=f"{team} {'ML' if category=='Moneyline' else (f'{line:+.1f}' if line is not None else '')} ({matchup})",
                        category=category, matchup=matchup
                    ))
    df = pd.DataFrame(rows)
    if df.empty: return df
    return df.drop_duplicates(subset=["event_id","market_type","team_abbr","line","american_odds"])

@st.cache_data(show_spinner=False, ttl=APP_TTL_MIN*60)
def fetch_props_by_events_cached(event_ids_key: tuple, markets_key: tuple) -> pd.DataFrame:
    return _fetch_props_by_events_network(list(event_ids_key), list(markets_key))

def _fetch_props_by_events_network(event_ids:list[str], markets:list[str]) -> pd.DataFrame:
    key = api_key()
    if not key or not event_ids: return pd.DataFrame()
    rows = []
    for eid in event_ids:
        u = f"https://api.the-odds-api.com/v4/sports/{SPORT}/events/{eid}/odds"
        params = dict(apiKey=key, regions=REGION, bookmakers=BOOK,
                      oddsFormat="american", dateFormat="iso", markets=",".join(markets))
        status, js, err = _get(u, params)
        if err or not isinstance(js, dict):
            continue
        home = TEAM_ABBR.get(js.get("home_team"), _cap3(js.get("home_team")))
        away = TEAM_ABBR.get(js.get("away_team"), _cap3(js.get("away_team")))
        matchup = f"{away}@{home}"
        for bk in js.get("bookmakers", []):
            if bk.get("key") != BOOK: continue
            for m in bk.get("markets", []):
                mkey = m.get("key")
                if mkey not in markets: continue
                for o in m.get("outcomes", []):
                    price = o.get("price"); line  = o.get("point")
                    side  = (_s(o.get("name")).lower() if o.get("name") else None)
                    player = o.get("description") or o.get("participant") or o.get("player")
                    if isinstance(player, str) and (" Over" in player or " Under" in player):
                        player = re.split(r"\sOver|\sUnder", player)[0]
                    player = _s(player).strip()
                    team = TEAM_ABBR.get(o.get("team") or o.get("participant"), "")
                    rows.append(dict(
                        event_id=eid, market_type=mkey, team_abbr=team, player_name=player,
                        side=side, line=line, american_odds=price,
                        decimal_odds=american_to_decimal(price), p_market=american_to_prob(price),
                        game_id=eid, home_abbr=home, away_abbr=away,
                        description="", category=_pretty_category(mkey), matchup=matchup
                    ))
    return pd.DataFrame(rows)

# ----------------------- Description / Pool --------------------------
def _clean_market_text(mkey:str) -> str:
    mkey = _s(mkey)
    if "strikeout" in mkey: return "Ks"
    if "walk" in mkey and "pitcher" in mkey: return "BB"
    if "outs" in mkey and "pitcher" in mkey: return "Outs"
    if "record_a_win" in mkey: return "Win"
    return mkey

def describe_row(row: pd.Series) -> str:
    mtext = _clean_market_text(row.get("market_type",""))
    side = _s(row.get("side")).title() if row.get("side") else ""
    line = row.get("line")
    matchup = row.get("matchup") or f"{_cap3(row.get('away_abbr'))}@{_cap3(row.get('home_abbr'))}"
    team = _cap3(row.get("team_abbr"))
    if row.get("player_name"):
        nm = first_init_last(row["player_name"])
        if line not in (None, float("nan")):
            bet = f"{nm} ({team or '--'}) {side} {line:g} {mtext}"
        else:
            bet = f"{nm} ({team or '--'}) {side} {mtext}"
    else:
        if row.get("market_type") == "Moneyline":
            bet = f"{team} ML ({matchup})"
        else:
            bet = f"{team} {row.get('line',0):+0.1f} ({matchup})" if row.get("line") is not None else f"{team} RL ({matchup})"
    return bet

def build_pool(board_df: pd.DataFrame, props_df: pd.DataFrame) -> pd.DataFrame:
    b = board_df.copy() if isinstance(board_df, pd.DataFrame) else pd.DataFrame()
    p = props_df.copy() if isinstance(props_df, pd.DataFrame) else pd.DataFrame()
    cols = ["event_id","market_type","team_abbr","player_name","side","line","american_odds",
            "decimal_odds","p_market","home_abbr","away_abbr","category","matchup"]
    if not b.empty: b = b[cols]
    else: b = pd.DataFrame(columns=cols)
    if not p.empty: p = p[cols]
    else: p = pd.DataFrame(columns=cols)

    df = pd.concat([b, p], ignore_index=True, sort=False)
    if df.empty: return df

    df["p_market"] = df["p_market"].fillna(df["american_odds"].map(american_to_prob))
    df["side"] = df["side"].fillna("")
    df["team_abbr"] = df["team_abbr"].where(df["team_abbr"].isin(ABBR_SET), df["home_abbr"])
    df["description"] = df.apply(describe_row, axis=1)
    df["player_key"] = df["player_name"].fillna("").str.replace(r"\.","", regex=True).str.strip().str.lower()
    df["leg_id"] = (df["event_id"].astype(str)+"|"+df["market_type"].astype(str)+"|"+
                    df["team_abbr"].astype(str)+"|"+df["player_name"].astype(str)+"|"+
                    df["side"].astype(str)+"|"+df["line"].astype(str)+"|"+df["american_odds"].astype(str))
    df = df.drop_duplicates(subset=["leg_id"]).reset_index(drop=True)
    return df

# --------------------------- Streamlit shell --------------------------
st.set_page_config(page_title="MLB Parlay Picker", page_icon="⚾", layout="wide")
st.markdown("<style>body{background:#fafafa}</style>", unsafe_allow_html=True)
st.title("MLB Parlay Picker — Clean")

ss = st.session_state
for k, default in [
    ("events_df", None), ("board_df", None), ("props_df", None), ("rates_df", None),
    ("picks", {}), ("lock_data", False),
    ("last_snapshot_bytes", None), ("last_snapshot_path", None), ("last_snapshot_ts", None),
    ("auto_snapshot", True),
]:
    ss.setdefault(k, default)

# Sidebar — fetch & persistence
with st.sidebar:
    st.subheader("Data")
    st.caption("Manual fetch keeps OddsAPI credits under your control.")
    colA, colB = st.columns(2)
    with colA:
        btn_events = st.button("Fetch events", disabled=ss["lock_data"])
        btn_board  = st.button("Fetch board",  disabled=ss["lock_data"])
    with colB:
        btn_props  = st.button("Fetch props",  disabled=ss["lock_data"])
        btn_rates  = st.button("Opp. rates",   disabled=ss["lock_data"])

    st.markdown("---")
    st.subheader("Persistence")
    ss["auto_snapshot"] = st.toggle("Auto‑save after fetch", value=ss["auto_snapshot"])
    ss["lock_data"] = st.toggle("🔒 Data Lock", value=ss["lock_data"])

    colS1, colS2 = st.columns(2)
    with colS1: save_snap = st.button("Save snapshot")
    with colS2: restore_snap = st.button("Restore last snapshot")
    up_zip = st.file_uploader("Load snapshot (zip)", type=["zip"])

    if ss.get("last_snapshot_bytes"):
        st.download_button(
            "Download latest auto‑snapshot",
            data=ss["last_snapshot_bytes"],
            file_name=f"mlb_snapshot_latest.zip",
            mime="application/zip",
            key="dl_latest_auto"
        )
        if ss.get("last_snapshot_ts"):
            st.caption(f"Auto‑saved: {ss['last_snapshot_ts']}")

    if save_snap:
        bytes_zip = save_snapshot_to_zip(ss.get("events_df"), ss.get("board_df"),
                                         ss.get("props_df"), ss.get("rates_df"))
        ss["last_snapshot_bytes"] = bytes_zip
        path = write_server_snapshot(bytes_zip)
        ss["last_snapshot_path"] = path
        ss["last_snapshot_ts"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.success(f"Snapshot saved: {path}")
        st.download_button("Download snapshot zip", data=bytes_zip,
                           file_name=f"mlb_snapshot_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                           mime="application/zip", key="dl_manual_snap")

    if restore_snap:
        p = last_server_snapshot_path()
        if p and os.path.exists(p):
            with open(p, "rb") as f:
                E,B,P,R = load_snapshot_from_zip(f)
            if not ss["lock_data"]:
                ss["events_df"], ss["board_df"], ss["props_df"], ss["rates_df"] = E,B,P,R
            st.success(f"Restored: {p}")
        else:
            st.warning("No server snapshot found.")

    if up_zip is not None:
        E,B,P,R = load_snapshot_from_zip(up_zip)
        if not ss["lock_data"]:
            ss["events_df"], ss["board_df"], ss["props_df"], ss["rates_df"] = E,B,P,R
        st.success("Loaded uploaded snapshot.")

def _auto_after_fetch(label: str):
    if ss["auto_snapshot"]:
        path = autosave_snapshot(ss)
        if path:
            st.sidebar.success(f"Auto‑saved after {label} → {os.path.basename(path)}")
        else:
            st.sidebar.warning(f"Auto‑save after {label} failed.")

if btn_events and not ss["lock_data"]:
    ss["events_df"] = fetch_events(); _auto_after_fetch("events")
if btn_board and not ss["lock_data"]:
    ss["board_df"]  = fetch_board();  _auto_after_fetch("board")
if btn_props and not ss["lock_data"]:
    eids = ss["events_df"]["event_id"].tolist() if isinstance(ss.get("events_df"), pd.DataFrame) and not ss["events_df"].empty else []
    ek = tuple(sorted(set(_s(e) for e in eids)))
    mk = tuple(sorted(set(_s(m) for m in PROP_MARKETS)))
    ss["props_df"] = fetch_props_by_events_cached(ek, mk)
    _auto_after_fetch("props")
if btn_rates and not ss["lock_data"]:
    ss["rates_df"] = get_opponent_rates(); _auto_after_fetch("opponent rates")

# --------------------------- Build pool + features -------------------
events_df = ss["events_df"] if isinstance(ss["events_df"], pd.DataFrame) else pd.DataFrame()
board_df  = ss["board_df"]  if isinstance(ss["board_df"],  pd.DataFrame) else pd.DataFrame()
props_df  = ss["props_df"]  if isinstance(ss["props_df"],  pd.DataFrame) else pd.DataFrame()
rates_df  = ss["rates_df"]  if isinstance(ss["rates_df"],  pd.DataFrame) else get_opponent_rates()

pool_base = build_pool(board_df, props_df)

def enrich_features(pool: pd.DataFrame) -> pd.DataFrame:
    if pool is None or pool.empty: return pool
    df = pool.copy()

    # Opponent K/BB join (by team_abbr)
    if not rates_df.empty and "team_abbr" in df.columns:
        df = df.merge(rates_df, on="team_abbr", how="left")
    df["opp_k_pct"]  = df["opp_k_pct"].fillna(0.22)
    df["opp_bb_pct"] = df["opp_bb_pct"].fillna(0.08)

    # Park factors (ballpark proxy = team home for our purposes)
    df["park_k_factor"]  = df["team_abbr"].map(lambda t: PARK.get(_cap3(t), {"k":1.0})["k"]).fillna(1.0)
    df["park_bb_factor"] = df["team_abbr"].map(lambda t: PARK.get(_cap3(t), {"bb":1.0})["bb"]).fillna(1.0)

    # Weather factor (cached API)
    df["weather_factor"] = df["team_abbr"].map(get_weather_factor).fillna(1.0)

    # Pitcher form/expectation (last 5)
    def _form_map(name):
        try: return pitcher_last5(name)
        except Exception: return {}
    forms = df["player_name"].fillna("").apply(_form_map)
    df["exp_ip"]       = forms.map(lambda d: d.get("exp_ip", np.nan))
    df["exp_pitches"]  = forms.map(lambda d: d.get("exp_pitches", np.nan))
    df["form_k9"]      = forms.map(lambda d: d.get("form_k9", np.nan))
    df["form_bb9"]     = forms.map(lambda d: d.get("form_bb9", np.nan))

    df["exp_ip"]      = df["exp_ip"].fillna(5.6)
    df["exp_pitches"] = df["exp_pitches"].fillna(92.0)

    # Placeholders (can be wired to bullpen workload ETL later)
    df["pitch_mix_fit"]  = 0.0
    df["bullpen_fatigue"]= 0.0

    return df

pool_feat = enrich_features(pool_base)
enh = apply_enhanced_model(pool_feat, weights_path="parlay/model_weights.json", blend=0.35)

# ------------------------ Grouped player‑prop cards -------------------
PROP_FAMS = {"Pitcher Ks":"PITCHER_KS","Pitcher BB":"PITCHER_BB","Pitcher Outs":"PITCHER_OUTS","Pitcher Win":"PITCHER_WIN"}

def _fam_key(cat: str) -> tuple[str|None, str|None]:
    c = (cat or "").lower()
    if "strikeout" in c: return "PITCHER_KS","Pitcher Ks"
    if "walk" in c and "pitcher" in c: return "PITCHER_BB","Pitcher BB"
    if "outs" in c and "pitcher" in c: return "PITCHER_OUTS","Pitcher Outs"
    if "record_a_win" in c: return "PITCHER_WIN","Pitcher Win"
    return None, None

def _choose_primary(line_map: dict) -> float|None:
    """Pick the line whose Over/Under market is closest to 50/50."""
    best_line, best_score = None, 1e9
    for ln, sides in line_map.items():
        # Use market probs to judge book balance near 0.5
        cands = []
        po = sides.get("over", {}).get("p")
        pu = sides.get("under", {}).get("p")
        if po is not None: cands.append(abs(po - 0.5))
        if pu is not None: cands.append(abs(pu - 0.5))
        if not cands: 
            # final fallback: odds closest to -110
            oov = sides.get("over", {}).get("odds")
            ouv = sides.get("under", {}).get("odds")
            if oov is None and ouv is None: continue
            tie = min(abs((oov if oov is not None else -110) + 110),
                      abs((ouv if ouv is not None else -110) + 110))
            score = 0.50 + tie / 1000.0
        else:
            score = float(np.nanmin(cands))
        if score < best_score:
            best_score, best_line = score, ln
    return best_line

def group_props_rows(df: pd.DataFrame):
    """Return a list of grouped dicts for cards."""
    if df is None or df.empty: return []
    props = df[df["player_name"].fillna("").ne("")].copy()
    # attach family labels
    fams = props["market_type"].apply(_fam_key)
    props["fam_key"]   = fams.map(lambda t: t[0])
    props["fam_label"] = fams.map(lambda t: t[1])
    props = props[props["fam_key"].notna()]
    props["line_float"] = pd.to_numeric(props["line"], errors="coerce")

    groups = []
    gcols = ["event_id","player_key","player_name","team_abbr","fam_key","fam_label"]
    for gkey, gdf in props.groupby(gcols, dropna=False):
        event_id, pkey, pname, team, fkey, flabel = gkey
        # Build map line -> {over: {...}, under:{...}}
        line_map = {}
        for _, r in gdf.iterrows():
            ln   = float(r["line"]) if pd.notna(r["line"]) else np.nan
            side = (r.get("side") or "").lower()
            if side not in ("over","under","yes","no"): 
                continue
            side = "over" if side in ("over","yes") else "under"
            entry = dict(
                q=float(r.get("q_final", r.get("q_model", np.nan))),
                p=float(r.get("p_market", np.nan)),
                odds=(int(r["american_odds"]) if pd.notna(r["american_odds"]) else None),
                leg_id=r["leg_id"], desc=r["description"],
                line=ln, team=_cap3(team)
            )
            line_map.setdefault(ln, {})[side] = entry

        if not line_map:
            continue

        primary_ln = _choose_primary(line_map)
        alt_lines  = sorted([ln for ln in line_map.keys() if ln != primary_ln], key=lambda x: (np.isnan(x), x))

        groups.append(dict(
            event_id=event_id,
            player_key=pkey,
            player_name=pname,
            team_abbr=_cap3(team),
            fam_key=fkey,
            fam_label=flabel,
            primary_line=primary_ln,
            primary=line_map.get(primary_ln, {}),
            alts=[(ln, line_map[ln]) for ln in alt_lines],
        ))
    # Sort by best primary edge
    def best_primary_edge(g):
        over = g["primary"].get("over", {})
        under = g["primary"].get("under", {})
        e_over  = (over.get("q", np.nan)  - over.get("p", np.nan))
        e_under = (under.get("q", np.nan) - under.get("p", np.nan))
        return max(e_over, e_under)
    groups.sort(key=lambda g: best_primary_edge(g), reverse=True)
    return groups

def _chip(text): 
    return f'<span style="background:#f5f6f7;border:1px solid #eee;border-radius:999px;padding:3px 8px;font-size:12px;color:#111;">{text}</span>'

def render_grouped_cards(df_filtered: pd.DataFrame, key_prefix: str = "grp"):
    groups = group_props_rows(df_filtered)
    if not groups:
        st.info("No player props available under current filters.")
        return
    for idx, g in enumerate(groups, 1):
        short = first_init_last(g["player_name"])
        team  = g["team_abbr"]
        fam   = g["fam_label"]
        head  = f"{short} ({team}) · {fam}"

        # Primary chips
        primary = g["primary"]
        ln = g["primary_line"]
        o = primary.get("over", {})
        u = primary.get("under", {})
        o_chip = _chip(f"Over {ln:g} · q {pct(o.get('q'))} · Mkt {pct(o.get('p'))} · {o.get('odds', '--'):+d}" if o else "Over —")
        u_chip = _chip(f"Under {ln:g} · q {pct(u.get('q'))} · Mkt {pct(u.get('p'))} · {u.get('odds', '--'):+d}" if u else "Under —")

        # Selection controls
        st.markdown(
            f"""
            <div style="border:1px solid #eee;border-radius:14px;padding:14px 16px;background:#fff;margin-bottom:10px;">
                <div style="font-weight:600;margin-bottom:6px;">{head}</div>
                <div style="display:flex;gap:8px;flex-wrap:wrap;margin:6px 0;">{o_chip}{u_chip}</div>
            """, unsafe_allow_html=True
        )
        c1, c2, c3 = st.columns([1,1,2])
        if o:
            take_o = c1.checkbox("Took Over ✓", key=f"{key_prefix}_o_{idx}_{o['leg_id']}", value=(o["leg_id"] in ss["picks"]))
            if take_o: ss["picks"][o["leg_id"]] = dict(time=datetime.now().isoformat(timespec="seconds"),
                         leg_id=o["leg_id"], description=o["desc"], american_odds=o["odds"],
                         q_final=o["q"], p_market=o["p"], edge=float(o["q"]-o["p"]),
                         market_type=fam, team_abbr=team, player_name=g["player_name"], side="Over", line=ln)
            else: ss["picks"].pop(o["leg_id"], None)
        if u:
            take_u = c2.checkbox("Took Under ✓", key=f"{key_prefix}_u_{idx}_{u['leg_id']}", value=(u["leg_id"] in ss["picks"]))
            if take_u: ss["picks"][u["leg_id"]] = dict(time=datetime.now().isoformat(timespec="seconds"),
                         leg_id=u["leg_id"], description=u["desc"], american_odds=u["odds"],
                         q_final=u["q"], p_market=u["p"], edge=float(u["q"]-u["p"]),
                         market_type=fam, team_abbr=team, player_name=g["player_name"], side="Under", line=ln)
            else: ss["picks"].pop(u["leg_id"], None)

        # Alt lines dropdown
        with st.expander("Alt Lines", expanded=False):
            if not g["alts"]:
                st.caption("No alternates listed for this market.")
            else:
                # Picker
                alt_options = [ln for ln,_ in g["alts"]]
                sel = st.selectbox("Choose alt line", options=alt_options, key=f"{key_prefix}_altselect_{idx}")
                # Show chips for chosen alt
                alt_map = {ln: sides for ln, sides in g["alts"]}
                sides = alt_map.get(sel, {})
                ao = sides.get("over", {})
                au = sides.get("under", {})
                st.markdown(" — ".join([
                    f"**Over {sel:g}** · q {pct(ao.get('q'))} · Mkt {pct(ao.get('p'))} · {ao.get('odds','--'):+d}" if ao else "Over —",
                    f"**Under {sel:g}** · q {pct(au.get('q'))} · Mkt {pct(au.get('p'))} · {au.get('odds','--'):+d}" if au else "Under —",
                ]))
                colA, colB = st.columns(2)
                if ao:
                    pick = colA.checkbox("Pick Over (alt) ✓", key=f"{key_prefix}_alt_o_{idx}_{ao['leg_id']}", value=(ao["leg_id"] in ss["picks"]))
                    if pick: ss["picks"][ao["leg_id"]] = dict(time=datetime.now().isoformat(timespec="seconds"),
                                  leg_id=ao["leg_id"], description=ao["desc"], american_odds=ao["odds"],
                                  q_final=ao["q"], p_market=ao["p"], edge=float(ao["q"]-ao["p"]),
                                  market_type=fam, team_abbr=team, player_name=g["player_name"], side="Over", line=sel)
                    else: ss["picks"].pop(ao["leg_id"], None)
                if au:
                    pick = colB.checkbox("Pick Under (alt) ✓", key=f"{key_prefix}_alt_u_{idx}_{au['leg_id']}", value=(au["leg_id"] in ss["picks"]))
                    if pick: ss["picks"][au["leg_id"]] = dict(time=datetime.now().isoformat(timespec="seconds"),
                                  leg_id=au["leg_id"], description=au["desc"], american_odds=au["odds"],
                                  q_final=au["q"], p_market=au["p"], edge=float(au["q"]-au["p"]),
                                  market_type=fam, team_abbr=team, player_name=g["player_name"], side="Under", line=sel)
                    else: ss["picks"].pop(au["leg_id"], None)
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------ Filters & UI scaffold ----------------------
with st.expander("Filters", expanded=True):
    games = sorted(set(enh["matchup"].dropna().tolist())) if not enh.empty else []
    all_games = st.checkbox("Select all games", True, disabled=(len(games)==0))
    sel_games = st.multiselect("Games", games, default=games if all_games else [])
    teams = sorted(ABBR_SET)
    all_teams = st.checkbox("Select all teams", True)
    sel_teams = st.multiselect("Teams", teams, default=teams if all_teams else [])
    cats = ["Moneyline","Run Line","Pitcher Ks","Pitcher BB","Pitcher Outs","Pitcher Win"]
    sel_cats = st.multiselect("Categories", cats, default=cats)
    od_min, od_max = st.slider("American odds", -700, 700, (-700, 700))

f = enh.copy()
if not f.empty:
    if sel_games: f = f[f["matchup"].isin(sel_games)]
    if sel_teams: f = f[f["team_abbr"].isin(sel_teams)]
    if sel_cats:  f = f[f["category"].isin(sel_cats)]
    f = f[(f["american_odds"].fillna(0).astype(float) >= od_min) &
          (f["american_odds"].fillna(0).astype(float) <= od_max)]

tabs = st.tabs(["Props (Grouped)","ML/RL","My Picks","Downloads","Debug"])

with tabs[0]:
    st.subheader("Player Props — Grouped by Player/Market")
    render_grouped_cards(f, key_prefix="grp")

with tabs[1]:
    st.subheader("ML & Run Line")
    ml = f[f["category"].isin(["Moneyline","Run Line"])].copy() if not f.empty else pd.DataFrame()
    if ml.empty:
        st.info("No ML/RL rows available.")
    else:
        show = ml[["description","american_odds","q_final","p_market"]].rename(columns={
            "american_odds":"Odds","q_final":"q (model)","p_market":"Market %"
        })
        show["q (model)"] = show["q (model)"].map(lambda x: pct(x,1))
        show["Market %"]  = show["Market %"].map(lambda x: pct(x,1))
        st.dataframe(show, use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("My Picks")
    if not ss["picks"]:
        st.info("No picks selected yet.")
    else:
        picks_df = pd.DataFrame(list(ss["picks"].values())).sort_values("time", ascending=False)
        show = picks_df[["time","description","american_odds","q_final","p_market","edge"]].copy()
        show = show.rename(columns={"american_odds":"Odds","q_final":"q (model)","p_market":"Market %","edge":"Edge"})
        show["q (model)"] = show["q (model)"].map(lambda x: pct(x,1))
        show["Market %"]  = show["Market %"].map(lambda x: pct(x,1))
        show["Edge"]      = show["Edge"].map(lambda x: f"{x:+.1%}")
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.download_button(
            "Download My Picks CSV",
            picks_df.to_csv(index=False).encode("utf-8"),
            file_name=f"my_picks_{datetime.now().date().isoformat()}.csv",
            mime="text/csv"
        )

with tabs[3]:
    st.subheader("Downloads")
    def dl(df, label, fn):
        if df is None or df.empty: st.button(label, disabled=True)
        else:
            st.download_button(label, df.to_csv(index=False).encode("utf-8"),
                               file_name=fn, mime="text/csv")
    today = datetime.now().date().isoformat()
    dl(events_df, "Download Events CSV", f"events_{today}.csv")
    dl(board_df,  "Download Board CSV",  f"board_{today}.csv")
    dl(props_df,  "Download Props CSV",  f"props_{today}.csv")

with tabs[4]:
    dbg = dict(
        events_rows=len(events_df),
        board_rows=len(board_df),
        props_rows=len(props_df),
        pool_rows=len(pool_base),
        filtered_rows=len(f),
        has_api_key=bool(api_key()),
        markets=PROP_MARKETS,
        last_server_snapshot=last_server_snapshot_path(),
        last_snapshot_ts=ss.get("last_snapshot_ts"),
    )
    st.code(json.dumps(dbg, indent=2))