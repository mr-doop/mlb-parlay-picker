# app.py -- MLB Parlay Picker (clean UI • robust model • proper opponent/park)
# -----------------------------------------------------------------------------
# What’s new (Aug 2025):
# • Top20 ranks by q_final (prob to hit), then edge, then price
# • Opponent rates use true opponent team per row (no more 22%/8% everywhere)
# • Park/weather keyed to home park
# • Form (L5) disabled by default; weight slider if you want to use it
# • Smaller feature weights + calibration to market to keep q realistic
# • Alt lines shown in a compact collapsible; card shows one primary line
# • "Opportunities" tab: q ≥ threshold (default 60%) + reasonable odds
# • Safer dataframe guards to prevent KeyErrors

from __future__ import annotations
import os, re, json, math, glob, zipfile
from io import BytesIO
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------------------- App Config ---------------------------------
APP_TTL_MIN = 30
BOOK   = "draftkings"
REGION = "us"
SPORT  = "baseball_mlb"

# Odds API props (use /events/{id}/odds)
PROP_MARKETS = [
    "pitcher_strikeouts",
    "pitcher_strikeouts_alternate",
    "pitcher_walks",
    "pitcher_walks_alternate",
    "pitcher_outs",
    "pitcher_record_a_win",
]

SNAP_DIR = "snapshots"

# --------------------------- Small helpers -------------------------------
def _s(x): return "" if x is None else str(x)
def _cap3(s): return _s(s).upper()[:3]

def american_to_decimal(a):
    try:
        a = float(a)
        return 1 + (a/100.0) if a >= 0 else 1 + (100.0/abs(a))
    except Exception:
        return np.nan

def american_to_prob(a):
    try:
        a = float(a)
        if a >= 0: return 100.0/(a+100.0)
        return (-a)/((-a)+100.0)
    except Exception:
        return np.nan

def pct(x, d=1):
    try: return f"{float(x):.{d}%}"
    except Exception: return "--"

def first_init_last(full_name:str) -> str:
    s = _s(full_name).strip()
    if not s: return ""
    parts = s.split()
    if len(parts) == 1: return parts[0].title()
    return f"{parts[0][0].upper()}. {' '.join(p.title() for p in parts[1:])}"

def api_key():
    try: return st.secrets["ODDS_API_KEY"]
    except Exception: return ""

# --------------------------- Teams & Parks -------------------------------
TEAM_ABBR = {
    "Baltimore Orioles":"BAL","Boston Red Sox":"BOS","New York Yankees":"NYY","Tampa Bay Rays":"TBR","Toronto Blue Jays":"TOR",
    "Chicago White Sox":"CWS","Cleveland Guardians":"CLE","Detroit Tigers":"DET","Kansas City Royals":"KC","Minnesota Twins":"MIN",
    "Houston Astros":"HOU","Los Angeles Angels":"LAA","Oakland Athletics":"OAK","Seattle Mariners":"SEA","Texas Rangers":"TEX",
    "Atlanta Braves":"ATL","Miami Marlins":"MIA","New York Mets":"NYM","Philadelphia Phillies":"PHI","Washington Nationals":"WSH",
    "Chicago Cubs":"CHC","Cincinnati Reds":"CIN","Milwaukee Brewers":"MIL","Pittsburgh Pirates":"PIT","St. Louis Cardinals":"STL",
    "Arizona Diamondbacks":"ARI","Colorado Rockies":"COL","Los Angeles Dodgers":"LAD","San Diego Padres":"SDP","San Francisco Giants":"SFG",
}
ABBR_SET = set(TEAM_ABBR.values())

# Mild, varied park factors (k/BB). Used only if no features file overrides.
PARK = {
    "ARI":{"k":1.00,"bb":0.99},"ATL":{"k":0.99,"bb":0.98},"BAL":{"k":1.02,"bb":1.00},
    "BOS":{"k":0.98,"bb":1.02},"CHC":{"k":1.00,"bb":1.00},"CWS":{"k":1.00,"bb":0.99},
    "CIN":{"k":1.02,"bb":1.01},"CLE":{"k":1.00,"bb":1.00},"COL":{"k":0.94,"bb":1.04},
    "DET":{"k":1.01,"bb":1.00},"HOU":{"k":0.99,"bb":0.99},"KC":{"k":1.00,"bb":1.00},
    "LAA":{"k":0.99,"bb":0.99},"LAD":{"k":0.99,"bb":0.98},"MIA":{"k":1.02,"bb":1.02},
    "MIL":{"k":1.00,"bb":1.00},"MIN":{"k":1.01,"bb":1.00},"NYM":{"k":0.99,"bb":1.01},
    "NYY":{"k":0.98,"bb":0.99},"OAK":{"k":1.02,"bb":1.03},"PHI":{"k":0.99,"bb":0.99},
    "PIT":{"k":1.01,"bb":1.01},"SDP":{"k":1.01,"bb":1.00},"SFG":{"k":1.03,"bb":1.01},
    "SEA":{"k":1.01,"bb":0.99},"STL":{"k":1.00,"bb":1.00},"TBR":{"k":1.00,"bb":1.00},
    "TEX":{"k":0.99,"bb":0.99},"TOR":{"k":1.00,"bb":1.00},"WSH":{"k":1.00,"bb":1.01},
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
    "TEX": (32.7473, -97.0842),"TOR":(43.6414,-79.3894),"WSH":(38.8730,-77.0074),
}

# A varied static opponent K/BB table (used only if no uploaded "rates.csv")
STATIC_RATES = pd.DataFrame([
    # 2025-ish plausible ranges (NOT official; serves as varied fallback)
    ("ARI",0.215,0.081),("ATL",0.199,0.074),("BAL",0.221,0.079),("BOS",0.217,0.083),
    ("CHC",0.230,0.085),("CWS",0.241,0.082),("CIN",0.226,0.088),("CLE",0.198,0.086),
    ("COL",0.226,0.086),("DET",0.237,0.088),("HOU",0.197,0.078),("KC",0.208,0.082),
    ("LAA",0.224,0.079),("LAD",0.205,0.083),("MIA",0.236,0.081),("MIL",0.230,0.095),
    ("MIN",0.233,0.090),("NYM",0.217,0.086),("NYY",0.214,0.087),("OAK",0.239,0.093),
    ("PHI",0.213,0.081),("PIT",0.231,0.086),("SDP",0.222,0.082),("SFG",0.221,0.090),
    ("SEA",0.235,0.091),("STL",0.219,0.082),("TBR",0.226,0.085),("TEX",0.210,0.079),
    ("TOR",0.220,0.090),("WSH",0.217,0.087),
], columns=["team_abbr","opp_k_pct","opp_bb_pct"])

# --------------------------- Persistence --------------------------------
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
                    with z.open(name) as f: return pd.read_csv(f)
                except Exception: return pd.DataFrame()
            return _rd("events.csv"), _rd("board.csv"), _rd("props.csv"), _rd("rates.csv")
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def write_server_snapshot(bytes_data: bytes) -> str:
    os.makedirs(SNAP_DIR, exist_ok=True)
    fname = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    path  = os.path.join(SNAP_DIR, fname)
    with open(path, "wb") as f: f.write(bytes_data)
    return path

def last_server_snapshot_path() -> str|None:
    files = sorted(glob.glob(os.path.join(SNAP_DIR, "snapshot_*.zip")))
    return files[-1] if files else None

def autosave_snapshot(ss):
    try:
        b = save_snapshot_to_zip(ss.get("events_df"), ss.get("board_df"), ss.get("props_df"), ss.get("rates_df"))
        if b:
            ss["last_snapshot_bytes"] = b
            path = write_server_snapshot(b)
            ss["last_snapshot_path"] = path
            ss["last_snapshot_ts"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return path
    except Exception as e:
        ss["last_snapshot_error"] = str(e)
    return None

# --------------------------- HTTP --------------------------------------
def _get(url, params, timeout=15):
    r = requests.get(url, params=params, timeout=timeout)
    status = r.status_code
    try:
        r.raise_for_status()
        return status, r.json(), None
    except Exception as e:
        return status, None, str(e)

@st.cache_data(show_spinner=False, ttl=APP_TTL_MIN*60)
def fetch_events() -> pd.DataFrame:
    key = api_key()
    if not key: return pd.DataFrame()
    u = f"https://api.the-odds-api.com/v4/sports/{SPORT}/events"
    status, js, err = _get(u, dict(apiKey=key, dateFormat="iso"))
    if err or not isinstance(js, list): return pd.DataFrame()
    rows=[]
    for ev in js:
        home = TEAM_ABBR.get(ev.get("home_team"), _cap3(ev.get("home_team")))
        away = TEAM_ABBR.get(ev.get("away_team"), _cap3(ev.get("away_team")))
        rows.append(dict(
            event_id=ev.get("id"),
            start=ev.get("commence_time"),
            home_abbr=home, away_abbr=away,
            matchup=f"{away}@{home}"
        ))
    df = pd.DataFrame(rows).drop_duplicates("event_id")
    return df

@st.cache_data(show_spinner=False, ttl=APP_TTL_MIN*60)
def fetch_board() -> pd.DataFrame:
    key = api_key()
    if not key: return pd.DataFrame()
    u = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
    status, js, err = _get(u, dict(apiKey=key, regions=REGION, bookmakers=BOOK,
                                   oddsFormat="american", dateFormat="iso", markets="h2h,spreads"))
    if err or not isinstance(js, list): return pd.DataFrame()
    rows=[]
    for g in js:
        eid  = g.get("id")
        home = TEAM_ABBR.get(g.get("home_team"), _cap3(g.get("home_team")))
        away = TEAM_ABBR.get(g.get("away_team"), _cap3(g.get("away_team")))
        matchup = f"{away}@{home}"
        for bk in g.get("bookmakers", []):
            if bk.get("key") != BOOK: continue
            for m in bk.get("markets", []):
                mkey = m.get("key")
                if mkey not in ("h2h","spreads"): continue
                for o in m.get("outcomes", []):
                    price = o.get("price"); line = o.get("point"); name = o.get("name")
                    category = "Moneyline" if mkey=="h2h" else "Run Line"
                    team = TEAM_ABBR.get(name, _cap3(name))
                    rows.append(dict(
                        event_id=eid, market_type=category, team_abbr=team, player_name=None,
                        side=None, line=line, american_odds=price, decimal_odds=american_to_decimal(price),
                        p_market=american_to_prob(price), game_id=eid, home_abbr=home, away_abbr=away,
                        category=category, matchup=matchup
                    ))
    df = pd.DataFrame(rows)
    if df.empty: return df
    return df.drop_duplicates(subset=["event_id","market_type","team_abbr","line","american_odds"])

@st.cache_data(show_spinner=False, ttl=APP_TTL_MIN*60)
def fetch_props_by_events_cached(event_key: tuple, markets_key: tuple) -> pd.DataFrame:
    return _fetch_props_by_events_network(list(event_key), list(markets_key))

def _fetch_props_by_events_network(event_ids:list[str], markets:list[str]) -> pd.DataFrame:
    key = api_key()
    if not key or not event_ids: return pd.DataFrame()
    rows=[]
    for eid in event_ids:
        u = f"https://api.the-odds-api.com/v4/sports/{SPORT}/events/{eid}/odds"
        params = dict(apiKey=key, regions=REGION, bookmakers=BOOK,
                      oddsFormat="american", dateFormat="iso", markets=",".join(markets))
        status, js, err = _get(u, params)
        if err or not isinstance(js, dict): continue
        home = TEAM_ABBR.get(js.get("home_team"), _cap3(js.get("home_team")))
        away = TEAM_ABBR.get(js.get("away_team"), _cap3(js.get("away_team")))
        matchup = f"{away}@{home}"
        for bk in js.get("bookmakers", []):
            if bk.get("key") != BOOK: continue
            for m in bk.get("markets", []):
                mkey = m.get("key"); 
                if mkey not in markets: continue
                for o in m.get("outcomes", []):
                    price = o.get("price"); line = o.get("point")
                    side  = _s(o.get("name")).lower() if o.get("name") else None
                    # Try best-effort player name extraction
                    player = o.get("description") or o.get("participant") or o.get("player")
                    if isinstance(player, str) and (" Over" in player or " Under" in player):
                        player = re.split(r"\sOver|\sUnder", player)[0]
                    player = _s(player).strip()
                    team = TEAM_ABBR.get(o.get("team") or o.get("participant"), "")
                    rows.append(dict(
                        event_id=eid, market_type=mkey, team_abbr=team, player_name=player,
                        side=side, line=line, american_odds=price, decimal_odds=american_to_decimal(price),
                        p_market=american_to_prob(price), game_id=eid, home_abbr=home, away_abbr=away,
                        category=_pretty_category(mkey), matchup=matchup
                    ))
    return pd.DataFrame(rows)

# ---------------------- Rates, Weather, Small model ---------------------
@st.cache_data(show_spinner=False, ttl=900)
def get_weather_factor(park_abbr: str) -> float:
    ab = _cap3(park_abbr)
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
        return max(0.88, min(1.12, factor))
    except Exception:
        return 1.0

def get_opponent_rates_from_session() -> pd.DataFrame:
    # 1) uploaded / restored snapshot
    ss = st.session_state
    df = ss.get("rates_df")
    if isinstance(df, pd.DataFrame) and not df.empty:
        cols = [c.lower() for c in df.columns]
        if "team_abbr" not in cols:
            # allow 'team' column
            if "team" in df.columns: df = df.rename(columns={"team":"team_abbr"})
        return df[["team_abbr","opp_k_pct","opp_bb_pct"]].copy()
    # 2) static varied fallback
    return STATIC_RATES.copy()

def _poisson_cdf(k, lam):
    if lam is None or lam <= 0: return 0.0
    if lam > 30:
        z = (k + 0.5 - lam) / math.sqrt(lam)
        return 0.5*(1.0 + math.erf(z/math.sqrt(2)))
    p = math.exp(-lam); s = p
    for i in range(1, max(1,int(k))+1):
        p *= lam / i
        s += p
    return s

def _over_prob(line, lam):
    try:
        thr = math.floor(float(line) + 1e-9)
        return 1.0 - _poisson_cdf(thr - 1, lam)
    except Exception:
        return None

def _logit(p):
    p = max(1e-6, min(1-1e-6, float(p)))
    return math.log(p/(1-p))

def _inv(z): return 1.0/(1.0+math.exp(-float(z)))

def _clean_market_text(mkey:str) -> str:
    mkey = _s(mkey).lower()
    if "strikeout" in mkey: return "Ks"
    if "walk" in mkey and "pitcher" in mkey: return "BB"
    if "outs" in mkey and "pitcher" in mkey: return "Outs"
    if "record_a_win" in mkey: return "Win"
    return mkey

def _pretty_category(mkey:str) -> str:
    mkey = _s(mkey).lower()
    if "strikeout" in mkey: return "Pitcher Ks"
    if "walks" in mkey and "pitcher" in mkey: return "Pitcher BB"
    if "outs" in mkey and "pitcher" in mkey: return "Pitcher Outs"
    if "record_a_win" in mkey: return "Pitcher Win"
    return mkey

# ----------------------- Build pool (adds opponent/park) ----------------
def describe_row(row: pd.Series) -> str:
    mtext = _clean_market_text(row.get("market_type",""))
    side  = _s(row.get("side")).title() if row.get("side") else ""
    line  = row.get("line")
    matchup = row.get("matchup") or f"{_cap3(row.get('away_abbr'))}@{_cap3(row.get('home_abbr'))}"
    team = _cap3(row.get("team_abbr"))
    if row.get("player_name"):
        nm = first_init_last(row["player_name"])
        if pd.notna(line):
            bet = f"{nm} ({team or '--'}) {side} {line:g} {mtext}"
        else:
            bet = f"{nm} ({team or '--'}) {side} {mtext}"
    else:
        if row.get("market_type") == "Moneyline":
            bet = f"{team} ML ({matchup})"
        else:
            bet = f"{team} {row.get('line',0):+0.1f} ({matchup})" if pd.notna(row.get("line")) else f"{team} RL ({matchup})"
    return bet

def build_pool(board_df: pd.DataFrame, props_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["event_id","market_type","team_abbr","player_name","side","line",
            "american_odds","decimal_odds","p_market","home_abbr","away_abbr",
            "category","matchup","game_id"]
    def _norm(df):
        if isinstance(df, pd.DataFrame) and not df.empty:
            out = df.copy()
            # ensure required columns exist
            for c in ["game_id","category","matchup"]:
                if c not in out.columns:
                    if c == "game_id": out["game_id"] = out.get("event_id", "")
                    elif c == "category": out["category"] = out.get("market_type","")
                    elif c == "matchup":
                        out["matchup"] = out.apply(lambda r: f"{_cap3(r.get('away_abbr'))}@{_cap3(r.get('home_abbr'))}", axis=1)
            # subset/align
            return out[[c for c in cols if c in out.columns]]
        return pd.DataFrame(columns=cols)
    b = _norm(board_df)
    p = _norm(props_df)
    df = pd.concat([b, p], ignore_index=True, sort=False)
    if df.empty: return df

    # best-effort team on props; if missing, default to home team
    df["team_abbr"] = df["team_abbr"].where(df["team_abbr"].isin(ABBR_SET), df["home_abbr"])
    df["p_market"]  = df["p_market"].fillna(df["american_odds"].map(american_to_prob))

    # Derive opponent & park
    def _opp(row):
        team = _cap3(row.get("team_abbr"))
        home = _cap3(row.get("home_abbr"))
        away = _cap3(row.get("away_abbr"))
        # park = home field
        park = home
        if team == home: opp = away
        elif team == away: opp = home
        else: opp = away  # fallback
        return pd.Series(dict(opp_abbr=opp, park_abbr=park))
    extra = df.apply(_opp, axis=1)
    for c in ["opp_abbr","park_abbr"]: df[c] = extra[c]

    df["description"] = df.apply(describe_row, axis=1)

    # Unique leg
    df["leg_id"] = (df["event_id"].astype(str)+"|"+df["market_type"].astype(str)+"|"+
                    df["team_abbr"].astype(str)+"|"+df["player_name"].astype(str)+"|"+
                    df["side"].astype(str)+"|"+df["line"].astype(str)+"|"+df["american_odds"].astype(str))
    df = df.drop_duplicates(subset=["leg_id"]).reset_index(drop=True)
    return df

# --------------------------- Enhanced probabilities ---------------------
def enhanced_q(df: pd.DataFrame, form_weight: float = 0.0) -> pd.DataFrame:
    """
    Small, regularized model:
      • base p = average of market p and simple distributional p
      • additive adjustments in logit space with small weights
      • clamps to [0.05, 0.95]
    """
    if df is None or df.empty: return df.copy()

    # Baselines
    K9, BB9, IP0 = 8.4, 3.2, 5.8  # conservative league-ish
    opp_rates = get_opponent_rates_from_session().set_index("team_abbr")

    out = df.copy()
    q = []

    for _, r in out.iterrows():
        m   = _s(r.get("market_type")).lower()
        side= _s(r.get("side")).lower()
        ln  = r.get("line")
        p_m = r.get("p_market") if pd.notna(r.get("p_market")) else 0.5

        # simple distributional p_d
        if "strikeout" in m:
            lam = IP0 * (K9/9.0)
            p_d = _over_prob(ln, lam)
            p_d = p_d if side=="over" else (1-p_d) if p_d is not None else 0.5
        elif "walks" in m and "pitcher" in m:
            lam = IP0 * (BB9/9.0)
            p_d = _over_prob(ln, lam)
            p_d = p_d if side=="over" else (1-p_d) if p_d is not None else 0.5
        elif "outs" in m and "pitcher" in m:
            mu, sd = IP0*3.0, 3.2
            if pd.isna(ln):
                p_d = 0.5
            else:
                z0 = (float(ln) - 0.5 - mu)/sd
                p_over = 0.5*(1 - math.erf(z0/math.sqrt(2)))
                p_d = p_over if side=="over" else (1-p_over)
        elif "record_a_win" in m:
            # blend ML if available for team; otherwise 0.52 baseline for favorites, 0.48 otherwise
            p_d = 0.50
        else:
            p_d = 0.5

        p0 = 0.5*(p_m + (p_d if p_d is not None else 0.5))  # conservative base
        z  = _logit(p0)
        bits = []

        # Opponent (true opponent)
        opp = _cap3(r.get("opp_abbr"))
        ok  = float(opp_rates.loc[opp,"opp_k_pct"]) if opp in opp_rates.index else 0.22
        ob  = float(opp_rates.loc[opp,"opp_bb_pct"]) if opp in opp_rates.index else 0.08

        if "strikeout" in m:
            z += 0.30*((ok-0.22)/0.10); bits.append(f"oppK {ok:.0%}")
        if "walks" in m and "pitcher" in m:
            z += 0.25*((ob-0.08)/0.08); bits.append(f"oppBB {ob:.0%}")

        # Park & weather (home park)
        park = _cap3(r.get("park_abbr"))
        pf   = PARK.get(park, {"k":1.0,"bb":1.0})
        if "strikeout" in m:
            z += 0.08*((pf["k"]-1.0)/0.10); bits.append(f"ParkK×{pf['k']:.2f}")
        if "walks" in m and "pitcher" in m:
            z += 0.06*((pf["bb"]-1.0)/0.10); bits.append(f"ParkBB×{pf['bb']:.2f}")

        if ("strikeout" in m) or ("outs" in m and "pitcher" in m):
            wf = get_weather_factor(park)
            z += 0.05*((wf-1.0)*2.0); bits.append(f"Wind×{wf:.2f}")

        # Pitcher form (disabled by default; can be dialed in via sidebar)
        if form_weight > 0 and _s(r.get("player_name")):
            # until we have reliable L5 gamelogs, apply a tiny neutral bump only if alt lines suggest usage
            # we read typical line level as proxy (if provided)
            base_line = r.get("line")
            if pd.notna(base_line):
                z += float(form_weight) * 0.05 * ((base_line - (5.0 if "strikeout" in m else 15.0))/5.0)
                bits.append("Form* (proxy)")
        q.append(max(0.05, min(0.95, _inv(z))))

        out.at[_, "q_notes"] = " • ".join(bits)

    out["q_enh"] = q
    return out

def calibrate_vs_market(df: pd.DataFrame, lam: float=0.50) -> pd.DataFrame:
    """Blend our signal with market to keep q realistic."""
    if df is None or df.empty: 
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    out = df.copy()
    p = out["q_enh"].fillna(out["p_market"].fillna(0.5)).clip(1e-6, 1-1e-6).astype(float)
    m = out["p_market"].fillna(0.5).clip(1e-6, 1-1e-6).astype(float)
    z = (1-lam)*p.map(lambda x: math.log(x/(1-x))) + lam*m.map(lambda x: math.log(x/(1-x)))
    out["q_final"] = z.map(lambda x: 1.0/(1.0+math.exp(-x))).clip(0.05, 0.95)
    return out

def ensure_prob_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None: return pd.DataFrame()
    out = df.copy()
    out["p_market"] = out.get("p_market", np.nan)
    mask_na = out["p_market"].isna()
    out.loc[mask_na, "p_market"] = out.loc[mask_na, "american_odds"].map(american_to_prob)
    if "q_enh" not in out.columns:   out["q_enh"] = out["p_market"].fillna(0.5)
    if "q_final" not in out.columns: out["q_final"] = out["q_enh"].fillna(out["p_market"]).fillna(0.5)
    return out

# --------------------------- Streamlit Shell ----------------------------
st.set_page_config(page_title="MLB Parlay Picker -- Clean", page_icon="⚾", layout="wide")
st.markdown("<style>body{background:#fafafa}</style>", unsafe_allow_html=True)
st.title("MLB Parlay Picker -- MVP")

ss = st.session_state
for k, default in [
    ("events_df", None), ("board_df", None), ("props_df", None), ("rates_df", None),
    ("picks", {}), ("lock_data", False),
    ("last_snapshot_bytes", None), ("last_snapshot_path", None), ("last_snapshot_ts", None),
    ("auto_snapshot", True),
]:
    ss.setdefault(k, default)

# Sidebar
with st.sidebar:
    st.subheader("Data")
    st.caption("Manual fetch keeps credits under your control.")
    c1,c2 = st.columns(2)
    with c1:
        btn_events = st.button("Fetch events", disabled=ss["lock_data"])
        btn_board  = st.button("Fetch board",  disabled=ss["lock_data"])
    with c2:
        btn_props  = st.button("Fetch props",  disabled=ss["lock_data"])
        btn_rates  = st.button("Opp. rates",   disabled=ss["lock_data"])

    st.markdown("---")
    st.subheader("Model")
    form_w = st.slider("Form weight (tiny)", 0.0, 1.0, 0.0, 0.1, help="Leave 0 until we wire reliable L5 gamelogs.")
    cal_lam = st.slider("Calibrate toward market", 0.0, 1.0, 0.50, 0.05)

    st.markdown("---")
    st.subheader("Persistence")
    ss["auto_snapshot"] = st.toggle("Auto‑save snapshot after fetch", value=ss["auto_snapshot"])
    ss["lock_data"]     = st.toggle("🔒 Data Lock (prevent overwrite)", value=ss["lock_data"])
    s1,s2 = st.columns(2)
    with s1:
        save_snap = st.button("Save snapshot (server & download)")
    with s2:
        restore_snap = st.button("Restore last server snapshot")
    up_zip = st.file_uploader("Load snapshot (zip with events/board/props/rates)", type=["zip"])

    if ss.get("last_snapshot_bytes"):
        st.download_button(
            "Download latest auto‑snapshot",
            data=ss["last_snapshot_bytes"], file_name="mlb_snapshot_latest.zip",
            mime="application/zip", key="dl_latest_auto"
        )
        if ss.get("last_snapshot_ts"):
            st.caption(f"Auto‑saved: {ss['last_snapshot_ts']}")

def _auto_after(label:str):
    if ss["auto_snapshot"]:
        p = autosave_snapshot(ss)
        if p: st.sidebar.success(f"Auto‑saved after {label} → {os.path.basename(p)}")
        else: st.sidebar.warning(f"Auto‑save after {label} failed.")

if btn_events and not ss["lock_data"]:
    ss["events_df"] = fetch_events(); _auto_after("events")

if btn_board and not ss["lock_data"]:
    ss["board_df"] = fetch_board();   _auto_after("board")

if btn_props and not ss["lock_data"]:
    eids = []
    if isinstance(ss.get("events_df"), pd.DataFrame) and not ss["events_df"].empty:
        eids = ss["events_df"]["event_id"].dropna().astype(str).tolist()
    ek = tuple(sorted(set(eids)))
    mk = tuple(sorted(set(PROP_MARKETS)))
    ss["props_df"] = fetch_props_by_events_cached(ek, mk)
    _auto_after("props")

if btn_rates and not ss["lock_data"]:
    ss["rates_df"] = STATIC_RATES.copy()  # you can also upload a precise rates.csv
    _auto_after("opponent rates")

if save_snap:
    b = save_snapshot_to_zip(ss.get("events_df"), ss.get("board_df"), ss.get("props_df"), ss.get("rates_df"))
    ss["last_snapshot_bytes"] = b
    path = write_server_snapshot(b)
    ss["last_snapshot_path"] = path
    ss["last_snapshot_ts"]   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    st.sidebar.success(f"Snapshot saved: {path}")
    st.sidebar.download_button("Download snapshot zip", data=b,
        file_name=f"mlb_snapshot_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
        mime="application/zip", key="dl_manual_snap")

if restore_snap:
    p = last_server_snapshot_path()
    if p and os.path.exists(p):
        with open(p, "rb") as f:
            E,B,P,R = load_snapshot_from_zip(f)
        if not ss["lock_data"]:
            ss["events_df"], ss["board_df"], ss["props_df"], ss["rates_df"] = E,B,P,R
        st.sidebar.success(f"Restored snapshot: {p}")
    else:
        st.sidebar.warning("No server snapshot found.")

if up_zip is not None:
    E,B,P,R = load_snapshot_from_zip(up_zip)
    if not ss["lock_data"]:
        ss["events_df"], ss["board_df"], ss["props_df"], ss["rates_df"] = E,B,P,R
    st.sidebar.success("Loaded uploaded snapshot.")

# --------------------------- Build + Model ------------------------------
events_df = ss.get("events_df"); board_df = ss.get("board_df"); props_df = ss.get("props_df")
pool_base = build_pool(board_df, props_df)

enh_raw = enhanced_q(pool_base, form_weight=form_w) if isinstance(pool_base, pd.DataFrame) and not pool_base.empty else pool_base
enh_cal = calibrate_vs_market(enh_raw, lam=cal_lam)
enh     = ensure_prob_columns(enh_cal)
if isinstance(enh, pd.DataFrame) and not enh.empty:
    enh["edge"] = enh["q_final"].fillna(0.5) - enh["p_market"].fillna(0.5)

# --------------------------- Filters -----------------------------------
with st.expander("Filters", expanded=True):
    games = []
    if isinstance(enh, pd.DataFrame) and "matchup" in enh.columns and not enh.empty:
        games = sorted(pd.Series(enh["matchup"].dropna().unique()).tolist())
    all_games = st.checkbox("Select all games", True, disabled=(len(games)==0))
    sel_games = st.multiselect("Games", games, default=games if all_games else [])
    teams = sorted(ABBR_SET)
    all_teams = st.checkbox("Select all teams", True)
    sel_teams = st.multiselect("Teams", teams, default=teams if all_teams else [])
    cats = ["Moneyline","Run Line","Pitcher Ks","Pitcher BB","Pitcher Outs","Pitcher Win"]
    sel_cats = st.multiselect("Categories", cats, default=cats)
    od_min, od_max = st.slider("American odds", -700, 700, (-700, 700))

f = enh.copy() if isinstance(enh, pd.DataFrame) else pd.DataFrame()
if not f.empty:
    if sel_games: f = f[f["matchup"].isin(sel_games)]
    if sel_teams: f = f[f["team_abbr"].isin(sel_teams)]
    if sel_cats:  f = f[f["category"].isin(sel_cats)]
    # robust odds filter
    aa = pd.to_numeric(f["american_odds"], errors="coerce")
    f = f[(aa >= od_min) & (aa <= od_max)]

# --------------------------- Tabs --------------------------------------
tabs = st.tabs(["Candidates","Top 20","Opportunities","Best Value (60%+)","Downloads","Debug"])

with tabs[0]:
    if f.empty:
        st.info("No rows. Fetch/upload and/or relax filters.")
    else:
        show = f[["description","category","american_odds","q_final","p_market","edge"]].copy()
        show = show.rename(columns={"american_odds":"Odds","q_final":"q (model)","p_market":"Market %","edge":"Edge"})
        show["q (model)"] = show["q (model)"].map(lambda x: pct(x,1))
        show["Market %"] = show["Market %"].map(lambda x: pct(x,1))
        show["Edge"]     = show["Edge"].map(lambda x: f"{x:+.1%}")
        st.dataframe(show, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Top 20 (highest probability)")
    if f.empty:
        st.info("No rows to rank.")
    else:
        top = f.sort_values(["q_final","edge","american_odds"], ascending=[False,False,True]).head(20)
        for _, r in top.iterrows():
            chips = []
            if "alternate" in _s(r["market_type"]).lower(): chips.append("Alt")
            chips.extend([
                f"Odds {int(r['american_odds']) if pd.notna(r['american_odds']) else '--'}",
                f"q {pct(r['q_final'])}",
                f"Market {pct(r['p_market'])}",
                f"Edge {r['edge']:+.1%}",
                f"Trend +0.0pp",
            ])
            st.checkbox("Took ✓", key=f"take_{r['leg_id']}", value=(r["leg_id"] in ss["picks"]))
            if st.session_state.get(f"take_{r['leg_id']}"):
                ss["picks"][r["leg_id"]] = dict(
                    time=datetime.now().isoformat(timespec="seconds"),
                    leg_id=r["leg_id"], description=r["description"],
                    american_odds=int(r["american_odds"]) if pd.notna(r["american_odds"]) else None,
                    q_final=float(r["q_final"]), p_market=float(r["p_market"]),
                    edge=float(r["edge"]), market_type=r["market_type"],
                    team_abbr=r["team_abbr"], player_name=r.get("player_name"),
                    side=r.get("side"), line=r.get("line"), matchup=r.get("matchup"),
                )
            else:
                ss["picks"].pop(r["leg_id"], None)

            st.markdown(
                f"""
                <div style="border:1px solid #eee;border-radius:14px;padding:14px 16px;background:#fff;margin:6px 0;">
                    <div style="font-weight:600;margin-bottom:6px;">{r['description']}</div>
                    <div style="display:flex;gap:8px;flex-wrap:wrap;margin:6px 0;">
                        {''.join([f'<span style="background:#f5f6f7;border:1px solid #eee;border-radius:999px;padding:3px 8px;font-size:12px;color:#111;">{c}</span>' for c in chips])}
                    </div>
                    <div style="color:#333;font-size:13px;line-height:1.35;margin-bottom:4px">{_s(r.get('q_notes',''))}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            # Alt lines (compact)
            alts = f[
                (f["player_name"]==r.get("player_name")) &
                (f["market_type"].str.contains(_s(r.get("market_type")).split("_")[0], case=False, na=False)) &
                (f["event_id"]==r.get("event_id"))
            ][["side","line","american_odds","q_final","p_market","edge"]].copy()
            if not alts.empty:
                with st.expander("Alt lines", expanded=False):
                    alts = alts.rename(columns={"american_odds":"Odds","q_final":"q","p_market":"Market","edge":"Δedge"})
                    alts["q"]      = alts["q"].map(lambda x: pct(x,1))
                    alts["Market"] = alts["Market"].map(lambda x: pct(x,1))
                    alts["Δedge"]  = alts["Δedge"].map(lambda x: f"{x:+.1%}")
                    st.dataframe(alts, use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Opportunities (q ≥ 60% by default)")
    thr = st.slider("q threshold", 0.50, 0.80, 0.60, 0.01)
    if f.empty:
        st.info("No rows.")
    else:
        cand = f[(f["q_final"]>=thr)]
        # reasonable odds bands
        aa = pd.to_numeric(cand["american_odds"], errors="coerce")
        cand = cand[((cand["category"].str.contains("Win", case=False)) & (aa.between(-200, 180))) |
                    ((~cand["category"].str.contains("Win", case=False)) & (aa.between(-250, 220)))]
        cand = cand.sort_values(["q_final","edge","american_odds"], ascending=[False,False,True]).head(50)
        show = cand[["description","american_odds","q_final","p_market","edge"]].rename(columns={
            "american_odds":"Odds","q_final":"q","p_market":"Market","edge":"Edge"
        })
        show["q"]      = show["q"].map(lambda x: pct(x,1))
        show["Market"] = show["Market"].map(lambda x: pct(x,1))
        show["Edge"]   = show["Edge"].map(lambda x: f"{x:+.1%}")
        st.dataframe(show, use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Best Value (≥60%)")
    if f.empty: st.info("No rows.")
    else:
        g = f[(f["q_final"]>=0.60)].copy()
        g["value_score"] = (g["q_final"] - g["p_market"]) + 0.10*np.clip(g["american_odds"], -300, 300)/300.0
        g = g.sort_values(["value_score","q_final"], ascending=[False,False]).head(50)
        show = g[["description","american_odds","q_final","p_market","edge"]].rename(columns={
            "american_odds":"Odds","q_final":"q","p_market":"Market","edge":"Edge"
        })
        show["q"]      = show["q"].map(lambda x: pct(x,1))
        show["Market"] = show["Market"].map(lambda x: pct(x,1))
        show["Edge"]   = show["Edge"].map(lambda x: f"{x:+.1%}")
        st.dataframe(show, use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Downloads")
    def dl(df, label, fn):
        if not isinstance(df, pd.DataFrame) or df.empty: st.button(label, disabled=True)
        else: st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=fn, mime="text/csv")
    today = datetime.now().date().isoformat()
    dl(events_df, "Download Events CSV", f"events_{today}.csv")
    dl(board_df,  "Download Board CSV",  f"board_{today}.csv")
    dl(props_df,  "Download Props CSV",  f"props_{today}.csv")
    dl(enh,       "Download Candidates CSV", f"candidates_{today}.csv")

with tabs[5]:
    dbg = dict(
        events_rows = int(len(events_df)) if isinstance(events_df,pd.DataFrame) else 0,
        board_rows  = int(len(board_df))  if isinstance(board_df,pd.DataFrame) else 0,
        props_rows  = int(len(props_df))  if isinstance(props_df,pd.DataFrame) else 0,
        pool_rows   = int(len(pool_base)) if isinstance(pool_base,pd.DataFrame) else 0,
        filtered_rows = int(len(f))       if isinstance(f,pd.DataFrame) else 0,
        has_api_key = bool(api_key()),
        markets     = PROP_MARKETS,
        auto_snapshot = ss["auto_snapshot"],
        last_snapshot = last_server_snapshot_path(),
        opponent_rates_source = "uploaded/snapshot" if isinstance(ss.get("rates_df"), pd.DataFrame) and not ss["rates_df"].empty else "static_fallback"
    )
    st.code(json.dumps(dbg, indent=2))
    