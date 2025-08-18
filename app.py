# app.py -- MLB Parlay Picker (Auto‑Snapshot + Persistence + Alternates + Picks)
# -----------------------------------------------------------------------------
# Features:
# - Manual fetch (you control credits)
# - Auto-save a snapshot after each successful fetch (toggleable, ON by default)
# - Odds API player props via /events/{id}/odds (DraftKings)
# - Alternate pitcher Ks & BB included
# - Heuristic model + calibration; crash-safe column guards
# - Persistence: Save/Load snapshot (download/upload), Restore last server snapshot
# - Top-20 cards with "Took ✓" and My Picks tab with CSV export
# - Clean, white, minimal UI

import os, re, json, math, glob, zipfile
from io import BytesIO
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------------------- Config ---------------------------------
APP_TTL_MIN = 30
BOOK = "draftkings"
REGION = "us"
SPORT = "baseball_mlb"

# Include alternates for pitcher Ks & BB
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
        a = int(a)
        return (1 + a/100) if a > 0 else (1 + 100/(-a))
    except: return None
def american_to_prob(a):
    try:
        a = int(a)
        if a >= 0: return 100.0/(a+100.0)
        return (-a)/((-a)+100.0)
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

PARK = {  # coarse defaults
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
    "TEX": (32.7473, -97.0842),"TOR":(43.6414,-79.3894),"WSH":(38.8730,-77.0074),
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

@st.cache_data(show_spinner=False, ttl=900)
def get_weather_factor(team_abbr: str) -> float:
    ab = _cap3(team_abbr)
    latlon = BALLPARK_COORDS.get(ab)
    if not latlon: return 1.0
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params=dict(latitude=latlon[0], longitude=latlon[1], hourly="wind_speed_10m,wind_direction_10m", forecast_days=1),
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
                rows.append(dict(team_abbr=abbr, opp_k_pct=float(r["opp_k_pct"]), opp_bb_pct=float(r["opp_bb_pct"])))
        df = pd.DataFrame(rows)
        if df.empty: raise RuntimeError("fallback")
        return df
    except Exception:
        return pd.DataFrame([dict(team_abbr=ab, opp_k_pct=0.22, opp_bb_pct=0.08) for ab in ABBR_SET])

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

# Helper to auto-save snapshot and update session state
def autosave_snapshot(ss) -> str|None:
    try:
        bytes_zip = save_snapshot_to_zip(ss.get("events_df"), ss.get("board_df"), ss.get("props_df"), ss.get("rates_df"))
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
def _stable_events_arg(event_ids):
    if not event_ids: return tuple()
    return tuple(sorted(set(_s(e) for e in event_ids)))

def _stable_markets_arg(markets):
    if not markets: return tuple()
    return tuple(sorted(set(_s(m) for m in markets)))

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
def fetch_props_by_events_cached(event_key: tuple, markets_key: tuple) -> pd.DataFrame:
    """Cached layer keyed only by stable args. Calls the actual network function."""
    return _fetch_props_by_events_network(list(event_key), list(markets_key))

def _fetch_props_by_events_network(event_ids:list[str], markets:list[str]) -> pd.DataFrame:
    key = api_key()
    if not key or not event_ids: return pd.DataFrame()
    rows = []
    for eid in event_ids:
        u = f"https://api.the-odds-api.com/v4/sports/{SPORT}/events/{eid}/odds"
        params = dict(apiKey=key, regions=REGION, bookmakers=BOOK, oddsFormat="american", dateFormat="iso", markets=",".join(markets))
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
                    price = o.get("price")
                    line  = o.get("point")
                    side  = (_s(o.get("name")).lower() if o.get("name") else None)
                    # Try best-effort player name extraction:
                    player = o.get("description") or o.get("participant") or o.get("player")
                    if isinstance(player, str) and (" Over" in player or " Under" in player):
                        player = re.split(r"\sOver|\sUnder", player)[0]
                    player = _s(player).strip()
                    team = TEAM_ABBR.get(o.get("team") or o.get("participant"), "")
                    rows.append(dict(
                        event_id=eid, market_type=mkey, team_abbr=team, player_name=player,
                        side=side, line=line, american_odds=price, decimal_odds=american_to_decimal(price),
                        p_market=american_to_prob(price), game_id=eid, home_abbr=home, away_abbr=away,
                        description="", category=_pretty_category(mkey), matchup=matchup
                    ))
    return pd.DataFrame(rows)

def _pretty_category(mkey:str) -> str:
    mkey = _s(mkey)
    if "strikeout" in mkey: return "Pitcher Ks"
    if "walks" in mkey and "pitcher" in mkey: return "Pitcher BB"
    if "outs" in mkey and "pitcher" in mkey: return "Pitcher Outs"
    if "record_a_win" in mkey: return "Pitcher Win"
    return mkey

# ---------------------- Modeling (heuristic) --------------------------
def _poisson_cdf(k, lam):
    if lam is None or lam <= 0: return 0.0
    if lam > 30:
        z = (k + 0.5 - lam) / math.sqrt(lam)
        return 0.5*(1.0 + math.erf(z/math.sqrt(2)))
    from math import exp
    p, s = exp(-lam), exp(-lam)
    for i in range(1, int(k)+1):
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
    p = max(1e-6, min(1-1e-6, float(p))); return math.log(p/(1-p))
def _inv(z): return 1.0/(1.0+math.exp(-float(z)))

@st.cache_data(show_spinner=False, ttl=6*3600)
def pitcher_last5(name: str) -> dict:
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
        last_game = pd.to_datetime(g["game_date"].iloc[0]).date()
        rest = (datetime.now().date() - last_game).days
        return {"ip_l5":float(ip), "pc_l5":float(pc), "k9_l5":float(k9), "bb9_l5":float(bb9), "rest":int(rest)}
    except Exception:
        return {}

def enhanced_q(df: pd.DataFrame, use_weather=True, use_park=True, use_form=True) -> pd.DataFrame:
    if df is None or df.empty: return df.copy()
    K9, BB9, IP0 = 8.5, 3.3, 5.8
    opp_rates = get_opponent_rates().set_index("team_abbr")

    q_list, why = [], []
    for _, r in df.iterrows():
        m = _s(r.get("market_type"))
        side = _s(r.get("side")).lower()
        line = r.get("line")
        team = _cap3(r.get("team_abbr") or r.get("home_abbr"))
        p_mkt = r.get("p_market") or 0.5

        if "strikeout" in m:
            lam = IP0 * (K9/9.0); p0 = _over_prob(line, lam); p0 = p0 if side=="over" else (1-p0)
        elif "walks" in m and "pitcher" in m:
            lam = IP0 * (BB9/9.0); p0 = _over_prob(line, lam); p0 = p0 if side=="over" else (1-p0)
        elif "outs" in m and "pitcher" in m:
            mu, sd = IP0*3.0, 3.0
            if line is None: p0 = p_mkt
            else:
                z0 = (float(line) - 0.5 - mu)/sd
                p_over = 0.5*(1 - math.erf(z0/math.sqrt(2)))
                p0 = p_over if side=="over" else (1-p_over)
        elif "record_a_win" in m:
            p0 = 0.55 if side in ("yes","over") else 0.45
        else:
            p0 = p_mkt
        z = _logit(p0); bits = []

        ok = float(opp_rates.loc[team,"opp_k_pct"]) if team in opp_rates.index else 0.22
        ob = float(opp_rates.loc[team,"opp_bb_pct"]) if team in opp_rates.index else 0.08
        if "strikeout" in m:
            z += 0.60*((ok-0.22)/0.10); bits.append(f"oppK {ok:.0%}")
        if "walks" in m and "pitcher" in m:
            z += 0.60*((ob-0.08)/0.10); bits.append(f"oppBB {ob:.0%}")

        if use_park:
            pf = PARK.get(team, {"k":1.0,"bb":1.0})
            if "strikeout" in m:
                z += 0.30*((pf["k"]-1.0)/0.10); bits.append(f"ParkK×{pf['k']:.2f}")
            if "walks" in m and "pitcher" in m:
                z += 0.20*((pf["bb"]-1.0)/0.10); bits.append(f"ParkBB×{pf['bb']:.2f}")

        if use_weather and (("strikeout" in m) or ("outs" in m and "pitcher" in m)):
            wf = get_weather_factor(team)
            z += 0.15*((wf-1.0)*2.0); bits.append(f"Wind×{wf:.2f}")

        if use_form and _s(r.get("player_name")):
            f = pitcher_last5(r["player_name"])
            if f:
                if "strikeout" in m and f.get("k9_l5"): z += 0.35*((f["k9_l5"]-K9)/2.0); bits.append(f"K9L5 {f['k9_l5']:.1f}")
                if "walks" in m and "pitcher" in m and f.get("bb9_l5"): z += -0.35*((f["bb9_l5"]-BB9)/2.0); bits.append(f"BB9L5 {f['bb9_l5']:.1f}")
                if "outs" in m and "pitcher" in m and f.get("ip_l5"): z += 0.25*((f["ip_l5"]-IP0)); bits.append(f"IPL5 {f['ip_l5']:.1f}")

        q_list.append(1.0/(1.0+math.exp(-z)))
        why.append(" • ".join(bits))
    out = df.copy()
    out["q_enh"] = q_list
    out["q_notes"] = why
    return out

def calibrate_vs_market(df: pd.DataFrame, lam: float=0.25) -> pd.DataFrame:
    if df is None: return pd.DataFrame()
    if df.empty:
        df = df.copy()
        if "p_market" not in df.columns: df["p_market"] = []
        if "q_enh" not in df.columns: df["q_enh"] = []
        df["q_final"] = df.get("q_enh", df.get("p_market", 0.5))
        return df
    p = df["q_enh"].fillna(df["p_market"].fillna(0.5)).clip(1e-6, 1-1e-6).astype(float)
    m = df["p_market"].fillna(0.5).clip(1e-6, 1-1e-6).astype(float)
    z = (1-lam)*p.map(lambda x: math.log(x/(1-x))) + lam*m.map(lambda x: math.log(x/(1-x)))
    df = df.copy()
    df["q_final"] = z.map(lambda x: 1.0/(1.0+math.exp(-x)))
    return df

def ensure_prob_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None: return pd.DataFrame()
    df = df.copy()
    if "p_market" not in df.columns:
        df["p_market"] = df["american_odds"].map(american_to_prob)
    else:
        df["p_market"] = df["p_market"].fillna(df["american_odds"].map(american_to_prob))
    if "q_enh" not in df.columns:
        df["q_enh"] = df["p_market"].fillna(0.5)
    if "q_final" not in df.columns:
        df["q_final"] = df["q_enh"].fillna(df["p_market"]).fillna(0.5)
    return df

# ----------------------- Pool & Formatting ---------------------------
def _clean_market_text(mkey:str) -> str:
    mkey = _s(mkey)
    if "strikeout" in mkey: return "Ks"
    if "walk" in mkey and "pitcher" in mkey: return "BB"
    if "outs" in mkey and "pitcher" in mkey: return "Outs"
    if "record_a_win" in mkey: return "Win"
    return mkey

def _pretty_category(mkey:str) -> str:
    mkey = _s(mkey)
    if "strikeout" in mkey: return "Pitcher Ks"
    if "walks" in mkey and "pitcher" in mkey: return "Pitcher BB"
    if "outs" in mkey and "pitcher" in mkey: return "Pitcher Outs"
    if "record_a_win" in mkey: return "Pitcher Win"
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
    cols = ["event_id","market_type","team_abbr","player_name","side","line","american_odds","decimal_odds","p_market","home_abbr","away_abbr","category","matchup"]
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
    df["leg_id"] = (df["event_id"].astype(str)+"|"+df["market_type"].astype(str)+"|"+
                    df["team_abbr"].astype(str)+"|"+df["player_name"].astype(str)+"|"+
                    df["side"].astype(str)+"|"+df["line"].astype(str))
    df = df.drop_duplicates(subset=["leg_id"])
    return df

# --------------------------- App shell --------------------------------
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

# Sidebar -- Data fetch & persistence
with st.sidebar:
    st.subheader("Data")
    st.caption("Manual fetch keeps credits under your control.")
    colA, colB = st.columns(2)
    with colA:
        btn_events = st.button("Fetch events", disabled=ss["lock_data"])
        btn_board  = st.button("Fetch board",  disabled=ss["lock_data"])
    with colB:
        btn_props  = st.button("Fetch props",  disabled=ss["lock_data"])
        btn_rates  = st.button("Opp. rates",   disabled=ss["lock_data"])

    st.markdown("---")
    st.subheader("Persistence")
    ss["auto_snapshot"] = st.toggle("Auto‑save snapshot after fetch", value=ss["auto_snapshot"])
    ss["lock_data"] = st.toggle("🔒 Data Lock (prevent overwrite)", value=ss["lock_data"])

    colS1, colS2 = st.columns(2)
    with colS1:
        save_snap = st.button("Save snapshot (server & download)")
    with colS2:
        restore_snap = st.button("Restore last server snapshot")
    up_zip = st.file_uploader("Load snapshot (zip with events/board/props/rates)", type=["zip"])

    # Show latest auto-snapshot download if available
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
        bytes_zip = save_snapshot_to_zip(ss.get("events_df"), ss.get("board_df"), ss.get("props_df"), ss.get("rates_df"))
        ss["last_snapshot_bytes"] = bytes_zip
        path = write_server_snapshot(bytes_zip)
        ss["last_snapshot_path"] = path
        ss["last_snapshot_ts"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.success(f"Snapshot saved: {path}")
        st.download_button("Download snapshot zip", data=bytes_zip, file_name=f"mlb_snapshot_{datetime.now().strftime('%Y%m%d_%H%M')}.zip", mime="application/zip", key="dl_manual_snap")

    if restore_snap:
        p = last_server_snapshot_path()
        if p and os.path.exists(p):
            with open(p, "rb") as f:
                E,B,P,R = load_snapshot_from_zip(f)
            if not ss["lock_data"]:
                ss["events_df"], ss["board_df"], ss["props_df"], ss["rates_df"] = E,B,P,R
            st.success(f"Restored snapshot: {p}")
        else:
            st.warning("No server snapshot found.")

    if up_zip is not None:
        E,B,P,R = load_snapshot_from_zip(up_zip)
        if not ss["lock_data"]:
            ss["events_df"], ss["board_df"], ss["props_df"], ss["rates_df"] = E,B,P,R
        st.success("Loaded uploaded snapshot.")

# Handle fetch buttons (only if not locked) + AUTO‑SAVE after each fetch
def _auto_after_fetch(label: str):
    if ss["auto_snapshot"]:
        path = autosave_snapshot(ss)
        if path:
            st.sidebar.success(f"Auto‑saved after {label} → {os.path.basename(path)}")
        else:
            st.sidebar.warning(f"Auto‑save after {label} failed.")

if btn_events and not ss["lock_data"]:
    ss["events_df"] = fetch_events()
    _auto_after_fetch("events")

if btn_board and not ss["lock_data"]:
    ss["board_df"] = fetch_board()
    _auto_after_fetch("board")

if btn_props and not ss["lock_data"]:
    eids = ss["events_df"]["event_id"].tolist() if isinstance(ss.get("events_df"), pd.DataFrame) and not ss["events_df"].empty else []
    ek = tuple(sorted(set(_s(e) for e in eids)))
    mk = tuple(sorted(set(_s(m) for m in PROP_MARKETS)))
    ss["props_df"] = fetch_props_by_events_cached(ek, mk)  # cache avoids re‑hits if same args+TTL
    _auto_after_fetch("props")

if btn_rates and not ss["lock_data"]:
    ss["rates_df"] = get_opponent_rates()
    _auto_after_fetch("opponent rates")

# Current dataframes
events_df = ss["events_df"] if isinstance(ss["events_df"], pd.DataFrame) else pd.DataFrame()
board_df  = ss["board_df"]  if isinstance(ss["board_df"], pd.DataFrame)  else pd.DataFrame()
props_df  = ss["props_df"]  if isinstance(ss["props_df"], pd.DataFrame)  else pd.DataFrame()
rates_df  = ss["rates_df"]  if isinstance(ss["rates_df"], pd.DataFrame)  else get_opponent_rates()

# Build pool
def _clean_market_text(mkey:str) -> str:
    mkey = _s(mkey)
    if "strikeout" in mkey: return "Ks"
    if "walk" in mkey and "pitcher" in mkey: return "BB"
    if "outs" in mkey and "pitcher" in mkey: return "Outs"
    if "record_a_win" in mkey: return "Win"
    return mkey

def _pretty_category(mkey:str) -> str:
    mkey = _s(mkey)
    if "strikeout" in mkey: return "Pitcher Ks"
    if "walks" in mkey and "pitcher" in mkey: return "Pitcher BB"
    if "outs" in mkey and "pitcher" in mkey: return "Pitcher Outs"
    if "record_a_win" in mkey: return "Pitcher Win"
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
    cols = ["event_id","market_type","team_abbr","player_name","side","line","american_odds","decimal_odds","p_market","home_abbr","away_abbr","category","matchup"]
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
    df["leg_id"] = (df["event_id"].astype(str)+"|"+df["market_type"].astype(str)+"|"+
                    df["team_abbr"].astype(str)+"|"+df["player_name"].astype(str)+"|"+
                    df["side"].astype(str)+"|"+df["line"].astype(str))
    df = df.drop_duplicates(subset=["leg_id"])
    return df

pool_base = build_pool(board_df, props_df)

# Heuristic → calibration (crash‑safe)
def enhanced_q(df: pd.DataFrame, use_weather=True, use_park=True, use_form=True) -> pd.DataFrame:
    if df is None or df.empty: return df.copy()
    K9, BB9, IP0 = 8.5, 3.3, 5.8
    opp_rates = get_opponent_rates().set_index("team_abbr")

    q_list, why = [], []
    for _, r in df.iterrows():
        m = _s(r.get("market_type"))
        side = _s(r.get("side")).lower()
        line = r.get("line")
        team = _cap3(r.get("team_abbr") or r.get("home_abbr"))
        p_mkt = r.get("p_market") or 0.5

        if "strikeout" in m:
            lam = IP0 * (K9/9.0); p0 = _over_prob(line, lam); p0 = p0 if side=="over" else (1-p0)
        elif "walks" in m and "pitcher" in m:
            lam = IP0 * (BB9/9.0); p0 = _over_prob(line, lam); p0 = p0 if side=="over" else (1-p0)
        elif "outs" in m and "pitcher" in m:
            mu, sd = IP0*3.0, 3.0
            if line is None: p0 = p_mkt
            else:
                z0 = (float(line) - 0.5 - mu)/sd
                p_over = 0.5*(1 - math.erf(z0/math.sqrt(2)))
                p0 = p_over if side=="over" else (1-p_over)
        elif "record_a_win" in m:
            p0 = 0.55 if side in ("yes","over") else 0.45
        else:
            p0 = p_mkt
        z = _logit(p0); bits = []

        ok = float(opp_rates.loc[team,"opp_k_pct"]) if team in opp_rates.index else 0.22
        ob = float(opp_rates.loc[team,"opp_bb_pct"]) if team in opp_rates.index else 0.08
        if "strikeout" in m:
            z += 0.60*((ok-0.22)/0.10); bits.append(f"oppK {ok:.0%}")
        if "walks" in m and "pitcher" in m:
            z += 0.60*((ob-0.08)/0.10); bits.append(f"oppBB {ob:.0%}")

        if use_park:
            pf = PARK.get(team, {"k":1.0,"bb":1.0})
            if "strikeout" in m:
                z += 0.30*((pf["k"]-1.0)/0.10); bits.append(f"ParkK×{pf['k']:.2f}")
            if "walks" in m and "pitcher" in m:
                z += 0.20*((pf["bb"]-1.0)/0.10); bits.append(f"ParkBB×{pf['bb']:.2f}")

        if use_weather and (("strikeout" in m) or ("outs" in m and "pitcher" in m)):
            wf = get_weather_factor(team)
            z += 0.15*((wf-1.0)*2.0); bits.append(f"Wind×{wf:.2f}")

        if use_form and _s(r.get("player_name")):
            f = pitcher_last5(r["player_name"])
            if f:
                if "strikeout" in m and f.get("k9_l5"): z += 0.35*((f["k9_l5"]-K9)/2.0); bits.append(f"K9L5 {f['k9_l5']:.1f}")
                if "walks" in m and "pitcher" in m and f.get("bb9_l5"): z += -0.35*((f["bb9_l5"]-BB9)/2.0); bits.append(f"BB9L5 {f['bb9_l5']:.1f}")
                if "outs" in m and "pitcher" in m and f.get("ip_l5"): z += 0.25*((f["ip_l5"]-IP0)); bits.append(f"IPL5 {f['ip_l5']:.1f}")

        q_list.append(1.0/(1.0+math.exp(-z)))
        why.append(" • ".join(bits))
    out = df.copy()
    out["q_enh"] = q_list
    out["q_notes"] = why
    return out

def calibrate_vs_market(df: pd.DataFrame, lam: float=0.25) -> pd.DataFrame:
    if df is None: return pd.DataFrame()
    if df.empty:
        df = df.copy()
        if "p_market" not in df.columns: df["p_market"] = []
        if "q_enh" not in df.columns: df["q_enh"] = []
        df["q_final"] = df.get("q_enh", df.get("p_market", 0.5))
        return df
    p = df["q_enh"].fillna(df["p_market"].fillna(0.5)).clip(1e-6, 1-1e-6).astype(float)
    m = df["p_market"].fillna(0.5).clip(1e-6, 1-1e-6).astype(float)
    z = (1-lam)*p.map(lambda x: math.log(x/(1-x))) + lam*m.map(lambda x: math.log(x/(1-x)))
    df = df.copy()
    df["q_final"] = z.map(lambda x: 1.0/(1.0+math.exp(-x)))
    return df

def ensure_prob_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None: return pd.DataFrame()
    df = df.copy()
    if "p_market" not in df.columns:
        df["p_market"] = df["american_odds"].map(american_to_prob)
    else:
        df["p_market"] = df["p_market"].fillna(df["american_odds"].map(american_to_prob))
    if "q_enh" not in df.columns:
        df["q_enh"] = df["p_market"].fillna(0.5)
    if "q_final" not in df.columns:
        df["q_final"] = df["q_enh"].fillna(df["p_market"]).fillna(0.5)
    return df

enh_raw = enhanced_q(pool_base, use_weather=True, use_park=True, use_form=True) if not pool_base.empty else pool_base.copy()
enh_cal = calibrate_vs_market(enh_raw, lam=0.25)
enh = ensure_prob_columns(enh_cal)
if not enh.empty:
    enh["edge"] = enh["q_final"].fillna(0.5) - enh["p_market"].fillna(0.5)

# ===================== High-Odds Parlay Builder (8–9 legs) =====================
import math
import re

ALLOWED_MARKETS = {
    "Pitcher Win", "Pitcher Wins", "pitcher_record_a_win",
    "Pitcher Outs", "pitcher_outs",
    "Pitcher Strikeouts", "pitcher_strikeouts",
    "Pitcher Walks", "pitcher_walks",
}

def _col(df, names, default=None):
    for n in names:
        if n in df.columns:
            return df[n]
    return default

def _to_decimal_from_american(a):
    try:
        a = float(a)
        if a >= 0:
            return 1 + (a / 100.0)
        else:
            return 1 + (100.0 / abs(a))
    except Exception:
        return math.nan

def _to_prob_from_decimal(d):
    try:
        d = float(d)
        return 1.0 / d if d > 1 else math.nan
    except Exception:
        return math.nan

def _to_prob_from_percentish(x):
    # accepts 0.62, "62%", "0.62", 62
    try:
        if isinstance(x, str) and x.strip().endswith("%"):
            return float(x.strip().replace("%",""))/100.0
        xv = float(x)
        return xv/100.0 if xv > 1.01 else xv
    except Exception:
        return math.nan

def _pct(x): 
    return f"{round(100*x):d}%" if isinstance(x,(float,int)) and x==x else "--"

def _parse_desc_for_player_team(desc):
    # tries formats like "A. Gray (STL) O5.5 Ks" or "Over 5.5 Ks -- C. Rodon (NYY)"
    if not isinstance(desc, str): return ("", "")
    m = re.search(r"([A-Z]\.\s*[A-Za-z\-']+)\s*\(([A-Z]{2,3})\)", desc)
    if m: return (m.group(1).strip(), m.group(2).strip())
    # fallback: ALLCAPS team code in parentheses anywhere
    m2 = re.search(r"\(([A-Z]{2,3})\)", desc)
    return ("", m2.group(1)) if m2 else ("","")

def normalize_pool_columns(df):
    df = df.copy()
    # Base odds
    if "american_odds" not in df.columns:
        if "Odds" in df.columns: df["american_odds"] = df["Odds"]
        elif "american" in df.columns: df["american_odds"] = df["american"]
        else: df["american_odds"] = math.nan
    # Decimal odds
    if "decimal_odds" not in df.columns:
        if "Dec" in df.columns: df["decimal_odds"] = df["Dec"]
        else: df["decimal_odds"] = df["american_odds"].apply(_to_decimal_from_american)
    # Market probs
    if "p_market" not in df.columns:
        if "Market" in df.columns: df["p_market"] = df["Market"].apply(_to_prob_from_percentish)
        else: df["p_market"] = df["decimal_odds"].apply(_to_prob_from_decimal)
    # Model probs
    if "q_model" not in df.columns:
        if "q" in df.columns: df["q_model"] = df["q"].apply(_to_prob_from_percentish)
        elif "Model q %" in df.columns: df["q_model"] = df["Model q %"].apply(_to_prob_from_percentish)
        else: df["q_model"] = math.nan
    # Category
    if "category" not in df.columns:
        if "market_type" in df.columns: df["category"] = df["market_type"]
        else:
            df["category"] = _col(df, ["Category","Market","type"], "")
    # Side/Line
    if "side" not in df.columns:
        df["side"] = _col(df, ["side","Side"], "")
    if "line" not in df.columns:
        df["line"] = _col(df, ["line","Line"], "")
    # Player/Team
    if "player_name" not in df.columns:
        df["player_name"] = _col(df, ["player","Player","name"], "")
        # parse from description if still blank
        from_desc = df.get("description")
        if from_desc is not None:
            extra = from_desc.apply(_parse_desc_for_player_team).apply(lambda t: t[0])
            df.loc[df["player_name"].eq(""), "player_name"] = extra
    if "team_abbr" not in df.columns:
        df["team_abbr"] = _col(df, ["team","Team","team_abbr"], "")
        if "description" in df.columns:
            t2 = df["description"].apply(_parse_desc_for_player_team).apply(lambda t: t[1])
            df.loc[df["team_abbr"].eq(""), "team_abbr"] = t2
    # Game / pitcher key
    if "game_id" not in df.columns:
        df["game_id"] = _col(df, ["game_id","Game","match_id"], "")
    df["player_key"] = df["player_name"].fillna("").str.replace(r"\.","", regex=True).str.strip().str.lower()
    # Edge/EV
    if "edge" not in df.columns:
        df["edge"] = df["q_model"] - df["p_market"]
    if "ev" not in df.columns:
        df["ev"] = df["q_model"] * (df["decimal_odds"] - 1) - (1 - df["q_model"])
    return df

def is_allowed_market(cat: str):
    if not isinstance(cat,str): return False
    c = cat.strip().lower()
    for k in ALLOWED_MARKETS:
        if c == k.lower():
            return True
    # soft mapping for your internal strings
    # e.g., "Pitcher Ks" -> Pitcher Strikeouts
    if "strikeout" in c: return True
    if "outs" in c: return True
    if "walk" in c: return True
    if "win" in c: return True
    return False

def variance_penalty(row):
    desc = str(row.get("description","")).upper()
    # avoid Coors/altitude chaos unless it's a prop with big edge
    if "COL@STL" in desc or "COL@" in desc or "@COL" in desc or "COORS" in desc:
        return 8.0
    return 0.0

def value_boost(american):
    try:
        a = float(american)
        # encourage plus money modestly; don’t over-reward giant dogs
        return 0.20 * (a/100.0) if a > 0 else 0.10 * (-a/150.0)
    except Exception:
        return 0.0

def parlay_score(row):
    q = float(row.get("q_model", math.nan))
    p = float(row.get("p_market", math.nan))
    a = row.get("american_odds", 0)
    if not (q==q and p==p): return -1e9
    edge = (q - p) * 100.0
    conf = (q - 0.50) * 100.0
    vb   = value_boost(a) * 100.0
    pen  = variance_penalty(row)
    return edge + 0.75*conf + vb - pen

def build_high_odds_parlay(pool, target_legs=9, min_q=0.60, min_edge=0.0):
    df = normalize_pool_columns(pool)
    df = df[df["category"].apply(is_allowed_market)].copy()
    # odds guardrails per market
    def ok_odds(row):
        a = row.get("american_odds", -9999)
        cat = str(row.get("category","")).lower()
        try: a=float(a)
        except: return False
        if "win" in cat:
            return -180 <= a <= +180
        # props can be a little wider
        return -250 <= a <= +220
    df = df[df.apply(ok_odds, axis=1)]
    df = df[(df["q_model"] >= min_q) & (df["edge"] >= min_edge)].copy()
    if df.empty:
        return [], {}
    # Rank by score, preferring Pitcher Win first
    df["score"] = df.apply(parlay_score, axis=1)
    # Diversity constraints
    used_games = set()
    used_pitchers = set()
    picks = []

    # Aim for 4-5 Pitcher Win YES first
    wins = df[df["category"].str.lower().str.contains("win")].sort_values("score", ascending=False)
    for _, r in wins.iterrows():
        g = r.get("game_id",""); pk = r.get("player_key","")
        if g in used_games or pk in used_pitchers: continue
        picks.append(r)
        used_games.add(g); used_pitchers.add(pk)
        if len([x for x in picks if "win" in str(x.get("category","")).lower()]) >= 5: break

    # Fill remainder with Outs/Ks/Walks
    rest = df[~df.index.isin([r.name for r in picks])].sort_values("score", ascending=False)
    for _, r in rest.iterrows():
        g = r.get("game_id",""); pk = r.get("player_key","")
        if g in used_games or pk in used_pitchers: continue
        picks.append(r)
        used_games.add(g); used_pitchers.add(pk)
        if len(picks) >= target_legs: break

    # Compute parlay metrics
    if not picks:
        return [], {}

    import numpy as np
    P = np.prod([float(x.get("q_model",0.0)) for x in picks])
    D = np.prod([float(x.get("decimal_odds",1.0)) for x in picks])
    EV = P * (D - 1) - (1 - P)

    # Card formatting
    def fmt_card(r):
        name = str(r.get("player_name","")).strip()
        team = str(r.get("team_abbr","")).strip().upper()[:3]
        # Shorten first name to initial if full
        if name and "." not in name and " " in name:
            name = f"{name.split()[0][0]}. {' '.join(name.split()[1:])}"
        # Try to infer a compact bet label
        cat = str(r.get("category",""))
        side = str(r.get("side","")).capitalize()
        line = r.get("line","")
        # Fallbacks from description if needed
        if not side or side.lower() not in {"over","under","yes","no"}:
            desc = str(r.get("description",""))
            m = re.search(r"\b(Over|Under|Yes|No)\b", desc, re.I)
            if m: side = m.group(1).capitalize()
        if not line or (isinstance(line,str) and not line.strip()):
            desc = str(r.get("description",""))
            m = re.search(r"([0-9]+(\.[05])?)", desc)
            if m: line = m.group(1)
        label = ""
        lc = cat.lower()
        if "strikeout" in lc:
            label = "Ks"
        elif "outs" in lc:
            label = "Outs"
        elif "walk" in lc:
            label = "BB"
        elif "win" in lc:
            label = "Win"
        else:
            label = cat
        odds = int(float(r.get("american_odds",0)))
        q = float(r.get("q_model", math.nan)); p = float(r.get("p_market", math.nan))
        # Build chip line
        chips = f"Odds {odds:+d} · q {_pct(q)} · Market {_pct(p)} · Edge {_pct(q-p)}"
        # Headline
        who = (name if name else "").strip()
        who = who if who else "(?)"
        t = f" ({team})" if team else ""
        head = f"{who}{t} -- {side} {line} {label}" if label!="Win" else f"{who}{t} -- Win {side}"
        return head, chips

    cards = []
    for r in picks:
        h, c = fmt_card(r)
        cards.append((h,c))

    summary = {
        "legs": len(picks),
        "hit_prob_est": P,
        "decimal_combined": D,
        "american_combined": (D-1)*100 if D>1 else 0,
        "ev_est": EV,
        "note": "Greedy high-odds builder: 4–5 Wins first, then Outs/Ks/Walks; q>=60%, edge>=0; no duplicate pitcher/game; Coors penalized."
    }
    return cards, summary

# ---------- Streamlit UI hook (put this where you render tabs/buttons) ----------
if "pool_base" in globals():
    _cards, _sum = build_high_odds_parlay(pool_base, target_legs=9, min_q=0.60, min_edge=0.00)
    with st.expander("High‑Odds Parlay (8–9 legs) -- auto‑built", expanded=True):
        if _cards:
            colA, colB, colC = st.columns([1,1,1])
            colA.metric("Legs", _sum["legs"])
            colB.metric("Est. Hit", f"{_pct(_sum['hit_prob_est'])}")
            colC.metric("Est. American", f"{int(round(_sum['american_combined'])):+d}")
            st.caption(_sum["note"])
            for i,(h,c) in enumerate(_cards,1):
                st.markdown(f"**{i}. {h}**")
                st.caption(c)
        else:
            st.info("No qualifying legs found with q≥60% and edge≥0. Widen odds or include more categories.")
# ================================================================================
# ===================== TOP-8 PARLAY PICKER (Pitcher-centric) =====================
import math, re
import numpy as np
import pandas as pd
import streamlit as st

ALLOWED_MARKETS = {
    "Pitcher Win", "Pitcher Wins", "pitcher_record_a_win",
    "Pitcher Outs", "pitcher_outs",
    "Pitcher Strikeouts", "pitcher_strikeouts",
    "Pitcher Walks", "pitcher_walks",
}

def _to_decimal_from_american(a):
    try:
        a = float(a)
        return 1 + (a/100.0) if a >= 0 else 1 + (100.0/abs(a))
    except Exception:
        return math.nan

def _to_prob_from_decimal(d):
    try:
        d = float(d)
        return 1.0/d if d > 1 else math.nan
    except Exception:
        return math.nan

def _to_prob_from_percentish(x):
    try:
        if isinstance(x, str) and x.strip().endswith("%"):
            return float(x.strip().replace("%",""))/100.0
        xv = float(x)
        return xv/100.0 if xv > 1.01 else xv
    except Exception:
        return math.nan

def _pct(x):
    return f"{round(100*float(x))}%" if isinstance(x,(float,int)) or (isinstance(x,float) and x==x) else "--"

def _parse_desc_for_player_team(desc):
    if not isinstance(desc, str): return ("", "")
    m = re.search(r"([A-Z]\.\s*[A-Za-z\-']+)\s*\(([A-Z]{2,3})\)", desc)
    if m: return (m.group(1).strip(), m.group(2).strip())
    m2 = re.search(r"\(([A-Z]{2,3})\)", desc)
    return ("", m2.group(1)) if m2 else ("","")

def normalize_pool_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = (df or pd.DataFrame()).copy()

    # --- Odds ---
    if "american_odds" not in df.columns:
        if "Odds" in df.columns: df["american_odds"] = df["Odds"]
        elif "american" in df.columns: df["american_odds"] = df["american"]
        else: df["american_odds"] = math.nan

    if "decimal_odds" not in df.columns:
        if "Dec" in df.columns: df["decimal_odds"] = df["Dec"]
        else: df["decimal_odds"] = df["american_odds"].apply(_to_decimal_from_american)

    # --- Market prob ---
    if "p_market" not in df.columns:
        if "Market" in df.columns: df["p_market"] = df["Market"].apply(_to_prob_from_percentish)
        else: df["p_market"] = df["decimal_odds"].apply(_to_prob_from_decimal)

    # --- Model prob ---
    if "q_model" not in df.columns:
        for c in ["q","Model q %","q_model"]:
            if c in df.columns:
                df["q_model"] = df[c].apply(_to_prob_from_percentish) if c != "q_model" else df[c]
                break
        if "q_model" not in df.columns: df["q_model"] = math.nan

    # --- Category/Side/Line ---
    if "category" not in df.columns:
        if "market_type" in df.columns: df["category"] = df["market_type"]
        else:
            for c in ["Category","Market","type"]:
                if c in df.columns: df["category"] = df[c]; break
            if "category" not in df.columns: df["category"] = ""

    if "side" not in df.columns: df["side"] = df.get("side","")
    if "line" not in df.columns: df["line"] = df.get("line","")

    # --- Player/Team/Game ---
    if "player_name" not in df.columns:
        for c in ["player","Player","name"]:
            if c in df.columns: df["player_name"] = df[c]; break
        if "player_name" not in df.columns: df["player_name"] = ""
        if "description" in df.columns:
            df.loc[df["player_name"].eq(""), "player_name"] = df["description"].apply(_parse_desc_for_player_team).apply(lambda t: t[0])

    if "team_abbr" not in df.columns:
        for c in ["team_abbr","team","Team"]:
            if c in df.columns: df["team_abbr"] = df[c]; break
        if "team_abbr" not in df.columns: df["team_abbr"] = ""
        if "description" in df.columns:
            t2 = df["description"].apply(_parse_desc_for_player_team).apply(lambda t: t[1])
            df.loc[df["team_abbr"].eq(""), "team_abbr"] = t2

    if "game_id" not in df.columns:
        for c in ["game_id","Game","match_id","event_id"]:
            if c in df.columns: df["game_id"] = df[c]; break
        if "game_id" not in df.columns: df["game_id"] = ""

    # --- Keys/Edge/EV ---
    df["player_key"] = df["player_name"].fillna("").astype(str)\
        .str.replace(r"\.","", regex=True).str.strip().str.lower()
    if "edge" not in df.columns:
        df["edge"] = df["q_model"] - df["p_market"]
    if "ev" not in df.columns:
        df["ev"] = df["q_model"] * (df["decimal_odds"] - 1) - (1 - df["q_model"])

    # De-dup noisy legs
    dedup_keys = ["game_id","team_abbr","player_key","category","side","line","american_odds"]
    dedup_keys = [k for k in dedup_keys if k in df.columns]
    if dedup_keys:
        df = df.drop_duplicates(subset=dedup_keys).reset_index(drop=True)

    return df

def is_allowed_market(cat: str) -> bool:
    if not isinstance(cat,str): return False
    c = cat.strip().lower()
    if any(c == k.lower() for k in ALLOWED_MARKETS): return True
    # Fuzzy map common labels
    if "strikeout" in c or "outs" in c or "walk" in c or "win" in c:
        return True
    return False

def value_boost(american):
    try:
        a = float(american)
        return 0.20*(a/100.0) if a > 0 else 0.10*(-a/150.0)
    except Exception:
        return 0.0

def variance_penalty(row):
    # Light heuristic to avoid chaos spots; customize if you like.
    desc = str(row.get("description","")).upper()
    if "COORS" in desc or " @ COL" in desc or "COL@" in desc:
        return 8.0
    return 0.0

def parlay_score(row):
    q = float(row.get("q_model", math.nan))
    p = float(row.get("p_market", math.nan))
    a = row.get("american_odds", 0)
    if not (q==q and p==p): return -1e9
    edge = (q - p) * 100.0
    conf = (q - 0.50) * 100.0
    vb   = value_boost(a) * 100.0
    pen  = variance_penalty(row)
    # Prefer Pitcher Win slightly when edges tie
    bonus = 2.0 if "win" in str(row.get("category","")).lower() else 0.0
    return edge + 0.75*conf + vb + bonus - pen

def build_top8(pool: pd.DataFrame, target_legs=8, min_q=0.60, min_edge=0.0):
    df = normalize_pool_columns(pool)
    if df.empty:
        return [], {}

    # Allowed markets & reasonable odds windows
    def ok_row(r):
        if not is_allowed_market(r.get("category","")): return False
        try: a = float(r.get("american_odds", -9999))
        except: return False
        lc = str(r.get("category","")).lower()
        if "win" in lc:      # keep in a tighter band
            return -200 <= a <= +180
        # props a bit wider
        return -260 <= a <= +220

    df = df[df.apply(ok_row, axis=1)].copy()
    df = df[(df["q_model"] >= min_q) & (df["edge"] >= min_edge)].copy()
    if df.empty:
        return [], {}

    df["score"] = df.apply(parlay_score, axis=1)

    # Diversity: avoid duplicate pitcher/game
    used_games, used_pitchers = set(), set()
    picks = []

    # 1) take up to 4–5 Pitcher Win YES first
    wins = df[df["category"].str.lower().str.contains("win")].sort_values("score", ascending=False)
    for _, r in wins.iterrows():
        g, pk = r.get("game_id",""), r.get("player_key","")
        if g in used_games or pk in used_pitchers: continue
        picks.append(r); used_games.add(g); used_pitchers.add(pk)
        if len([x for x in picks if "win" in str(x.get("category","")).lower()]) >= 5: break

    # 2) fill the rest with Outs/Ks/Walks
    rest = df[~df.index.isin([r.name for r in picks])].sort_values("score", ascending=False)
    for _, r in rest.iterrows():
        g, pk = r.get("game_id",""), r.get("player_key","")
        if g in used_games or pk in used_pitchers: continue
        picks.append(r); used_games.add(g); used_pitchers.add(pk)
        if len(picks) >= target_legs: break

    if not picks:
        return [], {}

    # Parlay math
    P = float(np.prod([float(x.get("q_model",0.0)) for x in picks]))
    D = float(np.prod([float(x.get("decimal_odds",1.0)) for x in picks]))
    EV = P*(D - 1) - (1 - P)

    # Card formatting
    def fmt_card(r):
        name = str(r.get("player_name","")).strip()
        team = str(r.get("team_abbr","")).strip().upper()[:3]
        if name and "." not in name and " " in name:
            name = f"{name.split()[0][0]}. {' '.join(name.split()[1:])}"
        cat  = str(r.get("category",""))
        side = str(r.get("side","")).capitalize() if r.get("side","") else ""
        line = str(r.get("line",""))
        # Fallback from description
        desc = str(r.get("description",""))
        if not side or side.lower() not in {"over","under","yes","no"}:
            m = re.search(r"\b(Over|Under|Yes|No)\b", desc, re.I)
            if m: side = m.group(1).capitalize()
        if not line.strip():
            m = re.search(r"([0-9]+(\.[05])?)", desc)
            if m: line = m.group(1)
        label = ("Ks" if "strikeout" in cat.lower()
                 else "Outs" if "outs" in cat.lower()
                 else "BB" if "walk" in cat.lower()
                 else "Win" if "win" in cat.lower()
                 else cat)
        odds = int(float(r.get("american_odds",0)))
        q = float(r.get("q_model", math.nan)); p = float(r.get("p_market", math.nan))
        chips = f"Odds {odds:+d} · q {_pct(q)} · Market {_pct(p)} · Edge {_pct(q-p)}"
        who = name or "(?)"; t = f" ({team})" if team else ""
        head = f"{who}{t} -- {('Win '+side) if label=='Win' else f'{side} {line} {label}'}"
        return head, chips

    cards = [fmt_card(r) for r in picks]
    summary = {
        "legs": len(picks),
        "hit_prob_est": P,
        "decimal_combined": D,
        "american_combined": (D-1)*100 if D>1 else 0,
        "ev_est": EV,
        "note": "Top‑8: up to five Pitcher Wins, then Outs/Ks/Walks; q≥60%, edge≥0; no dup pitcher/game."
    }
    return cards, summary

def render_top8_section(pool_base: pd.DataFrame):
    st.subheader("Top 8 Picks -- Today")
    if pool_base is None or (isinstance(pool_base, pd.DataFrame) and pool_base.empty):
        st.info("No candidate legs in memory. Fetch or upload board/props first.")
        return
    cards, summ = build_top8(pool_base, target_legs=8, min_q=0.60, min_edge=0.00)
    if not cards:
        st.warning("No qualifying legs with current filters (q≥60%, edge≥0). Loosen odds/markets or refresh.")
        return
    c1,c2,c3 = st.columns(3)
    c1.metric("Legs", summ["legs"])
    c2.metric("Est. Hit", _pct(summ["hit_prob_est"]))
    c3.metric("Est. American", f"{int(round(summ['american_combined'])):+d}")
    st.caption(summ["note"])
    lines = []
    for i,(h,c) in enumerate(cards,1):
        st.markdown(f"**{i}. {h}**")
        st.caption(c)
        lines.append(f"{i}. {h}  --  {c}")
    st.text_area("Copy picks", value="\n".join(lines), height=180)
# ===============================================================================

# ---- UI hook: call this where you render tabs/sections (after pool_base exists)
try:
    if isinstance(pool_base, pd.DataFrame):
        with st.expander("Auto Picks", expanded=True):
            render_top8_section(pool_base)
except NameError:
    # pool_base not created yet; safe to ignore
    pass
#--------------------------- Filters & UI -----------------------------
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

tabs = st.tabs(["Candidates","Top 20","My Picks","ML Winners & RL Locks","Downloads","Debug"])

with tabs[0]:
    if f.empty:
        st.info("No rows. Fetch data or relax filters.")
    else:
        show_cols = ["description","category","american_odds","q_final","p_market","edge"]
        pretty = f[show_cols].rename(columns={"american_odds":"Odds","q_final":"q (model)","p_market":"Market %","edge":"Edge"})
        pretty["q (model)"] = pretty["q (model)"].map(lambda x: pct(x,1))
        pretty["Market %"] = pretty["Market %"].map(lambda x: pct(x,1))
        pretty["Edge"] = pretty["Edge"].map(lambda x: f"{x:+.1%}")
        st.dataframe(pretty, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Top 20 (by edge, then odds)")
    if f.empty:
        st.info("No rows to rank.")
    else:
        top = f.sort_values(["edge","q_final","american_odds"], ascending=[False,False,True]).head(20)
        for _, r in top.iterrows():
            chips = []
            if "alternate" in _s(r["market_type"]).lower():
                chips.append("Alt")
            chips.extend([
                f"Odds {int(r['american_odds']) if pd.notna(r['american_odds']) else '--'}",
                f"q {pct(r['q_final'])}",
                f"Market {pct(r['p_market'])}",
                f"Edge {r['edge']:+.1%}",
            ])
            checked = st.checkbox("Took ✓", key=f"take_{r['leg_id']}", value=(r["leg_id"] in ss["picks"]))
            if checked:
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
            # White card
            st.markdown(
                f"""
                <div style="border:1px solid #eee;border-radius:14px;padding:14px 16px;background:#fff;">
                    <div style="font-weight:600;margin-bottom:6px;">{r['description']}</div>
                    <div style="display:flex;gap:8px;flex-wrap:wrap;margin:6px 0;">
                        {''.join([f'<span style="background:#f5f6f7;border:1px solid #eee;border-radius:999px;padding:3px 8px;font-size:12px;color:#111;">{c}</span>' for c in chips])}
                    </div>
                    <div style="color:#333;font-size:13px;line-height:1.35;margin-bottom:4px">{_s(r.get('q_notes',''))}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

with tabs[2]:
    st.subheader("My Picks")
    if not ss["picks"]:
        st.info("No picks selected yet. Use the checkboxes in Top 20.")
    else:
        picks_df = pd.DataFrame(list(ss["picks"].values())).sort_values("time", ascending=False)
        show = picks_df[["time","description","american_odds","q_final","p_market","edge"]].copy()
        show = show.rename(columns={"american_odds":"Odds","q_final":"q (model)","p_market":"Market %","edge":"Edge"})
        show["q (model)"] = show["q (model)"].map(lambda x: pct(x,1))
        show["Market %"] = show["Market %"].map(lambda x: pct(x,1))
        show["Edge"] = show["Edge"].map(lambda x: f"{x:+.1%}")
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.download_button(
            "Download My Picks CSV",
            picks_df.to_csv(index=False).encode("utf-8"),
            file_name=f"my_picks_{datetime.now().date().isoformat()}.csv",
            mime="text/csv"
        )

with tabs[3]:
    st.subheader("ML Winners & RL Locks")
    ml = f[f["category"].isin(["Moneyline","Run Line"])].copy() if not f.empty else pd.DataFrame()
    if ml.empty:
        st.info("No ML/RL rows available.")
    else:
        show = ml[["description","american_odds","q_final","p_market","edge"]].rename(columns={
            "american_odds":"Odds","q_final":"q (model)","p_market":"Market %","edge":"Edge"
        })
        show["q (model)"] = show["q (model)"].map(lambda x: pct(x,1))
        show["Market %"] = show["Market %"].map(lambda x: pct(x,1))
        show["Edge"] = show["Edge"].map(lambda x: f"{x:+.1%}")
        st.dataframe(show, use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Downloads")
    def dl(df, label, fn):
        if df is None or df.empty: st.button(label, disabled=True)
        else:
            st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=fn, mime="text/csv")
    today = datetime.now().date().isoformat()
    dl(events_df, "Download Events CSV", f"events_{today}.csv")
    dl(board_df,  "Download Board CSV",  f"board_{today}.csv")
    dl(props_df,  "Download Props CSV",  f"props_{today}.csv")

with tabs[5]:
    dbg = dict(
        events_rows=len(events_df),
        board_rows=len(board_df),
        props_rows=len(props_df),
        pool_rows=len(pool_base),
        filtered_rows=len(f),
        has_api_key=bool(api_key()),
        markets=PROP_MARKETS,
        lock_data=ss["lock_data"],
        auto_snapshot=ss["auto_snapshot"],
        last_server_snapshot=last_server_snapshot_path(),
        last_snapshot_ts=ss.get("last_snapshot_ts"),
    )
    st.code(json.dumps(dbg, indent=2))