# etl/fetch_and_build.py
from __future__ import annotations
import os, io, requests, pandas as pd
import numpy as np

from .sources.mlb_schedule import fetch_schedule
from .sources.probable_pitchers import fetch_probable_pitchers
from .sources.opp_rates import load_opp_rates

ODDS_BASE = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"

TEAM_ABBR = {
    "Atlanta Braves":"ATL","Miami Marlins":"MIA","New York Mets":"NYM","Philadelphia Phillies":"PHI","Washington Nationals":"WSH",
    "Chicago Cubs":"CHC","Cincinnati Reds":"CIN","Milwaukee Brewers":"MIL","Pittsburgh Pirates":"PIT","St. Louis Cardinals":"STL",
    "Arizona Diamondbacks":"ARI","Colorado Rockies":"COL","Los Angeles Dodgers":"LAD","San Diego Padres":"SDP","San Francisco Giants":"SFG",
    "Baltimore Orioles":"BAL","Boston Red Sox":"BOS","New York Yankees":"NYY","Tampa Bay Rays":"TBR","Toronto Blue Jays":"TOR",
    "Chicago White Sox":"CWS","Cleveland Guardians":"CLE","Detroit Tigers":"DET","Kansas City Royals":"KCR","Minnesota Twins":"MIN",
    "Houston Astros":"HOU","Los Angeles Angels":"LAA","Oakland Athletics":"OAK","Seattle Mariners":"SEA","Texas Rangers":"TEX"
}

def _get_api_key() -> str:
    k = os.getenv("ODDS_API_KEY", "")
    if not k:
        raise RuntimeError("Missing ODDS_API_KEY in environment/secrets")
    return k

def _american_to_decimal(x):
    try:
        x = int(x)
        if x > 0:  return 1 + x/100
        if x < 0:  return 1 + 100/abs(x)
    except Exception:
        pass
    return 1.0

def _fetch_dk_board(date_str: str) -> pd.DataFrame:
    params = {
        "apiKey": _get_api_key(),
        "regions": "us",
        "markets": "h2h,spreads",
        "oddsFormat": "american",
        "bookmakers": "draftkings",
        "dateFormat": "iso",
    }
    r = requests.get(ODDS_BASE, params=params, timeout=20)
    if r.status_code == 401:
        raise RuntimeError("Unauthorized from The Odds API (check key/plan)")
    r.raise_for_status()
    data = r.json()

    rows = []
    for ev in data:
        home = ev.get("home_team","")
        away = ev.get("away_team","")
        gid  = f"{date_str}-{away.replace(' ','')}-{home.replace(' ','')}"
        for bm in ev.get("bookmakers", []):
            if bm.get("key") != "draftkings":
                continue
            for mk in bm.get("markets", []):
                key = mk.get("key")
                for out in mk.get("outcomes", []):
                    price = int(out.get("price", 0))
                    team  = out.get("name","")
                    if key == "h2h":
                        rows.append({
                            "date": date_str, "game_id": gid, "market_type": "MONEYLINE",
                            "team": team, "side": "YES", "alt_line": np.nan, "american_odds": price,
                            "player_id": None, "player_name": None
                        })
                    elif key == "spreads":
                        point = out.get("point", None)  # <-- FIX: populate alt_line
                        rows.append({
                            "date": date_str, "game_id": gid, "market_type": "RUN_LINE",
                            "team": team, "side": "YES", "alt_line": point, "american_odds": price,
                            "player_id": None, "player_name": None
                        })
    return pd.DataFrame(rows)

def _build_team_features(odds_df: pd.DataFrame, schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Create team-level projections for ML and RL, normalized for joining."""
    if odds_df is None or odds_df.empty:
        return pd.DataFrame(columns=["team_abbr","market_type","side","alt_line","q_proj"])
    df = odds_df.copy()
    # team_abbr via lookup
    df["team_abbr"] = df["team"].map(TEAM_ABBR).fillna("")
    # decimals & market prob
    df["decimal"] = df["american_odds"].apply(_american_to_decimal)
    df["p_market"] = (1.0/df["decimal"]).replace([np.inf,-np.inf], np.nan).clip(0.01,0.99).fillna(0.5)
    # light calibration: small prior to avoid q_proj == p_market for everything
    # home bump (~0.5pp) when we can infer home team from game_id
    hm = schedule_df[["game_id","home_abbr","away_abbr"]].copy()
    hm["home_is_team"] = False
    feat = df.merge(hm, on="game_id", how="left")
    feat["home_is_team"] = (feat["team_abbr"] == feat["home_abbr"])
    bump = np.where(feat["home_is_team"], 0.005, -0.005)  # +/-0.5 percentage point
    feat["alt_line"] = feat["alt_line"].astype(float)
    # fallback alt_line for RL if missing
    feat.loc[(feat["market_type"]=="RUN_LINE") & (feat["alt_line"].isna()), "alt_line"] = 1.5
    feat["q_proj"] = (feat["p_market"] + bump).clip(0.01, 0.99)
    return feat[["team_abbr","market_type","side","alt_line","q_proj"]].dropna(subset=["team_abbr"]).reset_index(drop=True)

def run(date_str: str):
    """Return (odds_df, feat_df, odds_csv, feat_csv, schedule_df, pitchers_df)."""
    schedule_df = fetch_schedule(date_str)
    try:
        odds_base = _fetch_dk_board(date_str)
    except Exception:
        odds_base = pd.DataFrame(columns=["date","game_id","market_type","team","side","alt_line","american_odds","player_id","player_name"])
    pitchers_df = fetch_probable_pitchers(date_str)
    _ = load_opp_rates()  # warm cache

    # NEW: build features from the board so the CSV is not empty
    feat_df = _build_team_features(odds_base, schedule_df)

    odds_csv = odds_base.to_csv(index=False).encode("utf-8")
    feat_csv = feat_df.to_csv(index=False).encode("utf-8")
    return odds_base, feat_df, odds_csv, feat_csv, schedule_df, pitchers_df