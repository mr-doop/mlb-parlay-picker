# etl/sources/opp_rates.py
# Opponent K% / BB% with robust fallbacks (Fangraphs → BRef → Statcast last 21d → built-in per-team table).
# Output columns per player_id: opp_k_rate, opp_bb_rate

from __future__ import annotations
from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np
import re
import datetime as dt

def _team_key(s: str) -> str:
    return re.sub(r'[^A-Za-z]', '', s or '')

TEAM_MAP: Dict[str, str] = {
    "ArizonaDiamondbacks":"Arizona Diamondbacks",
    "AtlantaBraves":"Atlanta Braves",
    "BaltimoreOrioles":"Baltimore Orioles",
    "BostonRedSox":"Boston Red Sox",
    "ChicagoCubs":"Chicago Cubs",
    "ChicagoWhiteSox":"Chicago White Sox",
    "CincinnatiReds":"Cincinnati Reds",
    "ClevelandGuardians":"Cleveland Guardians",
    "ColoradoRockies":"Colorado Rockies",
    "DetroitTigers":"Detroit Tigers",
    "HoustonAstros":"Houston Astros",
    "KansasCityRoyals":"Kansas City Royals",
    "LosAngelesAngels":"Los Angeles Angels",
    "LosAngelesDodgers":"Los Angeles Dodgers",
    "MiamiMarlins":"Miami Marlins",
    "MilwaukeeBrewers":"Milwaukee Brewers",
    "MinnesotaTwins":"Minnesota Twins",
    "NewYorkMets":"New York Mets",
    "NewYorkYankees":"New York Yankees",
    "OaklandAthletics":"Oakland Athletics",
    "PhiladelphiaPhillies":"Philadelphia Phillies",
    "PittsburghPirates":"Pittsburgh Pirates",
    "SanDiegoPadres":"San Diego Padres",
    "SanFranciscoGiants":"San Francisco Giants",
    "SeattleMariners":"Seattle Mariners",
    "StLouisCardinals":"St. Louis Cardinals",
    "TampaBayRays":"Tampa Bay Rays",
    "TexasRangers":"Texas Rangers",
    "TorontoBlueJays":"Toronto Blue Jays",
    "WashingtonNationals":"Washington Nationals",
}

# Last-resort, built-in season-ish baselines (≈ recent league tendencies).
# These make OppK/OppBB vary even if network sources fail on Streamlit Cloud.
FALLBACK_RATES = {
    "Arizona Diamondbacks": (0.205, 0.085),
    "Atlanta Braves": (0.214, 0.089),
    "Baltimore Orioles": (0.215, 0.082),
    "Boston Red Sox": (0.236, 0.078),
    "Chicago Cubs": (0.243, 0.090),
    "Chicago White Sox": (0.230, 0.066),
    "Cincinnati Reds": (0.245, 0.093),
    "Cleveland Guardians": (0.199, 0.082),
    "Colorado Rockies": (0.251, 0.073),
    "Detroit Tigers": (0.249, 0.077),
    "Houston Astros": (0.188, 0.083),
    "Kansas City Royals": (0.215, 0.072),
    "Los Angeles Angels": (0.246, 0.084),
    "Los Angeles Dodgers": (0.212, 0.095),
    "Miami Marlins": (0.238, 0.077),
    "Milwaukee Brewers": (0.238, 0.095),
    "Minnesota Twins": (0.261, 0.092),
    "New York Mets": (0.210, 0.090),
    "New York Yankees": (0.225, 0.101),
    "Oakland Athletics": (0.257, 0.082),
    "Philadelphia Phillies": (0.221, 0.090),
    "Pittsburgh Pirates": (0.235, 0.099),
    "San Diego Padres": (0.203, 0.089),
    "San Francisco Giants": (0.231, 0.092),
    "Seattle Mariners": (0.268, 0.095),
    "St. Louis Cardinals": (0.214, 0.093),
    "Tampa Bay Rays": (0.247, 0.086),
    "Texas Rangers": (0.218, 0.092),
    "Toronto Blue Jays": (0.208, 0.093),
    "Washington Nationals": (0.216, 0.081),
}

def _fg_rates(year: int) -> Optional[pd.DataFrame]:
    try:
        from pybaseball import team_batting
        df = team_batting(year)
        if df is None or df.empty: return None
        team_col = "Team" if "Team" in df.columns else "team"
        kcol = "SO%" if "SO%" in df.columns else "K%"
        bcol = "BB%" if "BB%" in df.columns else "BB%"
        out = df.rename(columns={team_col:"team", kcol:"K%", bcol:"BB%"} ) [["team","K%","BB%"]].copy()
        out["K%"] = pd.to_numeric(out["K%"], errors="coerce")/100.0
        out["BB%"] = pd.to_numeric(out["BB%"], errors="coerce")/100.0
        out = out.dropna()
        return out if not out.empty else None
    except Exception:
        return None

def _bref_rates(year: int) -> Optional[pd.DataFrame]:
    try:
        from pybaseball import batting_stats
        df = batting_stats(year, year)
        if df is None or df.empty: return None
        if not set(["Team","SO","BB","PA"]).issubset(set(df.columns)): return None
        g = df.groupby("Team", as_index=False)[["SO","BB","PA"]].sum()
        g = g[g["PA"] > 0].copy()
        g["K%"] = g["SO"]/g["PA"]
        g["BB%"] = g["BB"]/g["PA"]
        return g.rename(columns={"Team":"team"})[["team","K%","BB%"]]
    except Exception:
        return None

def _statcast_rates(days:int=21) -> Optional[pd.DataFrame]:
    try:
        from pybaseball import statcast
        end = dt.date.today()
        start = end - dt.timedelta(days=days)
        df = statcast(start_dt=start.strftime("%Y-%m-%d"), end_dt=end.strftime("%Y-%m-%d"))
        if df is None or df.empty: return None
        top = df["inning_topbot"].fillna("")
        bat_team = np.where(top.str.lower().str.startswith("top"), df["away_team"], df["home_team"])
        df = df.assign(bat_team=bat_team, events=df["events"].fillna(""))
        dpa = df[df["events"] != ""]
        if dpa.empty: return None
        g = dpa.groupby("bat_team").agg(
            PA=("events","size"),
            K=("events", lambda s: (s.str.contains("strikeout", case=False, na=False)).sum()),
            BB=("events", lambda s: (s.str.contains("walk", case=False, na=False)).sum())
        ).reset_index().rename(columns={"bat_team":"team"})
        g = g[g["PA"] > 0]
        g["K%"] = g["K"]/g["PA"]
        g["BB%"] = g["BB"]/g["PA"]
        return g[["team","K%","BB%"]]
    except Exception:
        return None

def _load_team_rates() -> pd.DataFrame:
    year = dt.date.today().year
    for fn in (_fg_rates, _bref_rates, _statcast_rates):
        df = fn(year) if fn in (_fg_rates,_bref_rates) else fn()
        if df is not None and not df.empty:
            return df
    # built-in fallback (team-specific, not flat)
    return pd.DataFrame(
        [{"team":t, "K%":k, "BB%":b} for t,(k,b) in FALLBACK_RATES.items()]
    )

def build(date: str, dk_df: pd.DataFrame) -> pd.DataFrame:
    m = dk_df[dk_df["player_id"].notna()][["player_id","game_id"]].drop_duplicates()
    if m.empty:
        return pd.DataFrame(columns=["player_id","opp_k_rate","opp_bb_rate"])

    # split game_id → last two tokens are home, away
    def parts(gid: str) -> Tuple[str,str]:
        try:
            p = gid.split("-")
            return _team_key(p[-2]), _team_key(p[-1])
        except Exception:
            return "", ""

    m[["home_key","away_key"]] = m["game_id"].map(lambda g: pd.Series(parts(g)))

    rates = _load_team_rates()
    name_to = rates.set_index("team")[["K%","BB%"]].to_dict("index")

    rows = []
    for _, row in m.iterrows():
        home = TEAM_MAP.get(row["home_key"], None)
        away = TEAM_MAP.get(row["away_key"], None)
        k1 = name_to.get(home, {"K%":0.22})["K%"]; b1 = name_to.get(home, {"BB%":0.08})["BB%"]
        k2 = name_to.get(away, {"K%":0.22})["K%"]; b2 = name_to.get(away, {"BB%":0.08})["BB%"]
        rows.append({"player_id":row["player_id"], "opp_k_rate": float((k1+k2)/2.0), "opp_bb_rate": float((b1+b2)/2.0)})
    return pd.DataFrame(rows)