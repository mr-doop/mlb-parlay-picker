# etl/sources/opp_rates.py
# Opponent K% / BB% with robust fallbacks (Fangraphs → BRef → Statcast last 21d).

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

def _fg_rates(year: int) -> Optional[pd.DataFrame]:
    try:
        from pybaseball import team_batting
        df = team_batting(year)
        if df is None or df.empty:
            return None
        team_col = "Team" if "Team" in df.columns else "team"
        kcol = "SO%" if "SO%" in df.columns else "K%"
        bcol = "BB%" if "BB%" in df.columns else "BB%"
        out = df.rename(columns={team_col:"team", kcol:"K%", bcol:"BB%"})[["team","K%","BB%"]].copy()
        out["K%"] = pd.to_numeric(out["K%"], errors="coerce")/100.0
        out["BB%"] = pd.to_numeric(out["BB%"], errors="coerce")/100.0
        return out.dropna()
    except Exception:
        return None

def _bref_rates(year: int) -> Optional[pd.DataFrame]:
    try:
        from pybaseball import batting_stats
        df = batting_stats(year, year)
        if df is None or df.empty: return None
        keep = ["Team","SO","BB","PA"]
        if any(k not in df.columns for k in keep): return None
        g = df.groupby("Team", as_index=False)[["SO","BB","PA"]].sum()
        g = g[g["PA"] > 0].copy()
        g["K%"] = g["SO"]/g["PA"]
        g["BB%"] = g["BB"]/g["PA"]
        return g.rename(columns={"Team":"team"})[["team","K%","BB%"]]
    except Exception:
        return None

def _statcast_rates(days: int = 21) -> Optional[pd.DataFrame]:
    try:
        from pybaseball import statcast
        end = dt.date.today()
        start = end - dt.timedelta(days=days)
        df = statcast(start_dt=start.strftime("%Y-%m-%d"), end_dt=end.strftime("%Y-%m-%d"))
        if df is None or df.empty: return None
        top = df["inning_topbot"].fillna("")
        bat_team = np.where(top.str.lower().str.startswith("top"), df["away_team"], df["home_team"])
        df = df.assign(bat_team=bat_team, events=df["events"].fillna(""))
        is_pa = df["events"] != ""
        dpa = df.loc[is_pa]
        if dpa.empty: return None
        g = dpa.groupby("bat_team").agg(PA=("events","size"),
                                        K=("events", lambda s: (s.str.contains("strikeout", case=False, na=False)).sum()),
                                        BB=("events", lambda s: (s.str.contains("walk", case=False, na=False)).sum()))
        g = g.reset_index().rename(columns={"bat_team":"team"})
        g = g[g["PA"] > 0].copy()
        g["K%"] = g["K"]/g["PA"]
        g["BB%"] = g["BB"]/g["PA"]
        return g[["team","K%","BB%"]]
    except Exception:
        return None

def _load_team_rates() -> pd.DataFrame:
    year = dt.date.today().year
    for fn in (_fg_rates, _bref_rates, _statcast_rates):
        df = fn(year) if fn in (_fg_rates, _bref_rates) else fn()
        if df is not None and not df.empty:
            return df
    return pd.DataFrame({"team": list(TEAM_MAP.values()), "K%": 0.22, "BB%": 0.08})

def build(date: str, dk_df: pd.DataFrame) -> pd.DataFrame:
    defaults = {"opp_k_rate":0.22, "opp_bb_rate":0.08}
    m = dk_df[dk_df["player_id"].notna()][["player_id","game_id"]].drop_duplicates()
    if m.empty:
        return pd.DataFrame(columns=["player_id","opp_k_rate","opp_bb_rate"])

    def parts(gid: str) -> Tuple[str,str]:
        try:
            p = gid.split("-")
            return _team_key(p[-2]), _team_key(p[-1])  # ← use the last two tokens
        except Exception:
            return "", ""

    m[["home_key","away_key"]] = m["game_id"].map(lambda g: pd.Series(parts(g)))

    rates = _load_team_rates()
    name_to_rates = rates.set_index("team")[["K%","BB%"]].to_dict("index")

    rows = []
    for _, row in m.iterrows():
        home = TEAM_MAP.get(row["home_key"], None)
        away = TEAM_MAP.get(row["away_key"], None)
        k1 = name_to_rates.get(home, {"K%":defaults["opp_k_rate"]})["K%"]
        b1 = name_to_rates.get(home, {"BB%":defaults["opp_bb_rate"]})["BB%"]
        k2 = name_to_rates.get(away, {"K%":defaults["opp_k_rate"]})["K%"]
        b2 = name_to_rates.get(away, {"BB%":defaults["opp_bb_rate"]})["BB%"]
        rows.append({"player_id":row["player_id"],
                     "opp_k_rate": float((k1+k2)/2.0),
                     "opp_bb_rate": float((b1+b2)/2.0)})
    return pd.DataFrame(rows)