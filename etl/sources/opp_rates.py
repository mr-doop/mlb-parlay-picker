# etl/sources/opp_rates.py
# Opponent K% / BB% (seasonal via pybaseball if available; safe defaults otherwise).

from typing import Optional
import pandas as pd
import re

def _team_key(s: str) -> str:
    return re.sub(r'[^A-Za-z]', '', s or '')

def _try_pybaseball_fetch() -> Optional[pd.DataFrame]:
    try:
        from pybaseball import team_batting
        import datetime as dt
        year = dt.date.today().year
        tb = team_batting(year)
        tb = tb.rename(columns={"Team":"team","SO%":"K%","BB%":"BB%"})
        tb["team"] = tb["team"].astype(str)
        tb["K%"] = pd.to_numeric(tb["K%"], errors="coerce")/100.0
        tb["BB%"] = pd.to_numeric(tb["BB%"], errors="coerce")/100.0
        return tb[["team","K%","BB%"]]
    except Exception:
        return None

# Map sanitized tokens used in game_id to pybaseball team names
TEAM_MAP = {
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

def build(date: str, dk_df: pd.DataFrame) -> pd.DataFrame:
    tb = _try_pybaseball_fetch()
    defaults = {"opp_k_rate":0.22, "opp_bb_rate":0.08}

    m = dk_df[dk_df["player_id"].notna()][["player_id","game_id"]].drop_duplicates()
    if m.empty:
        return pd.DataFrame(columns=["player_id","opp_k_rate","opp_bb_rate"])

    def parts(gid):
        try:
            p = gid.split("-")
            return _team_key(p[1]), _team_key(p[2])
        except Exception:
            return "", ""

    m[["home_key","away_key"]] = m["game_id"].map(lambda g: pd.Series(parts(g)))

    if tb is not None:
        name_to_rates = tb.set_index("team")[["K%","BB%"]].to_dict("index")

        def get_rates(key):
            team_name = TEAM_MAP.get(key, None)
            if team_name and team_name in name_to_rates:
                d = name_to_rates[team_name]
                return d["K%"], d["BB%"]
            return defaults["opp_k_rate"], defaults["opp_bb_rate"]

        rates = []
        for _, row in m.iterrows():
            k1, b1 = get_rates(row["home_key"])
            k2, b2 = get_rates(row["away_key"])
            rates.append({"player_id":row["player_id"],
                          "opp_k_rate": (k1+k2)/2.0,
                          "opp_bb_rate": (b1+b2)/2.0})
        out = pd.DataFrame(rates)
    else:
        out = m.assign(opp_k_rate=defaults["opp_k_rate"], opp_bb_rate=defaults["opp_bb_rate"]) \
               .drop(columns=["game_id","home_key","away_key"])

    return out