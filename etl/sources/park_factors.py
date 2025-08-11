# etl/sources/park_factors.py
# Static run and strikeout factors by home park. Neutral defaults if not found.

from typing import Dict
import pandas as pd

PARK_FACTORS: Dict[str, Dict] = {
    "ArizonaDiamondbacks":   {"park_run_factor":1.03, "park_k_factor":0.98},
    "AtlantaBraves":         {"park_run_factor":1.04, "park_k_factor":0.99},
    "BaltimoreOrioles":      {"park_run_factor":0.98, "park_k_factor":1.01},
    "BostonRedSox":          {"park_run_factor":1.07, "park_k_factor":1.00},
    "ChicagoCubs":           {"park_run_factor":1.04, "park_k_factor":0.99},
    "ChicagoWhiteSox":       {"park_run_factor":1.02, "park_k_factor":1.00},
    "CincinnatiReds":        {"park_run_factor":1.10, "park_k_factor":0.98},
    "ClevelandGuardians":    {"park_run_factor":0.99, "park_k_factor":1.01},
    "ColoradoRockies":       {"park_run_factor":1.25, "park_k_factor":0.95},
    "DetroitTigers":         {"park_run_factor":0.99, "park_k_factor":1.01},
    "HoustonAstros":         {"park_run_factor":1.02, "park_k_factor":1.00},
    "KansasCityRoyals":      {"park_run_factor":1.01, "park_k_factor":1.00},
    "LosAngelesAngels":      {"park_run_factor":1.01, "park_k_factor":1.00},
    "LosAngelesDodgers":     {"park_run_factor":1.03, "park_k_factor":0.99},
    "MiamiMarlins":          {"park_run_factor":0.98, "park_k_factor":1.01},
    "MilwaukeeBrewers":      {"park_run_factor":1.00, "park_k_factor":1.00},
    "MinnesotaTwins":        {"park_run_factor":1.00, "park_k_factor":1.00},
    "NewYorkMets":           {"park_run_factor":0.98, "park_k_factor":1.01},
    "NewYorkYankees":        {"park_run_factor":1.05, "park_k_factor":0.99},
    "OaklandAthletics":      {"park_run_factor":0.99, "park_k_factor":1.01},
    "PhiladelphiaPhillies":  {"park_run_factor":1.03, "park_k_factor":1.00},
    "PittsburghPirates":     {"park_run_factor":0.99, "park_k_factor":1.01},
    "SanDiegoPadres":        {"park_run_factor":0.95, "park_k_factor":1.02},
    "SanFranciscoGiants":    {"park_run_factor":0.95, "park_k_factor":1.02},
    "SeattleMariners":       {"park_run_factor":0.99, "park_k_factor":1.01},
    "StLouisCardinals":      {"park_run_factor":1.02, "park_k_factor":1.00},
    "TampaBayRays":          {"park_run_factor":0.97, "park_k_factor":1.02},
    "TexasRangers":          {"park_run_factor":1.03, "park_k_factor":0.99},
    "TorontoBlueJays":       {"park_run_factor":1.02, "park_k_factor":0.99},
    "WashingtonNationals":   {"park_run_factor":1.01, "park_k_factor":1.00},
}

def build(date: str, dk_df: pd.DataFrame) -> pd.DataFrame:
    def home_from_gid(gid: str) -> str:
        try: return gid.split("-")[1]
        except Exception: return ""
    m = dk_df[dk_df["player_id"].notna()][["player_id","game_id"]].drop_duplicates()
    if m.empty: return pd.DataFrame(columns=["player_id","park_run_factor","park_k_factor"])
    m["home_key"] = m["game_id"].map(lambda x: home_from_gid(x))
    pf = pd.DataFrame([
        {"home_key": k, **v} for k, v in PARK_FACTORS.items()
    ])
    res = m.merge(pf, on="home_key", how="left").drop(columns=["game_id","home_key"])
    res["park_run_factor"] = res["park_run_factor"].fillna(1.00)
    res["park_k_factor"]   = res["park_k_factor"].fillna(1.00)
    return res