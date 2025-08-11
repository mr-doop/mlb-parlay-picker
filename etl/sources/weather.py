# etl/sources/weather.py
# Weather + roof inference for MLB ballparks via Open-Meteo (no API key).
# Produces per-game features keyed by game_id, broadcast to player_id.

import math, requests, datetime as dt, re
from typing import Dict
import pandas as pd

def _team_key(s: str) -> str:
    return re.sub(r'[^A-Za-z]', '', s or '')

def _home_from_gid(game_id: str) -> str:
    # "YYYY-MM-DD-Home-Away"
    try:
        return _team_key(game_id.split("-")[1])
    except Exception:
        return ""

# Minimal ballpark map (lat, lon, CF bearing in degrees, roof=True/False)
BALLPARKS: Dict[str, Dict] = {
    "ArizonaDiamondbacks":      {"lat":33.4457,"lon":-112.0667,"cf_bearing": 17, "roof": True},
    "AtlantaBraves":            {"lat":33.8907,"lon":-84.4677, "cf_bearing": 25, "roof": False},
    "BaltimoreOrioles":         {"lat":39.2840,"lon":-76.6216, "cf_bearing":  8, "roof": False},
    "BostonRedSox":             {"lat":42.3467,"lon":-71.0972, "cf_bearing": 15, "roof": False},
    "ChicagoCubs":              {"lat":41.9484,"lon":-87.6553, "cf_bearing":  6, "roof": False},
    "ChicagoWhiteSox":          {"lat":41.8300,"lon":-87.6340, "cf_bearing":  5, "roof": False},
    "CincinnatiReds":           {"lat":39.0979,"lon":-84.5083, "cf_bearing":  5, "roof": False},
    "ClevelandGuardians":       {"lat":41.4962,"lon":-81.6852, "cf_bearing":  4, "roof": False},
    "ColoradoRockies":          {"lat":39.7559,"lon":-104.994,"cf_bearing":  7, "roof": False},
    "DetroitTigers":            {"lat":42.3390,"lon":-83.0485, "cf_bearing":  5, "roof": False},
    "HoustonAstros":            {"lat":29.7573,"lon":-95.3555, "cf_bearing":  8, "roof": True},
    "KansasCityRoyals":         {"lat":39.0517,"lon":-94.4803, "cf_bearing":  6, "roof": False},
    "LosAngelesAngels":         {"lat":33.8003,"lon":-117.882,"cf_bearing":  8, "roof": False},
    "LosAngelesDodgers":        {"lat":34.0739,"lon":-118.241,"cf_bearing":  9, "roof": False},
    "MiamiMarlins":             {"lat":25.7781,"lon":-80.2197, "cf_bearing":  9, "roof": True},
    "MilwaukeeBrewers":         {"lat":43.0280,"lon":-87.9712, "cf_bearing":  8, "roof": True},
    "MinnesotaTwins":           {"lat":44.9817,"lon":-93.2776, "cf_bearing":  8, "roof": False},
    "NewYorkMets":              {"lat":40.7571,"lon":-73.8458, "cf_bearing":  7, "roof": False},
    "NewYorkYankees":           {"lat":40.8296,"lon":-73.9262, "cf_bearing":  8, "roof": False},
    "OaklandAthletics":         {"lat":37.7516,"lon":-122.200,"cf_bearing": 10, "roof": False},
    "PhiladelphiaPhillies":     {"lat":39.9057,"lon":-75.1665, "cf_bearing":  9, "roof": False},
    "PittsburghPirates":        {"lat":40.4469,"lon":-80.0057, "cf_bearing":  8, "roof": False},
    "SanDiegoPadres":           {"lat":32.7073,"lon":-117.157,"cf_bearing":  8, "roof": False},
    "SanFranciscoGiants":       {"lat":37.7786,"lon":-122.389,"cf_bearing":  7, "roof": False},
    "SeattleMariners":          {"lat":47.5914,"lon":-122.332,"cf_bearing":  8, "roof": True},
    "StLouisCardinals":         {"lat":38.6226,"lon":-90.1928, "cf_bearing":  8, "roof": False},
    "TampaBayRays":             {"lat":27.7682,"lon":-82.6534, "cf_bearing":  6, "roof": True},
    "TexasRangers":             {"lat":32.7473,"lon":-97.0831, "cf_bearing":  8, "roof": True},
    "TorontoBlueJays":          {"lat":43.6414,"lon":-79.3894, "cf_bearing":  8, "roof": True},
    "WashingtonNationals":      {"lat":38.8730,"lon":-77.0074, "cf_bearing":  7, "roof": False},
}

def _fetch_hourly(lat: float, lon: float, date: str) -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"
    params = dict(
        latitude=lat, longitude=lon, timezone="auto",
        start_date=date, end_date=date,
        hourly="temperature_2m,relative_humidity_2m,pressure_msl,windspeed_10m,winddirection_10m"
    )
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    if "hourly" not in j: return pd.DataFrame()
    h = j["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])
    return df

def _choose_game_hour(df_hourly: pd.DataFrame) -> pd.Series:
    if df_hourly.empty: return pd.Series(dtype=float)
    target = df_hourly["time"].dt.normalize() + pd.Timedelta(hours=19)  # ~7pm local
    idx = (df_hourly["time"] - target).abs().idxmin()
    return df_hourly.loc[idx]

def build(date: str, dk_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-player features derived from per-game weather/roof."""
    m = dk_df[dk_df["player_id"].notna()][["player_id","game_id"]].drop_duplicates()
    if m.empty:
        return pd.DataFrame(columns=[
            "player_id","temp_f","humidity","pressure_mb","wind_speed_mph",
            "wind_deg","wind_out_to_cf","roof_closed","air_density_index","run_env_delta"
        ])

    games = sorted(m["game_id"].unique().tolist())
    rows = []
    for gid in games:
        home = _home_from_gid(gid)
        park = BALLPARKS.get(home)
        if not park:
            rows.append(dict(game_id=gid, temp_f=70.0, humidity=50.0, pressure_mb=1015.0,
                             wind_speed_mph=0.0, wind_deg=0.0, wind_out_to_cf=0,
                             roof_closed=0, air_density_index=0.0, run_env_delta=0.0))
            continue
        lat, lon = park["lat"], park["lon"]
        cf = float(park["cf_bearing"])
        roof = bool(park["roof"])
        try:
            h = _fetch_hourly(lat, lon, date)
            row = _choose_game_hour(h)
            t_c = float(row.get("temperature_2m", 21.0))
            temp_f = t_c * 9/5 + 32
            humidity = float(row.get("relative_humidity_2m", 50.0))
            pressure_mb = float(row.get("pressure_msl", 1015.0))
            wind_kmh = float(row.get("windspeed_10m", 0.0))
            wind_speed_mph = wind_kmh * 0.621371
            wind_deg = float(row.get("winddirection_10m", 0.0))
        except Exception:
            temp_f, humidity, pressure_mb, wind_speed_mph, wind_deg = 70.0, 50.0, 1015.0, 0.0, 0.0

        out_flag = 1 if (abs(((wind_deg - cf + 180) % 360) - 180) <= 40 and wind_speed_mph >= 8) else 0
        roof_closed = 1 if roof and (temp_f >= 88 or humidity >= 65 or wind_speed_mph >= 14) else 0
        adi = ((temp_f-70)/30.0) + (humidity-50)/100.0 + (wind_speed_mph/20.0)*(1 if out_flag else -0.2)
        if roof_closed: adi = 0.0
        run_env_delta = max(-0.35, min(0.35, 0.10*(temp_f-70)/10 + 0.12*out_flag + 0.08*(humidity-50)/25))

        rows.append(dict(game_id=gid, temp_f=temp_f, humidity=humidity, pressure_mb=pressure_mb,
                         wind_speed_mph=wind_speed_mph, wind_deg=wind_deg, wind_out_to_cf=out_flag,
                         roof_closed=roof_closed, air_density_index=adi, run_env_delta=run_env_delta))

    gdf = pd.DataFrame(rows)
    out = m.merge(gdf, on="game_id", how="left").drop(columns=["game_id"])
    return out