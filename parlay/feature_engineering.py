# parlay/feature_engineering.py

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from pybaseball import team_batting, playerid_lookup, statcast_pitcher

# Static park factors (example values; replace or expand if you have more precise data)
PARK_K_FACTORS = {
    "ARI": 0.99, "ATL": 0.99, "BAL": 1.01, "BOS": 0.98, "CHC": 1.00,
    # ... complete for all MLB parks
}
PARK_BB_FACTORS = {
    "ARI": 1.00, "ATL": 0.98, "BAL": 1.00, "BOS": 1.01, "CHC": 1.00,
    # ... complete for all MLB parks
}

def get_weather_factor(team_abbr):
    """Return a wind-adjusted factor based on current weather at the team's home park."""
    # Coordinates for ballparks (example subset)
    coords = {
        "ARI": (33.4455, -112.0667),
        "ATL": (33.8907, -84.4677),
        # ... complete dictionary for all parks
    }
    latlon = coords.get(team_abbr)
    if not latlon:
        return 1.0
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latlon[0],
                "longitude": latlon[1],
                "hourly": "wind_speed_10m,wind_direction_10m",
                "forecast_days": 1
            },
            timeout=5
        )
        resp.raise_for_status()
        js = resp.json()
        wind_speed = float(js["hourly"]["wind_speed_10m"][0])
        wind_dir = float(js["hourly"]["wind_direction_10m"][0])
        # If wind is blowing out (approx. 200–340°), increase run environment (lower K); otherwise, decrease slightly
        factor = 1.0 + max(0.0, wind_speed/20.0) * (1.0 if 200 <= wind_dir <= 340 else -0.5)
        return max(0.85, min(1.15, factor))
    except Exception:
        return 1.0

def get_opponent_splits(team_abbr, pitcher_hand):
    """
    Fetch opponent strikeout and walk rate splits by hand using pybaseball team batting.
    Returns (opp_k_pct, opp_bb_pct) vs the specified pitcher_hand ('L' or 'R').
    """
    try:
        year = datetime.now().year
        tb = team_batting(year)
        tb["PA"] = tb["AB"] + tb["BB"] + tb["HBP"].fillna(0) + tb["SF"].fillna(0)
        tb["opp_k_pct"] = tb["SO"] / tb["PA"]
        tb["opp_bb_pct"] = tb["BB"] / tb["PA"]
        # Map full team names to abbreviations if necessary
        from ..data.mappings import TEAM_ABBR  # adjust import path based on your project structure
        team_full = [k for k,v in TEAM_ABBR.items() if v == team_abbr]
        if not team_full:
            return 0.22, 0.08
        row = tb[tb["Team"].str.contains(team_full[0])].iloc[0]
        return float(row["opp_k_pct"]), float(row["opp_bb_pct"])
    except Exception:
        return 0.22, 0.08  # league-average fallback

def get_pitcher_form(pitcher_name, lookback_days=60):
    """
    Compute rolling K9 and BB9 over last `lookback_days` using statcast data.
    Returns a dict: {'k9': value, 'bb9': value, 'ip': value, 'pc': value}
    """
    try:
        first, last = pitcher_name.split()[0], pitcher_name.split()[-1]
        ids = playerid_lookup(last=last, first=first)
        if ids.empty:
            return {}
        mlbam = int(ids.iloc[0]["key_mlbam"])
        end = datetime.now().date()
        start = end - timedelta(days=lookback_days)
        df = statcast_pitcher(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), mlbam)
        if df.empty:
            return {}
        df["isK"] = df["description"].str.contains("strikeout", na=False)
        df["isBB"] = df["description"].str.contains("walk", na=False)
        grouped = df.groupby("game_date").agg(
            pitches=("pitch_number","max"),
            Ks=("isK","sum"), BBs=("isBB","sum"),
            outs=("outs_when_up","max")
        ).sort_index(ascending=False).head(5)
        ip = (grouped["outs"].fillna(0) / 3.0).mean()
        pc = grouped["pitches"].mean()
        total_outs = grouped["outs"].fillna(0).sum()
        k9 = grouped["Ks"].sum() / (total_outs/3.0) * 9.0 if total_outs > 0 else 0
        bb9 = grouped["BBs"].sum() / (total_outs/3.0) * 9.0 if total_outs > 0 else 0
        return {"ip": float(ip), "pc": float(pc), "k9": float(k9), "bb9": float(bb9)}
    except Exception:
        return {}

def get_bullpen_fatigue(team_abbr, history_df):
    """
    Estimate bullpen fatigue based on the previous game's reliever usage.
    `history_df` should contain at least: team_abbr, game_date, relievers_used, bullpen_pitches.
    Returns a fatigue index between 0 and 1.
    """
    try:
        recent = history_df[history_df["team_abbr"] == team_abbr].sort_values("game_date", ascending=False).head(1)
        if recent.empty:
            return 0.0
        relievers_used = recent["relievers_used"].iloc[0]
        pitches = recent["bullpen_pitches"].iloc[0]
        # Normalize to [0,1] scale; adjust weights as needed
        return min(1.0, (relievers_used / 5.0) + (pitches / 140.0))
    except Exception:
        return 0.0

def pitch_mix_fit(pitcher_name, opponent_team):
    """
    Placeholder function for pitch mix fit. In practice, you would gather pitch type usage for the pitcher
    and cross-reference with the opponent's whiff rate by pitch type.
    For now, return 0.0 for neutral fit.
    """
    # Implement your logic here; e.g., using Baseball Savant or precomputed tables.
    return 0.0

def expected_ip(pitcher_name):
    """
    Simple heuristic: average IP over last 5 starts or fallback to league average (≈5.5).
    """
    form = get_pitcher_form(pitcher_name)
    return form.get("ip", 5.5)

def expected_pitches(pitcher_name):
    """
    Simple heuristic: average pitch count over last 5 starts or fallback to league average (~90).
    """
    form = get_pitcher_form(pitcher_name)
    return form.get("pc", 90.0)

def park_k_factor(park_abbr):
    return PARK_K_FACTORS.get(park_abbr, 1.0)

def park_bb_factor(park_abbr):
    return PARK_BB_FACTORS.get(park_abbr, 1.0)