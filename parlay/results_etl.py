# parlay/results_etl.py
from __future__ import annotations
import os, io, glob, zipfile
from datetime import datetime
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

# ------------- Helpers & safe defaults -------------
TEAM_ABBR = {
    "ARI","ATL","BAL","BOS","CHC","CWS","CIN","CLE","COL","DET","HOU","KC",
    "LAA","LAD","MIA","MIL","MIN","NYM","NYY","OAK","PHI","PIT","SDP","SFG",
    "SEA","STL","TBR","TEX","TOR","WSH"
}
def _cap3(s): return ("" if s is None else str(s)).upper()[:3]

def _read_zip_csv(z: zipfile.ZipFile, name: str) -> pd.DataFrame:
    try:
        with z.open(name) as f: return pd.read_csv(f)
    except Exception: return pd.DataFrame()

def _list_snapshot_paths(snapshots_dir: str) -> list[str]:
    os.makedirs(snapshots_dir, exist_ok=True)
    return sorted(glob.glob(os.path.join(snapshots_dir, "*.zip")))

def _unique_pitcher_dates_from_snapshots(snapshots_dir: str) -> pd.DataFrame:
    """Return unique (date, player_name, team_abbr, opp_abbr) from props."""
    rows = []
    for zp in _list_snapshot_paths(snapshots_dir):
        try:
            with zipfile.ZipFile(zp) as z:
                props = _read_zip_csv(z, "props.csv")
                ev    = _read_zip_csv(z, "events.csv")
                if props.empty: continue
                # Derive date map from events (preferred)
                date_map = {}
                if not ev.empty and "event_id" in ev and "start" in ev:
                    ev["date"] = pd.to_datetime(ev["start"]).dt.date.astype(str)
                    date_map = dict(zip(ev["event_id"].astype(str), ev["date"]))
                props["event_id"] = props.get("event_id", "").astype(str)
                props["date"] = props["event_id"].map(date_map)
                if props["date"].isna().all():
                    # fallback: file mtime
                    props["date"] = datetime.fromtimestamp(os.path.getmtime(zp)).date().isoformat()
                # Keep only pitcher markets (we train those)
                m = props.get("market_type","").astype(str).str.lower()
                props = props[m.str.contains("pitcher_")]
                # Identify opponent for Win logic
                team = props.get("team_abbr","").astype(str).str.upper().str[:3]
                home = props.get("home_abbr","").astype(str).str.upper().str[:3]
                away = props.get("away_abbr","").astype(str).str.upper().str[:3]
                opp  = np.where(team.eq(home), away, home)
                df = pd.DataFrame(dict(
                    date  = props["date"],
                    player_name = props.get("player_name","").fillna(""),
                    team_abbr = team,
                    opp_abbr  = opp,
                ))
                df = df[(df["player_name"]!="") & df["team_abbr"].isin(TEAM_ABBR)]
                rows.append(df.drop_duplicates())
        except Exception:
            continue
    if not rows: return pd.DataFrame(columns=["date","player_name","team_abbr","opp_abbr"])
    out = pd.concat(rows, ignore_index=True, sort=False)
    return out.drop_duplicates()

# ------------- External data via pybaseball (free) -------------------
def _lookup_mlbam(player_name: str) -> int | None:
    try:
        from pybaseball import playerid_lookup
        nm = str(player_name).strip()
        if not nm or " " not in nm: return None
        first, last = nm.split()[0], nm.split()[-1]
        ids = playerid_lookup(last=last, first=first)
        if ids is None or ids.empty: return None
        return int(ids.iloc[0]["key_mlbam"])
    except Exception:
        return None

def _pitcher_statcast_for_date(mlbam: int, date: str) -> pd.DataFrame:
    """Return pitch-by-pitch DF for that pitcher on that date (local)."""
    try:
        from pybaseball import statcast_pitcher
        d0 = pd.to_datetime(date).strftime("%Y-%m-%d")
        df = statcast_pitcher(d0, d0, mlbam)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def _team_schedule(year: int, team_abbr: str) -> pd.DataFrame:
    """Team schedule with W/L to estimate pitcher Win; fallback empty."""
    try:
        from pybaseball import schedule_and_record
        df = schedule_and_record(year, team_abbr)
        # Normalize date → str YYYY-MM-DD, W/L codes
        df = df.rename(columns={"Opp":"OPP","W/L":"WL"})
        df["date"] = pd.to_datetime(df["Date"]).dt.date.astype(str)
        keep = df[["date","WL","Tm","Opp"]].copy()
        keep = keep.rename(columns={"Tm":"TEAM","Opp":"OPP"})
        return keep
    except Exception:
        return pd.DataFrame()

# ------------- Aggregate one pitcher/date ----------------------------
def _derive_game_line_from_statcast(df: pd.DataFrame) -> dict:
    """Given a pitch-by-pitch table for one pitcher on one date, compute Ks/BB/Outs."""
    if df is None or df.empty: 
        return {"ks": np.nan, "bb": np.nan, "outs": np.nan, "pitch_count": np.nan}
    # Robust event parsing
    e = df.get("events")
    d = df.get("description")
    isK = (e.fillna("").str.contains("strikeout", case=False) if e is not None else pd.Series(False, index=df.index)) \
          | (d.fillna("").str.contains("strikeout", case=False) if d is not None else pd.Series(False, index=df.index))
    isBB = (e.fillna("").str.contains("walk", case=False) if e is not None else pd.Series(False, index=df.index)) \
           | (d.fillna("").str.contains("walk", case=False) if d is not None else pd.Series(False, index=df.index))
    Ks = int(isK.sum())
    BB = int(isBB.sum())
    outs = float(df.get("outs_when_up", pd.Series([np.nan])).max())
    pc = float(df.get("pitch_number", pd.Series([np.nan])).max())
    return {"ks": Ks, "bb": BB, "outs": outs if pd.notna(outs) else np.nan, "pitch_count": pc if pd.notna(pc) else np.nan}

def _estimate_win_flag(date: str, team_abbr: str, outs: float) -> float | np.nan:
    """Approximate Win: team won that day and SP went ≥15 outs (5.0 IP)."""
    try:
        yr = int(pd.to_datetime(date).year)
        sched = _team_schedule(yr, team_abbr)
        if sched.empty: return np.nan
        row = sched[sched["date"]==date]
        if row.empty: return np.nan
        wl = str(row.iloc[0]["WL"])
        team_won = str(wl).upper().startswith("W")
        if not team_won: return 0.0
        return 1.0 if float(outs or 0) >= 15.0 else 0.0
    except Exception:
        return np.nan

# ------------- Public API -------------------------------------------
def build_results_from_snapshots(
    snapshots_dir: str,
    out_csv: str = "data/pitcher_results.csv",
    refresh: bool = True
) -> pd.DataFrame:
    """
    Assemble (or update) pitcher_results.csv from your snapshot zips.
    Returns the full results dataframe (de-duplicated).
    """
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    base = _unique_pitcher_dates_from_snapshots(snapshots_dir)
    if base.empty:
        return pd.DataFrame(columns=["date","player_name","team_abbr","opp_abbr","ks","bb","outs","win","pitch_count"])

    if os.path.exists(out_csv) and not refresh:
        return pd.read_csv(out_csv)

    # Load existing to avoid rework
    existing = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()
    if not existing.empty:
        existing["date"] = pd.to_datetime(existing["date"]).dt.date.astype(str)
        have = set(zip(existing["date"], existing["player_name"], existing["team_abbr"]))
    else:
        have = set()

    rows = []
    for _, r in base.iterrows():
        date = str(r["date"])
        name = str(r["player_name"])
        team = _cap3(r["team_abbr"])
        opp  = _cap3(r.get("opp_abbr",""))
        key = (date, name, team)
        if key in have: 
            continue
        mlbam = _lookup_mlbam(name)
        sc = _pitcher_statcast_for_date(mlbam, date) if mlbam else pd.DataFrame()
        line = _derive_game_line_from_statcast(sc)
        win = _estimate_win_flag(date, team, line.get("outs", np.nan))
        rows.append(dict(date=date, player_name=name, team_abbr=team, opp_abbr=opp,
                         ks=line["ks"], bb=line["bb"], outs=line["outs"], win=win, pitch_count=line["pitch_count"]))
    add_df = pd.DataFrame(rows)

    out = pd.concat([existing, add_df], ignore_index=True, sort=False) if not existing.empty else add_df
    # clean & dedupe
    out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
    out["team_abbr"] = out["team_abbr"].map(_cap3)
    out = out.drop_duplicates(subset=["date","player_name","team_abbr"]).reset_index(drop=True)
    out.to_csv(out_csv, index=False)
    return out