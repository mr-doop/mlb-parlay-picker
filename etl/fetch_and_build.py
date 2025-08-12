# etl/fetch_and_build.py
from __future__ import annotations
import io
import numpy as np
import pandas as pd
from datetime import date

from etl.sources.schedule import load_schedule_for_date
from etl.sources.pitchers import load_probable_pitchers
from etl.sources.board import load_dk_board  # your existing ML/RL loader
from etl.sources.dk_props import pull_dk_pitcher_props

def _ensure_decimal(df: pd.DataFrame) -> pd.DataFrame:
    from parlay.odds import american_to_decimal
    if "decimal_odds" not in df.columns:
        df["decimal_odds"] = df["american_odds"].apply(
            lambda x: american_to_decimal(int(x)) if pd.notna(x) else np.nan
        )
    return df

def _fix_runline_alt(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["market_type"].isin(["RUN_LINE","ALT_RUN_LINE"])) & (df["alt_line"].isna())
    df.loc[mask, "alt_line"] = 1.5
    return df

def _build_features_from_board(board: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal features so it's never empty: team-level ML & RL with q_proj = market prob.
    Users can upload a richer CSV; the app will override via attach_projections.
    """
    if board is None or board.empty:
        return pd.DataFrame(columns=["team_abbr","market_type","side","alt_line","q_proj"])
    # Market probability from decimal odds
    f = board.copy()
    f["decimal_odds"] = pd.to_numeric(f["decimal_odds"], errors="coerce")
    f["q_proj"] = (1.0 / f["decimal_odds"]).clip(0.01, 0.99)
    keep = ["team_abbr","market_type","side","alt_line","q_proj"]
    # team_abbr will be set later in app for player props; these rows cover ML/RL safely
    f = f[keep].dropna(subset=["q_proj"]).drop_duplicates()
    return f

def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

def run(date_str: str):
    """
    Returns:
      odds_df, feat_df_auto, dk_csv_bytes, feat_csv_bytes, schedule_df, pitchers_df
    """
    sched = load_schedule_for_date(date_str)  # must include game_id, home_abbr, away_abbr, matchup
    pitchers = load_probable_pitchers(date_str)  # must include game_id, player_key, team_abbr

    # 1) DK board (Moneyline / Run Line / etc.)
    board = load_dk_board(date_str)  # your existing game-lines loader
    if board is None:
        board = pd.DataFrame(columns=["date","game_id","market_type","side","alt_line",
                                      "american_odds","player_id","player_name","team"])
    board = _ensure_decimal(board)
    board = _fix_runline_alt(board)

    # 2) DK Pitcher Props (NEW)
    props = pull_dk_pitcher_props(date_str, sched)
    if props is not None and not props.empty:
        props = _ensure_decimal(props)
        board = pd.concat([board, props], ignore_index=True, sort=False)

    # 3) Auto-features from board (non-empty)
    feat_auto = _build_features_from_board(board)

    # 4) Bytes for download buttons
    dk_bytes = _to_csv_bytes(board)
    feat_bytes = _to_csv_bytes(feat_auto)

    return board, feat_auto, dk_bytes, feat_bytes, sched, pitchers