from __future__ import annotations
import pandas as pd

def describe_row(row: pd.Series) -> str:
    mt = row.get("market_type", "")
    team = row.get("team", "")
    side = row.get("side", "")
    player = row.get("player_name", "")
    alt = row.get("alt_line", None)
    game = row.get("game_id", "")

    if mt in {"PITCHER_KS", "PITCHER_OUTS", "PITCHER_WALKS"}:
        return f"{player} {mt.replace('PITCHER_','').title()} {side} {alt} ({game})"
    elif mt == "PITCHER_WIN":
        return f"{player} To Win: {side} ({game})"
    elif mt in {"RUN_LINE", "ALT_RUN_LINE"}:
        return f"{team} {alt:+} RL ({game})"
    elif mt == "MONEYLINE":
        return f"{side} ML ({game})"
    else:
        return f"{mt} {side} {alt} ({game})"
