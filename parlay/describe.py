# parlay/describe.py
from __future__ import annotations
import re
import pandas as pd

ABBR = {
    "ArizonaDiamondbacks":"ARI","AtlantaBraves":"ATL","BaltimoreOrioles":"BAL","BostonRedSox":"BOS",
    "ChicagoCubs":"CHC","ChicagoWhiteSox":"CWS","CincinnatiReds":"CIN","ClevelandGuardians":"CLE",
    "ColoradoRockies":"COL","DetroitTigers":"DET","HoustonAstros":"HOU","KansasCityRoyals":"KC",
    "LosAngelesAngels":"LAA","LosAngelesDodgers":"LAD","MiamiMarlins":"MIA","MilwaukeeBrewers":"MIL",
    "MinnesotaTwins":"MIN","NewYorkMets":"NYM","NewYorkYankees":"NYY","OaklandAthletics":"OAK",
    "PhiladelphiaPhillies":"PHI","PittsburghPirates":"PIT","SanDiegoPadres":"SDP","SanFranciscoGiants":"SFG",
    "SeattleMariners":"SEA","StLouisCardinals":"STL","TampaBayRays":"TBR","TexasRangers":"TEX",
    "TorontoBlueJays":"TOR","WashingtonNationals":"WSH"
}

def _tok(s: str) -> str: return re.sub(r'[^A-Za-z]', '', s or '')
def _abbr(tok: str) -> str: return ABBR.get(tok, (tok[:3] or "???").upper())

def matchup(gid: str) -> str:
    try:
        p = gid.split("-")
        home = _abbr(_tok(p[-2])); away = _abbr(_tok(p[-1]))
        return f"{away}@{home}"
    except Exception:
        return "???@???"

def compact_player(name: str) -> str:
    parts = [p for p in name.replace(".", "").split(" ") if p]
    if not parts: return name
    if len(parts) == 1: return parts[0]
    return f"{parts[0][0]}. {parts[-1]}"

def describe_row(r: pd.Series) -> str:
    mt   = str(r.get("market_type",""))
    side = str(r.get("side",""))
    line = r.get("alt_line","")
    name = str(r.get("player_name","")).strip()
    gid  = r.get("game_id","")
    m    = matchup(gid)

    if mt == "MONEYLINE":
        team = _abbr(_tok(str(r.get("team",""))))
        return f"{team} ML ({m})"

    if mt in ("RUN_LINE","ALT_RUN_LINE"):
        team = _abbr(_tok(str(r.get("team",""))))
        try: ln = float(line); ln_txt = f"{ln:+.1f}"
        except Exception: ln_txt = str(line)
        return f"{team} {ln_txt} ({m})"

    if mt == "PITCHER_KS":
        return f"{compact_player(name)} ({m.split('@')[1]}) {('O' if side=='OVER' else 'U')}{line} Ks"
    if mt == "PITCHER_OUTS":
        return f"{compact_player(name)} ({m.split('@')[1]}) {('O' if side=='OVER' else 'U')}{line} Outs"
    if mt == "PITCHER_WALKS":
        return f"{compact_player(name)} ({m.split('@')[1]}) {('O' if side=='OVER' else 'U')}{line} BB"
    if mt == "PITCHER_WIN":
        return f"{compact_player(name)} ({m.split('@')[1]}) Win {side}"

    return f"{mt} -- {side} {line} ({m})"