# parlay/describe.py
# Build short, readable row descriptions using team abbreviations.

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

def _tok(s: str) -> str:
    return re.sub(r'[^A-Za-z]', '', s or '')

def _abbr(tok: str) -> str:
    return ABBR.get(tok, (tok[:3] or "???").upper())

def _matchup(gid: str) -> str:
    try:
        p = gid.split("-")
        home = _abbr(_tok(p[1])); away = _abbr(_tok(p[2]))
        return f"{away}@{home}"
    except Exception:
        return ""

def describe_row(r: pd.Series) -> str:
    mt   = str(r.get("market_type",""))
    side = str(r.get("side",""))
    line = r.get("alt_line","")
    name = str(r.get("player_name","")).strip()
    gid  = r.get("game_id","")
    m    = _matchup(gid)

    if mt == "MONEYLINE":
        team = _abbr(_tok(str(r.get("team",""))))
        return f"{team} ML ({m})"
    if mt in ("RUN_LINE","ALT_RUN_LINE"):
        team = _abbr(_tok(str(r.get("team",""))))
        try:
            ln = float(line)
            ln_txt = f"{ln:+.1f}" if abs(ln) != 1.0 else f"{ln:+.1f}"
        except Exception:
            ln_txt = str(line)
        return f"{team} {ln_txt} ({m})"
    if mt == "PITCHER_KS":
        return f"{name} Ks {side} {line} ({m})"
    if mt == "PITCHER_OUTS":
        return f"{name} Outs {side} {line} ({m})"
    if mt == "PITCHER_WALKS":
        return f"{name} BB {side} {line} ({m})"
    if mt == "PITCHER_WIN":
        return f"{name} To Win -- {side} ({m})"
    return f"{mt} -- {side} {line} ({m})"