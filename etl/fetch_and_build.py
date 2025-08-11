import os, csv, argparse, datetime, requests, streamlit as st
from dataclasses import dataclass

DATE_FMT = "%Y-%m-%d"
SPORT = "baseball_mlb"
ODDS_API = "https://api.the-odds-api.com/v4/sports/{sport}/odds"

@dataclass
class DKRow:
    date: str; game_id: str; market_type: str; side: str; team: str
    player_id: str; player_name: str; alt_line: str; american_odds: int

def fetch_dk_odds(date: str):
    key = st.secrets.get("ODDS_API_KEY") or os.getenv("ODDS_API_KEY")
    if not key: raise RuntimeError("ODDS_API_KEY missing (set in Streamlit Secrets).")
    params = {"apiKey": key, "regions": "us", "markets": "h2h,spreads,player_props", "oddsFormat": "american", "bookmakers": "draftkings"}
    url = ODDS_API.format(sport=SPORT)
    r = requests.get(url, params=params, timeout=60); r.raise_for_status()
    return r.json()

def normalize_game_id(date, home, away): return f"{date}-{home.replace(' ','')}-{away.replace(' ','')}"

def map_rows(date, data):
    rows=[]
    for g in data:
        home=g.get("home_team","HOME"); away=g.get("away_team","AWAY"); gid=normalize_game_id(date,home,away)
        for bm in g.get("bookmakers",[]):
            if bm.get("key")!="draftkings": continue
            for m in bm.get("markets",[]):
                key=m.get("key")
                if key=="h2h":
                    for out in m.get("outcomes",[]):
                        side="HOME" if out.get("name")==home else "AWAY"
                        rows.append(DKRow(date,gid,"MONEYLINE",side,out.get("name",""),"","", "", int(out.get("price",0))))
                elif key=="spreads":
                    for out in m.get("outcomes",[]):
                        spread=out.get("point"); team=out.get("name","")
                        mt="RUN_LINE" if abs(spread)==1.5 else "ALT_RUN_LINE"
                        rows.append(DKRow(date,gid,mt,team,team,"","", str(spread), int(out.get("price",0))))
                elif key=="player_props":
                    for out in m.get("outcomes",[]):
                        desc=str(out.get("description","")).lower(); name=str(out.get("name",""))
                        price=int(out.get("price",0)); point=out.get("point",None)
                        if "strikeouts" in desc or "k" in desc:
                            side = "OVER" if "over" in name.lower() else ("UNDER" if "under" in name.lower() else "")
                            rows.append(DKRow(date,gid,"PITCHER_KS",side,"",name,name, str(point) if point is not None else "", price))
                        elif "outs" in desc:
                            side = "OVER" if "over" in name.lower() else ("UNDER" if "under" in name.lower() else "")
                            rows.append(DKRow(date,gid,"PITCHER_OUTS",side,"",name,name, str(point) if point is not None else "", price))
                        elif "walks" in desc:
                            side = "OVER" if "over" in name.lower() else ("UNDER" if "under" in name.lower() else "")
                            rows.append(DKRow(date,gid,"PITCHER_WALKS",side,"",name,name, str(point) if point is not None else "", price))
                        elif "to record a win" in desc or "to get the win" in desc or "pitcher win" in desc:
                            side = "YES" if "yes" in name.lower() else ("NO" if "no" in name.lower() else "")
                            rows.append(DKRow(date,gid,"PITCHER_WIN",side,"",name,name,"", price))
    return rows

def write_csv(path, rows):
    with open(path,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["date","game_id","market_type","side","team","player_id","player_name","alt_line","american_odds"])
        for r in rows: w.writerow([r.date,r.game_id,r.market_type,r.side,r.team,r.player_id,r.player_name,r.alt_line,r.american_odds])

def run(date: str):
    data = fetch_dk_odds(date)
    rows = map_rows(date, data)
    out_odds = f"dk_markets_{date}.csv"
    write_csv(out_odds, rows)
    # Seed features template
    feat = f"features_{date}.csv"
    with open(feat,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["player_id","pitcher_k_rate","pitcher_bb_rate","opp_k_rate","opp_bb_rate","last5_pitch_ct_mean","days_rest","leash_bias","favorite_flag","bullpen_freshness","park_k_factor","ump_k_bias","team_ml_vigfree"])
        w.writerow(["cole_g",0.30,0.07,0.25,0.08,95,5,0.2,1,4.0,1.05,1.02,0.58])
    return out_odds, feat
