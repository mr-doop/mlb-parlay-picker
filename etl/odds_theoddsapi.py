"""
DraftKings odds fetcher using The Odds API (https://the-odds-api.com/).
You need an API key. Set env var ODDS_API_KEY and run:

    python etl/odds_theoddsapi.py --date 2025-08-11 --sport baseball_mlb --region us

This script writes `dk_markets_<date>.csv` with the columns expected by the app.
Note: This is a simplified example and may need adjustments based on API responses.
"""
import os, argparse, requests, datetime, csv

API_BASE = "https://api.the-odds-api.com/v4/sports/{sport}/odds"

def fetch_dk_odds(sport: str, regions: str = "us", markets="h2h,spreads,player_props", date=None):
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("Set ODDS_API_KEY in your environment.")
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
        "bookmakers": "draftkings"
    }
    url = API_BASE.format(sport=sport)
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--sport", default="baseball_mlb")
    ap.add_argument("--region", default="us")
    ap.add_argument("--outfile", default=None)
    args = ap.parse_args()

    data = fetch_dk_odds(args.sport, regions=args.region)
    date = args.date
    outfile = args.outfile or f"dk_markets_{date}.csv"

    # Minimal CSV writer (you will likely need to map markets carefully;
    # here we write ML and spreads; props mapping requires per-API analysis.)
    with open(outfile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date","game_id","market_type","side","team","player_id","player_name","alt_line","american_odds"])
        for game in data:
            gid = f"{date}-{game.get('home_team','HOME')}-{game.get('away_team','AWAY')}"
            # Moneyline
            for bm in game.get("bookmakers", []):
                if bm.get("key") != "draftkings":
                    continue
                for market in bm.get("markets", []):
                    mkey = market.get("key")
                    if mkey == "h2h":
                        for out in market.get("outcomes", []):
                            side = "HOME" if out.get("name")==game.get("home_team") else "AWAY"
                            w.writerow([date, gid, "MONEYLINE", side, out.get("name"), "", "", "", out.get("price")])
                    elif mkey == "spreads":
                        for out in market.get("outcomes", []):
                            spread = out.get("point")
                            team = out.get("name")
                            w.writerow([date, gid, "RUN_LINE" if abs(spread)==1.5 else "ALT_RUN_LINE",
                                        team, team, "", "", spread, out.get("price")])
                    # TODO: map player props to PITCHER_KS, PITCHER_OUTS, PITCHER_WALKS, PITCHER_WIN by parsing outcome names/keys

    print(f"Wrote {outfile}")

if __name__ == "__main__":
    main()
