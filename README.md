# MLB Parlay Picker — MVP (DraftKings + Alt Lines)

This is a **minimal working MVP** that helps you:
- Import a CSV of **DraftKings** markets (including **alternate lines**).
- Compute **vig-free implied probabilities** for each side/line.
- Build **cross-game** parlays (to avoid correlation in v1) targeting total odds (default **+500**).
- Focus on **Pitcher Ks, Pitcher Outs, Pitcher Win, Pitcher Walks**, **Moneyline**, **Run Line**, **Alternate Run Lines**.

> **Note:** This MVP uses **market-implied** probabilities (vig removed) as the model probability `q`.
> It's designed so you can use it **today/tomorrow**. A proper model can be added in a later iteration.

## Quick Start

1) **Install Python 3.11** (or 3.10+).
2) Create a virtualenv and install deps:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
3) Run the app:
```bash
streamlit run app.py
```
4) In the UI, **upload** a CSV of DK markets. A **sample** is included at `sample_data/sample_dk_markets.csv`.

## CSV Format (sample provided)

Columns (header required):

- `date` (YYYY-MM-DD)
- `game_id` (e.g., "2025-08-11-NYY-BOS")
- `market_type` — one of:
  - `MONEYLINE`, `RUN_LINE`, `ALT_RUN_LINE`,
  - `PITCHER_KS`, `PITCHER_OUTS`, `PITCHER_WIN`, `PITCHER_WALKS`
- `side` — one of:
  - For totals/props: `OVER` or `UNDER`
  - For win markets: `YES` or `NO`
  - For moneyline: `HOME` or `AWAY`
  - For run lines: use the **team code** (e.g., `NYY`, `BOS`) on that spread
- `team` (team code like `NYY`, `BOS`) — leave blank for player props
- `player_id` (optional), `player_name` (for player props)
- `alt_line` — numeric; examples:
  - Ks: `4.5`, `5.5`
  - Outs: `15.5`, `17.5`, `18.5`
  - Walks: `1.5`, `2.5`
  - Run line spreads: `-1.5`, `+1.5`, `-0.5`, `-2.5`
  - Moneyline: leave blank
- `american_odds` — integer like `-120`, `+105`

**Pairing Rules (for vig removal):**
- `PITCHER_KS`, `PITCHER_OUTS`, `PITCHER_WALKS`: need **both** `OVER` and `UNDER` rows per (game_id, player_id, alt_line).
- `PITCHER_WIN`: need `YES` and `NO` rows per (game_id, player_id).
- `MONEYLINE`: need `HOME` and `AWAY` rows per `game_id`.
- `RUN_LINE` and `ALT_RUN_LINE`: need **both teams** for same `abs(alt_line)` per `game_id`.

If a pair is missing, the app will compute implied probability from single odds (no vig removal).

## What the App Does

- Converts American odds → **implied probabilities** and removes **vig** when possible.
- Shows a **ranked table** of candidate legs with:
  - Market description, DK odds, **vig-free p**, decimal odds.
- Builds two parlays:
  - **Safety**: chooses highest-probability legs first.
  - **Value**: balances probability and price (`q × ln(odds)` heuristic).
- **Cross-game only** (no same-game combos) for MVP to avoid correlation errors.

## Next Steps (post-MVP)

- Add true projection models for Ks/Outs/BB/Win to get **model q** (not market p).
- Add **Gaussian copula** to support **same-game** combos safely.
- Add ingestion automation for odds/lineups/weather (sanctioned feeds).

Good luck, and hit those +500s responsibly!

---

## 🚀 Deploy to Streamlit Cloud (mobile-friendly)

1) Push this folder to a **public GitHub repo** (top-level contains `app.py`).  
2) Go to **share.streamlit.io** → New app → pick your repo/branch → **Deploy**.  
3) Under your app’s **Settings → Secrets**, paste:
```
ODDS_API_KEY = "YOUR_ODDS_API_KEY_HERE"
```
4) Open the app URL on your **iPhone**.  
5) In the app, toggle **Cloud Mode** to auto-fetch today’s DraftKings board.  
6) Upload the generated `dk_markets_<DATE>.csv` and optional `features_<DATE>.csv`, then click **Build Safety/Value Parlay**.

> Notes
> - Player prop mappings from The Odds API → DK sometimes vary; if a prop doesn’t appear, it may need a small parser tweak in `etl/fetch_and_build.py`.
> - For best results, upload a quick-edited features file to enable **true projections** (sample provided).

