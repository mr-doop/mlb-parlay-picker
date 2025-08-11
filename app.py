# app.py -- MLB Parlay Picker (Clean UI + new tabs + filters)
import os, io, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

from parlay.odds import american_to_decimal
from parlay.builder import build_presets, build_one_tap
from parlay.describe import describe_row, matchup, compact_player, _tok, _abbr
from parlay.ui import inject_css, parlay_card

# ---------- Page ----------
st.set_page_config(page_title="MLB Parlay Picker -- MVP", layout="wide", page_icon="⚾")
inject_css()

# ---------- Small helpers ----------
def add_missing(df: pd.DataFrame, cols, val=np.nan):
    d = df.copy()
    for c in cols:
        if c not in d.columns: d[c] = val
    return d

def dedupe_legs(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["player_id","market_type","side","alt_line","american_odds","game_id"]
    return df.drop_duplicates(subset=[c for c in keys if c in df.columns])

def build_pool(dk_df: pd.DataFrame, feat_df: pd.DataFrame) -> pd.DataFrame:
    df = dk_df.copy()
    if "decimal_odds" not in df.columns:
        df["decimal_odds"] = df["american_odds"].apply(lambda x: american_to_decimal(int(x)))
    if "p_market" not in df.columns:
        dec = df["decimal_odds"].replace(0, np.nan)
        df["p_market"] = (1.0/dec).clip(0.01,0.99).fillna(0.5)
    df = df.merge(feat_df, on="player_id", how="left")

    # model probs
    if "q_proj" in df.columns:
        df["q_model"] = df["q_proj"].clip(0.01,0.99)
    else:
        df["q_model"] = (0.6*df["p_market"] + 0.4*0.50).clip(0.01,0.99)

    df["edge"] = (df["q_model"] - df["p_market"]).fillna(0.0)
    df["ev"]   = (df["q_model"] * df["decimal_odds"] - 1.0).fillna(0.0)

    # Category + description
    def cat(r):
        return {"PITCHER_KS":"Pitcher Ks","PITCHER_OUTS":"Pitcher Outs","PITCHER_WALKS":"Pitcher BB",
                "PITCHER_WIN":"Pitcher Win","MONEYLINE":"Moneyline","RUN_LINE":"Run Line","ALT_RUN_LINE":"Alt Run Line"}.get(r["market_type"], r["market_type"])
    df["category"] = df.apply(cat, axis=1)
    df["description"] = df.apply(describe_row, axis=1)

    # Add helpers for filters
    df["game_code"] = df["game_id"].apply(lambda g: matchup(g))
    df["team_abbr"] = df["team"].apply(lambda t: _abbr(_tok(str(t)))) if "team" in df.columns else ""
    df["american_odds"] = df["american_odds"].astype(int)
    return df

def compact_note(r: pd.Series) -> str:
    parts = []
    if r.get("market_type") == "PITCHER_OUTS" and pd.notna(r.get("expected_ip")):
        ip = float(r["expected_ip"]); parts.append(f"Exp IP {ip:.1f} (O:{int(round(ip*3))}) vs Ln {r.get('alt_line')}")
    if r.get("market_type") == "PITCHER_KS" and pd.notna(r.get("mu_ks")):
        parts.append(f"μK {float(r['mu_ks']):.2f} vs Ln {r.get('alt_line')}")
    if r.get("market_type") == "PITCHER_WALKS" and pd.notna(r.get("mu_bb")):
        parts.append(f"μBB {float(r['mu_bb']):.2f} vs Ln {r.get('alt_line')}")
    if pd.notna(r.get("park_k_factor")): parts.append(f"ParkK {float(r['park_k_factor']):.2f}")
    if pd.notna(r.get("opp_k_rate")):    parts.append(f"OppK {float(r['opp_k_rate']):.0%}")
    if pd.notna(r.get("opp_bb_rate")):   parts.append(f"OppBB {float(r['opp_bb_rate']):.0%}")
    if pd.notna(r.get("last5_pc_mean")): parts.append(f"L5PC {float(r['last5_pc_mean']):.0f}")
    if pd.notna(r.get("days_rest")):     parts.append(f"Rest {int(float(r['days_rest']))}d")
    return " • ".join(parts) if parts else "--"

# ---------- Header ----------
st.title("MLB Parlay Picker -- MVP")
st.caption("DraftKings + Alternate Lines | Clean, compact UI | Model blends market + optional projections (upload features CSV).")

with st.expander("How to use"):
    st.markdown("""
1) Toggle **Cloud Mode** to auto‑fetch today’s board.  
2) (Optional) Upload a **Features CSV** to add true projection column `q_proj`.  
3) Use **Filters** (games/teams/categories + odds range).  
4) Review **Top 20**, **Parlay Presets**, **Alt Line Safety Board**, or **One‑Tap Ticket**.  
""")

# ---------- Cloud mode + uploads ----------
cloud_on = st.toggle("Enable Cloud Mode (auto‑fetch today)", value=False)
today = dt.date.today().strftime("%Y-%m-%d")
odds_file = None; feat_file = None
if cloud_on:
    from etl.fetch_and_build import run as etl_run
    try:
        odds_path, feat_path = etl_run(today)
        st.success(f"Fetched DraftKings board for {today}.")
        odds_file = pd.read_csv(odds_path)
        feat_file = pd.read_csv(feat_path)
    except Exception as e:
        st.error(f"Cloud fetch failed: {e}")

c1,c2 = st.columns(2)
with c1:
    up1 = st.file_uploader("Upload DK Markets CSV", type=["csv"], key="dk_csv")
    if up1 is not None:
        odds_file = pd.read_csv(up1)
with c2:
    up2 = st.file_uploader("Upload Features CSV (optional, enables true projections)", type=["csv"], key="feat_csv")
    if up2 is not None:
        feat_file = pd.read_csv(up2)

if odds_file is None:
    st.info("Upload a DK CSV to begin (sample in repo).")
    st.stop()
if feat_file is None:
    feat_file = pd.DataFrame({"player_id": odds_file["player_id"].dropna().unique()})

# ---------- Build + Filters ----------
pool_raw = build_pool(odds_file, feat_file)
pool_raw = add_missing(pool_raw, ["park_k_factor","opp_k_rate","opp_bb_rate","mu_ks","mu_bb","expected_ip","last5_pc_mean","days_rest"], np.nan)
pool_raw = dedupe_legs(pool_raw)

st.divider()
st.subheader("Filters")

f1,f2,f3,f4 = st.columns([1,1,1,1])
with f1:
    # Games with Select All
    all_games = sorted(pool_raw["game_id"].unique().tolist())
    gmap = {g: matchup(g) for g in all_games}
    select_all_games = st.checkbox("Select all games", value=True)
    games_sel = all_games if select_all_games else st.multiselect("Games", options=all_games, default=all_games, format_func=lambda k: gmap[k])
with f2:
    # Teams with Select All
    teams = sorted([t for t in pool_raw["team_abbr"].unique().tolist() if t])
    if not teams:
        teams_sel = []
    else:
        select_all_teams = st.checkbox("Select all teams", value=True)
        teams_sel = teams if select_all_teams else st.multiselect("Teams", options=teams, default=teams)
with f3:
    cats_all = sorted(pool_raw["category"].unique().tolist())
    cats = st.multiselect("Categories", options=cats_all, default=cats_all)
with f4:
    odds_min, odds_max = st.slider("American odds range", min_value=-700, max_value=700, value=(-700, 700), step=25)

mask = (pool_raw["game_id"].isin(games_sel)) & \
       (pool_raw["category"].isin(cats)) & \
       (pool_raw["american_odds"].between(odds_min, odds_max))
if teams:
    mask &= pool_raw["team_abbr"].isin(teams_sel)
pool = pool_raw[mask].copy()

# Coverage helper
with st.expander("Coverage (debug)"):
    st.write("Games detected:", ", ".join(sorted({matchup(g) for g in pool_raw['game_id'].unique()})))
    st.write("Note: if a matchup is missing (e.g., OAK@TBR, SD@SF, LAD@LAA, ARI@TEX, BOS@HOU), it may not be on the DraftKings board via the API yet, or odds range filter hides it. Widen the odds range to include more.")

if pool.empty:
    st.warning("No legs after filters.")
    st.stop()

# ---------- Tabs ----------
tab1, tab2, tab3, tabSafety, tabOneTap, tab4, tab5 = st.tabs([
    "Candidates", "Top 20", "Parlay Presets", "Alt Line Safety Board", "One‑Tap Ticket", "ML Winners & Alt RL Locks", "Locks (≥85%)"
])

# Candidates
with tab1:
    st.markdown('<div class="block-title">Candidate Legs (post‑filter)</div>', unsafe_allow_html=True)
    show = pool[["date","description","category","american_odds","decimal_odds","q_model","p_market","edge","ev","game_code","team_abbr"]].copy()
    show.rename(columns={"american_odds":"Odds","decimal_odds":"Dec","q_model":"q","p_market":"Market","edge":"Edge","ev":"EV","game_code":"Game","team_abbr":"Team"}, inplace=True)
    st.dataframe(show.sort_values(["q","Edge","Dec"], ascending=[False,False,False]),
                 use_container_width=True, hide_index=True)

# Top 20
with tab2:
    st.markdown('<div class="block-title">Top 20 Bets (≥60% hit, odds ≥ −350)</div>', unsafe_allow_html=True)
    top = pool[(pool["q_model"] >= 0.60) & (pool["american_odds"] >= -350)].copy()
    top = top.sort_values(["ev","q_model","decimal_odds"], ascending=[False,False,False]).head(20)
    if top.empty:
        st.info("No legs meet thresholds.")
    else:
        for i,(_,r) in enumerate(top.iterrows(), start=1):
            st.markdown(f"**{i}. {r['description']}**")
            st.markdown(f"- **Odds** {int(r['american_odds'])} · **Model q** {r['q_model']:.1%} · **Market** {r['p_market']:.1%} · **Edge** {r['edge']:+.1%} · **EV** {r['ev']:+.2f}")
            st.caption(compact_note(r))

# Parlay Presets
with tab3:
    st.markdown('<div class="block-title">Parlay Presets (4 · 5 · 6 · 8 legs)</div>', unsafe_allow_html=True)
    presets = build_presets(pool, legs_list=(4,5,6,8), min_parlay_am="+600", odds_min=odds_min, odds_max=odds_max)
    for legs in (4,5,6,8):
        st.markdown(f"#### {legs}‑Leg")
        cols = st.columns(3)
        for i, tier in enumerate(["Low","Medium","High"]):
            pr = presets.get((legs,tier), {"legs": [], "decimal": 1.0, "q_est": 0.0, "meets_min": False})
            with cols[i]:
                parlay_card(tier, pr["legs"], pr["decimal"], pr["q_est"], pr["meets_min"])

# Alt Line Safety Board
with tabSafety:
    st.markdown('<div class="block-title">Alt Line Safety Board</div>', unsafe_allow_html=True)
    s1,s2 = st.columns(2)
    with s1:
        q_thr = st.slider("Min model q", min_value=0.55, max_value=0.95, value=0.70, step=0.01)
    with s2:
        max_rows = st.slider("Rows", 10, 100, 40, 5)
    alt = pool[(pool["q_model"] >= q_thr) & (pool["american_odds"].between(odds_min, odds_max))].copy()
    # prefer alt-ish props & safe sides
    alt = alt[alt["category"].isin(["Pitcher Ks","Pitcher Outs","Pitcher BB","Alt Run Line","Run Line","Moneyline"])]
    alt = alt.sort_values(["q_model","edge","decimal_odds"], ascending=[False,False,False]).head(max_rows)
    show = alt[["description","american_odds","q_model","p_market","edge","ev","game_code","team_abbr"]].copy()
    show.rename(columns={"american_odds":"Odds","q_model":"q","p_market":"Market","edge":"Edge","ev":"EV","game_code":"Game","team_abbr":"Team"}, inplace=True)
    st.dataframe(show, use_container_width=True, hide_index=True)

# One‑Tap Ticket
with tabOneTap:
    st.markdown('<div class="block-title">One‑Tap Ticket</div>', unsafe_allow_html=True)
    cA,cB,cC = st.columns(3)
    with cA:
        legs = st.selectbox("Legs", [4,5,6,8], index=0)
    with cB:
        risk = st.selectbox("Risk", ["Low","Medium","High"], index=0, help="Low = safer; High = more value.")
    with cC:
        st.caption("Uses global **American odds range** above.")
    one = build_one_tap(pool, legs=legs, risk=risk, odds_min=odds_min, odds_max=odds_max)
    parlay_card(f"{legs}-Leg {risk}", one.get("legs", []), one.get("decimal", 1.0), one.get("q_est", 0.0), one.get("meets_min", False))

# ML Winners & Alt RL Locks
with tab4:
    st.markdown('<div class="block-title">ML Winners & Alt RL Locks</div>', unsafe_allow_html=True)
    ml = pool[pool["market_type"] == "MONEYLINE"].copy()
    if ml.empty:
        st.info("No moneyline markets in current pool.")
    else:
        ml_best = ml.sort_values(["q_model","ev"], ascending=[False, False]).drop_duplicates(subset=["game_id"], keep="first")
        ml_best["ML Lock?"] = (ml_best["q_model"] >= 0.65)
        show = ml_best[["description","american_odds","decimal_odds","q_model","p_market","edge","ev","ML Lock?","game_code"]]
        show.rename(columns={"american_odds":"Odds","decimal_odds":"Dec","q_model":"q","p_market":"Market","edge":"Edge","ev":"EV","game_code":"Game"}, inplace=True)
        st.dataframe(show.sort_values(["ML Lock?","q","EV"], ascending=[False,False,False]), use_container_width=True, hide_index=True)

    arl = pool[pool["market_type"].isin(["ALT_RUN_LINE","RUN_LINE"])].copy()
    st.markdown("**Alt Run Line Locks (q ≥ 70%)**")
    if arl.empty:
        st.info("No run line markets found.")
    else:
        arl_locks = arl[arl["q_model"] >= 0.70]
        show2 = arl_locks[["description","american_odds","decimal_odds","q_model","edge","ev","game_code"]]
        show2.rename(columns={"american_odds":"Odds","decimal_odds":"Dec","q_model":"q","edge":"Edge","ev":"EV","game_code":"Game"}, inplace=True)
        st.dataframe(show2.sort_values(["q","EV"], ascending=[False,False]), use_container_width=True, hide_index=True)

# Locks
with tab5:
    st.markdown('<div class="block-title">Locks (q ≥ 85%)</div>', unsafe_allow_html=True)
    locks = pool[pool["q_model"] >= 0.85].copy()
    if locks.empty:
        st.info("No 85%+ legs at the moment.")
    else:
        show = locks[["description","american_odds","q_model","ev","category","game_code"]]
        show.rename(columns={"american_odds":"Odds","q_model":"q","ev":"EV","game_code":"Game"}, inplace=True)
        st.dataframe(show.sort_values(["q","EV"], ascending=[False,False]),
                     use_container_width=True, hide_index=True)

st.caption("Compact labels: `A. Gray (STL) O7 Ks`. If a matchup seems missing, widen the odds range or verify it’s on the DK board.")