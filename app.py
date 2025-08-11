# app.py  -- MLB Parlay Picker (Clean UI)
import os, io, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

from parlay.odds import compute_vig_free_probs, american_to_decimal
from parlay.builder import build_presets
from parlay.describe import describe_row, matchup, compact_player
from parlay.ui import inject_css, parlay_card

# ----------------- Page + CSS -----------------
st.set_page_config(page_title="MLB Parlay Picker -- MVP", layout="wide", page_icon="⚾")
inject_css()

# ----------------- Utils ----------------------
def add_pct_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "q_model" in d: d["Model q %"] = (d["q_model"] * 100).round(1)
    if "p_market" in d: d["Market %"] = (d["p_market"] * 100).round(1)
    if "edge" in d: d["Edge %"] = (d["edge"] * 100).round(1)
    return d

def ensure_cols(df: pd.DataFrame, cols: list, fill=0.0) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        if c not in d.columns: d[c] = fill
    return d

def dedupe_legs(df: pd.DataFrame) -> pd.DataFrame:
    # remove exact duplicate legs (same player_id/market/line/side)
    cols = ["player_id","market_type","side","alt_line","american_odds","game_id"]
    exist = [c for c in cols if c in df.columns]
    if not exist: return df
    return df.drop_duplicates(subset=exist)

def build_pool(dk_df: pd.DataFrame, feat_df: pd.DataFrame) -> pd.DataFrame:
    df = dk_df.copy()

    # Decimal odds + market probabilities (vig-free within each event/market if available)
    if "decimal_odds" not in df.columns:
        df["decimal_odds"] = df["american_odds"].apply(lambda x: american_to_decimal(int(x)))
    if "p_market" not in df.columns:
        # coarse: convert American to implied p without vig; if you already compute by market, keep yours
        dec = df["decimal_odds"].replace(0, np.nan)
        df["p_market"] = 1.0 / dec
        df["p_market"] = df["p_market"].fillna(0.5).clip(0.01, 0.99)

    # Merge features → compute q_model
    q = feat_df.copy()
    df = df.merge(q, on="player_id", how="left")

    # If you supplied projections column 'q_proj', use it; else blend market with gentle shrink
    if "q_proj" in df.columns:
        df["q_model"] = df["q_proj"].clip(0.01, 0.99)
    else:
        df["q_model"] = (0.6*df["p_market"] + 0.4*0.50).clip(0.01, 0.99)

    # Edge & EV
    df["edge"] = (df["q_model"] - df["p_market"]).fillna(0.0)
    df["ev"] = (df["q_model"] * df["decimal_odds"] - 1.0).fillna(0.0)

    # Category and compact description
    def cat(r):
        mt = r["market_type"]
        return {"PITCHER_KS":"Pitcher Ks","PITCHER_OUTS":"Pitcher Outs","PITCHER_WALKS":"Pitcher BB",
                "PITCHER_WIN":"Pitcher Win","MONEYLINE":"Moneyline","RUN_LINE":"Run Line","ALT_RUN_LINE":"Alt Run Line"}.get(mt, mt)
    df["category"] = df.apply(cat, axis=1)
    df["description"] = df.apply(describe_row, axis=1)

    # Keep sane types
    df["american_odds"] = df["american_odds"].astype(int)
    return df

def compact_note(r: pd.Series) -> str:
    notes = []
    if r.get("market_type") == "PITCHER_OUTS" and pd.notna(r.get("expected_ip")):
        ip = float(r["expected_ip"]); notes.append(f"Exp IP: {ip:.1f} (O:{int(round(ip*3))}) vs Ln {r.get('alt_line')}")
    if r.get("market_type") == "PITCHER_KS" and pd.notna(r.get("mu_ks")):
        notes.append(f"μK {float(r['mu_ks']):.2f} vs Ln {r.get('alt_line')}")
    if r.get("market_type") == "PITCHER_WALKS" and pd.notna(r.get("mu_bb")):
        notes.append(f"μBB {float(r['mu_bb']):.2f} vs Ln {r.get('alt_line')}")
    if pd.notna(r.get("park_k_factor")):
        notes.append(f"ParkK {float(r['park_k_factor']):.2f}")
    if pd.notna(r.get("opp_k_rate")):
        notes.append(f"OppK {float(r['opp_k_rate']):.0%}")
    if pd.notna(r.get("opp_bb_rate")):
        notes.append(f"OppBB {float(r['opp_bb_rate']):.0%}")
    if pd.notna(r.get("last5_pc_mean")):
        notes.append(f"L5PC {float(r['last5_pc_mean']):.0f}")
    if pd.notna(r.get("days_rest")):
        notes.append(f"Rest {int(float(r['days_rest']))}d")
    return " • ".join(notes) if notes else "--"

# ----------------- Data entry (Cloud mode + uploads) -----------------
st.title("MLB Parlay Picker -- MVP")
st.caption("DraftKings + Alternate Lines | Cross‑game | Clean UI | Model uses market+features. Optional true projections via features CSV.")

with st.expander("How to use (tap to expand)"):
    st.markdown("""
1) Toggle **Cloud Mode** to auto-fetch today’s DraftKings board & props.  
2) (Optional) Upload a **Features CSV** to enable true projections (`q_proj`).  
3) Use **Filters** then review **Top 20** or **Parlay Presets** (card views).  
4) Long‑press to copy a pick line; odds shown as American + model probability.  
""")

# Cloud Mode controls
cloud_on = st.toggle("Enable Cloud Mode (auto‑fetch today)", value=False)
today = dt.date.today().strftime("%Y-%m-%d")

odds_file = None
feat_file = None

if cloud_on:
    from etl.fetch_and_build import run as etl_run
    try:
        odds_path, feat_path = etl_run(today)
        st.success(f"Fetched DraftKings board for {today}.")
        odds_file = pd.read_csv(odds_path)
        feat_file = pd.read_csv(feat_path)
    except Exception as e:
        st.error(f"Cloud fetch failed: {e}")

col_up1, col_up2 = st.columns(2)
with col_up1:
    dk_csv = st.file_uploader("Upload DK Markets CSV", type=["csv"], key="dk_csv")
    if dk_csv is not None:
        odds_file = pd.read_csv(dk_csv)
with col_up2:
    f_csv = st.file_uploader("Upload Features CSV (optional, enables true projections)", type=["csv"], key="feat_csv")
    if f_csv is not None:
        feat_file = pd.read_csv(f_csv)

if odds_file is None:
    st.info("Upload a DK CSV to begin. See `sample_data/sample_dk_markets.csv` in the repo.")
    st.stop()

if feat_file is None:
    # bare minimum features so app runs; values will be NaN where unknown
    feat_file = pd.DataFrame({"player_id": odds_file["player_id"].dropna().unique()})

# ----------------- Build pool + global filters -----------------
pool_raw = build_pool(odds_file, feat_file)
pool_raw = ensure_cols(pool_raw, ["park_k_factor","opp_k_rate","opp_bb_rate","mu_ks","mu_bb","expected_ip","last5_pc_mean","days_rest"], np.nan)
pool_raw = dedupe_legs(pool_raw)

st.divider()
st.subheader("Filters")
c1,c2,c3 = st.columns(3)
with c1:
    min_american = st.slider("Minimum Favorite Price (<=)", min_value=-700, max_value=+700, value=-350, step=25,
                             help="Filter legs by American odds (e.g., show only ≤ -350 favorites).")
with c2:
    cats = st.multiselect("Categories", options=sorted(pool_raw["category"].unique().tolist()),
                          default=sorted(pool_raw["category"].unique().tolist()))
with c3:
    # show compact game codes
    gmap = {g: matchup(g) for g in sorted(pool_raw["game_id"].unique())}
    games = st.multiselect("Games", options=list(gmap.keys()), format_func=lambda k: gmap[k], default=list(gmap.keys()))

pool = pool_raw[(pool_raw["american_odds"] <= min_american) &
                (pool_raw["category"].isin(cats)) &
                (pool_raw["game_id"].isin(games))].copy()

if pool.empty:
    st.warning("No legs after filters.")
    st.stop()

# ----------------- Tabs -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Candidates", "Top 20", "Parlay Presets", "ML Winners & Alt RL Locks", "Locks (≥85%)"])

# ---------- Candidates (clean table) ----------
with tab1:
    st.markdown('<div class="block-title">Candidate Legs (post‑filter)</div>', unsafe_allow_html=True)
    show_cols = ["date","description","category","american_odds","decimal_odds","q_model","p_market","edge","ev"]
    disp = pool[show_cols].copy()
    disp.rename(columns={"american_odds":"Odds","decimal_odds":"Dec","q_model":"q","p_market":"Market","edge":"Edge","ev":"EV"}, inplace=True)
    st.dataframe(disp.sort_values(["q","Edge","Dec"], ascending=[False,False,False]),
                 use_container_width=True, hide_index=True)

# ---------- Top 20 (compact list) ----------
with tab2:
    st.markdown('<div class="block-title">Top 20 Bets (≥60% hit, odds ≥ −350, deduped, ranked by EV)</div>', unsafe_allow_html=True)
    top = pool[(pool["q_model"] >= 0.60) & (pool["american_odds"] >= -350)].copy()
    top = top.sort_values(["ev","q_model","decimal_odds"], ascending=[False,False,False]).head(20)
    if top.empty:
        st.info("No legs meet the Top‑20 thresholds right now.")
    else:
        for i, (_,r) in enumerate(top.iterrows(), start=1):
            st.markdown(f"**{i}. {r['description']}**")
            st.markdown(f"- **Odds**: {int(r['american_odds'])} | **Model q**: {r['q_model']:.1%} | **Market**: {r['p_market']:.1%} | **Edge**: {r['edge']:+.1%} | **EV**: {r['ev']:+.2f}")
            st.caption(compact_note(r))

# ---------- Parlay Presets (card view, 4/5/6/8 × Low/Medium/High) ----------
with tab3:
    st.markdown('<div class="block-title">Parlay Presets</div>', unsafe_allow_html=True)
    presets = build_presets(pool, legs_list=(4,5,6,8), min_parlay_am="+600")
    # render rows by legs
    for legs in (4,5,6,8):
        st.markdown(f"#### {legs}‑Leg")
        cols = st.columns(3)
        for i, tier in enumerate(["Low","Medium","High"]):
            pr = presets.get((legs,tier), {"legs": [], "decimal": 1.0, "q_est": 0.0, "meets_min": False})
            with cols[i]:
                title = f"{tier}"
                parlay_card(title, pr["legs"], pr["decimal"], pr["q_est"], pr["meets_min"])

# ---------- ML Winners & Alt RL Locks ----------
with tab4:
    st.markdown('<div class="block-title">ML Winners & Alt RL Locks</div>', unsafe_allow_html=True)
    ml = pool[pool["market_type"] == "MONEYLINE"].copy()
    if ml.empty:
        st.info("No moneyline markets in the current pool.")
    else:
        ml_best = ml.sort_values(["q_model","ev"], ascending=[False, False]).drop_duplicates(subset=["game_id"], keep="first")
        ml_best["ML Lock?"] = (ml_best["q_model"] >= 0.65)
        ml_show = ml_best[["description","american_odds","decimal_odds","q_model","p_market","edge","ev","ML Lock?"]].copy()
        ml_show.rename(columns={"american_odds":"Odds","decimal_odds":"Dec","q_model":"q","p_market":"Market","edge":"Edge","ev":"EV"}, inplace=True)
        st.dataframe(ml_show.sort_values(["ML Lock?","q","EV"], ascending=[False,False,False]),
                     use_container_width=True, hide_index=True)

    arl = pool[pool["market_type"].isin(["ALT_RUN_LINE","RUN_LINE"])].copy()
    if not arl.empty:
        st.markdown("**Alt Run Line Locks (q ≥ 70%)**")
        arl_locks = arl[arl["q_model"] >= 0.70].copy()
        show = arl_locks[["description","american_odds","decimal_odds","q_model","p_market","edge","ev"]]
        show.rename(columns={"american_odds":"Odds","decimal_odds":"Dec","q_model":"q","p_market":"Market","edge":"Edge","ev":"EV"}, inplace=True)
        st.dataframe(show.sort_values(["q","EV"], ascending=[False,False]),
                     use_container_width=True, hide_index=True)
    else:
        st.info("No run line markets found.")

# ---------- Locks (≥85%) ----------
with tab5:
    st.markdown('<div class="block-title">Locks (q ≥ 85%)</div>', unsafe_allow_html=True)
    locks = pool[pool["q_model"] >= 0.85].copy()
    if locks.empty:
        st.info("No 85%+ legs at the moment.")
    else:
        show = locks[["description","american_odds","q_model","ev","category"]].copy()
        show.rename(columns={"american_odds":"Odds","q_model":"q","ev":"EV"}, inplace=True)
        st.dataframe(show.sort_values(["q","EV"], ascending=[False,False]),
                     use_container_width=True, hide_index=True)

st.caption("Tip: cards show compact labels like `A. Gray (STL) O7 Ks`. Top‑20 notes are concise and matchup‑aware.")