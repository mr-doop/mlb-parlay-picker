import streamlit as st
import pandas as pd
import numpy as np

from parlay.odds import compute_vig_free_probs, american_to_decimal
from parlay.builder import build_parlay_greedy
from parlay.describe import describe_row

st.set_page_config(page_title="MLB Parlay Picker — MVP", layout="wide")

st.title("MLB Parlay Picker — MVP")
st.caption("DraftKings + Alternate Lines | Cross-game only (MVP) | Uses vig-free market probabilities as q")

with st.expander("How to use (click to expand)", expanded=False):
    st.markdown(
        """
        1. Upload your **DraftKings markets CSV** (sample provided in repo).  
        2. Choose target total odds (e.g., **+500** = **6.0** decimal).  
        3. Pick which market types to include.  
        4. Click **Build Safety** or **Build Value** to auto-construct a parlay.  
        5. Copy the parlay summary into DraftKings (manually).

        **Note:** This MVP uses *vig-free market probabilities* as the model probability `q`.
        """
    )

uploaded = st.file_uploader("Upload DK Markets CSV", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to begin. See `sample_data/sample_dk_markets.csv` in the repo.")
    st.stop()

df = pd.read_csv(uploaded)
required_cols = {"date","game_id","market_type","side","team","player_id","player_name","alt_line","american_odds"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Missing columns in CSV: {missing}")
    st.stop()

# Clean types
df["alt_line"] = pd.to_numeric(df["alt_line"], errors="coerce")
df["american_odds"] = pd.to_numeric(df["american_odds"], errors="coerce")

# Compute vig-free probs and decimal odds
dfq = compute_vig_free_probs(df)

# For MVP, set q = vigfree_p (market-implied)
dfq["q"] = dfq["vigfree_p"]

# Optional: upload features for true projections
st.subheader("Optional: Upload Pitcher Feature CSV for True Projections")
st.caption("If provided, we'll compute model probabilities (q_proj) for Ks/Outs/Walks/Win. Otherwise we use vig-free market probs (q).")
feat_file = st.file_uploader("Upload Features CSV (see sample_data/sample_features.csv)", type=["csv"], key="feat_upl")

use_projections = False
features_df = None
if feat_file is not None:
    try:
        features_df = pd.read_csv(feat_file)
        required_feat_cols = {"player_id","pitcher_k_rate","pitcher_bb_rate","opp_k_rate","opp_bb_rate",
                              "last5_pitch_ct_mean","days_rest","leash_bias","favorite_flag",
                              "bullpen_freshness","park_k_factor","ump_k_bias"}
        missing_feat = required_feat_cols - set(features_df.columns)
        if missing_feat:
            st.warning(f"Missing feature columns: {missing_feat}. Projections disabled.")
        else:
            use_projections = True
            from parlay.projections import apply_projections
            dfq = apply_projections(dfq, features_df)
            # prefer q_proj when available; fallback to vigfree_p
            dfq["q"] = dfq.get("q_proj", dfq["q"]).fillna(dfq["q"])
            st.success("Applied projection-based probabilities (q).")
    except Exception as e:
        st.error(f"Failed to apply projections: {e}")

# Filters
with st.sidebar:
    st.header("Filters")
    target_decimal_odds = st.slider("Target Parlay Decimal Odds", min_value=3.0, max_value=15.0, value=6.0, step=0.1)
    min_legs = st.slider("Min legs", 2, 8, 4, 1)
    max_legs = st.slider("Max legs", 3, 10, 7, 1)

    market_types_all = ["MONEYLINE","RUN_LINE","ALT_RUN_LINE","PITCHER_WIN","PITCHER_KS","PITCHER_OUTS","PITCHER_WALKS"]
    selected_types = st.multiselect("Market types to include", market_types_all, default=market_types_all)

filtered = dfq[dfq["market_type"].isin(selected_types)].copy()

# Describe rows for UI
filtered["description"] = filtered.apply(describe_row, axis=1)

# Show table with sorting

st.divider()
st.subheader("One-click Fetch (optional)")
st.caption("Uses The Odds API with your ODDS_API_KEY (Streamlit secrets or .env) to pull DK odds and seed a features CSV template.")
colA, colB = st.columns([1,1])
with colA:
    fetch_date = st.text_input("Date (YYYY-MM-DD)", value=pd.to_datetime('today').strftime('%Y-%m-%d'))
with colB:
    run_fetch = st.button("Fetch DK Odds & Seed Features", type="primary")

if run_fetch:
    import subprocess, sys, os
    try:
        # Run the ETL script
        cmd = [sys.executable, "etl/fetch_and_build.py", "--date", fetch_date]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        st.success("Fetch complete.")
        st.text(res.stdout)
        st.error(res.stderr if res.stderr else "")
    except Exception as e:
        st.error(f"Fetch failed: {e}")



# ---- Cloud Mode: If ODDS_API_KEY exists in secrets and user toggles, auto-fetch today's slate ----
st.divider()
st.subheader("Cloud Mode (Auto-Fetch)")
st.caption("If your ODDS_API_KEY is set in Streamlit Secrets, I can auto-fetch today’s DK board and seed features.")
cloud_on = st.toggle("Enable Cloud Mode (auto-fetch today)")
if cloud_on:
    try:
        import os, sys, subprocess
        has_key = bool(st.secrets.get("ODDS_API_KEY", ""))
        if not has_key:
            st.warning("No ODDS_API_KEY in secrets. Add it under App settings → Secrets.")
        else:
            fetch_date_cloud = pd.to_datetime('today').strftime('%Y-%m-%d')
            cmd = [sys.executable, "etl/fetch_and_build.py", "--date", fetch_date_cloud]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            st.success("Fetched today’s DK odds and seeded features.")
            st.text(res.stdout)
            # Auto-load most recent files for convenience
            dk_auto = f"dk_markets_{fetch_date_cloud}.csv"
            feat_auto = f"features_{fetch_date_cloud}.csv"
            st.info(f"Now upload {dk_auto} below (and {feat_auto} for projections).")
    except Exception as e:
        st.error(f"Cloud fetch failed: {e}")


st.subheader("Candidate Legs (vig-free)")
st.dataframe(filtered[["date","game_id","market_type","side","team","player_name","alt_line","american_odds","decimal_odds","q","description"]]
             .sort_values(["q","decimal_odds"], ascending=[False, False]), use_container_width=True, height=420)

# Build parlays
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Build Safety Parlay")
    picks_safety = build_parlay_greedy(filtered, target_decimal_odds=target_decimal_odds,
                                       min_legs=min_legs, max_legs=max_legs, mode="SAFETY")
    if not picks_safety.empty:
        st.write(f"**Decimal Odds:** {picks_safety.attrs.get('total_decimal_odds', float('nan')):.3f}")
        st.write(f"**Est. Hit Prob (independence):** {picks_safety.attrs.get('est_hit_prob', float('nan')):.3f}")
        st.dataframe(picks_safety[["game_id","market_type","side","team","player_name","alt_line","american_odds","decimal_odds","q","description"]],
                     use_container_width=True, height=320)
        # Copy text
        summary = "\n".join([f"- {r['description']} | {int(r['american_odds'])} | q={r['q']:.2f}"
                              for _, r in picks_safety.iterrows()])
        st.text_area("Copy summary", summary, height=150)
    else:
        st.info("No safety parlay built. Try relaxing filters or reducing target odds.")

with col2:
    st.markdown("### Build Value Parlay")
    picks_value = build_parlay_greedy(filtered, target_decimal_odds=target_decimal_odds,
                                      min_legs=min_legs, max_legs=max_legs, mode="VALUE")
    if not picks_value.empty:
        st.write(f"**Decimal Odds:** {picks_value.attrs.get('total_decimal_odds', float('nan')):.3f}")
        st.write(f"**Est. Hit Prob (independence):** {picks_value.attrs.get('est_hit_prob', float('nan')):.3f}")
        st.dataframe(picks_value[["game_id","market_type","side","team","player_name","alt_line","american_odds","decimal_odds","q","description"]],
                     use_container_width=True, height=320)
        summary = "\n".join([f"- {r['description']} | {int(r['american_odds'])} | q={r['q']:.2f}"
                              for _, r in picks_value.iterrows()])
        st.text_area("Copy summary", summary, height=150, key="value_summary")
    else:
        st.info("No value parlay built. Try relaxing filters or reducing target odds.")
