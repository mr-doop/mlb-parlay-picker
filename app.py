import streamlit as st, pandas as pd, numpy as np, os, sys, subprocess
from parlay.odds import compute_vig_free_probs
from parlay.builder import build_parlay_greedy
from parlay.describe import describe_row
st.set_page_config(page_title="MLB Parlay Picker — MVP", layout="wide")

try:
    _ = st.secrets["ODDS_API_KEY"]
    st.sidebar.success("Secret detected ✔︎")
except Exception:
    st.sidebar.error("No ODDS_API_KEY in Secrets ✖︎")

st.title("MLB Parlay Picker — MVP")
st.caption("DraftKings + Alternate Lines | Cross-game only (MVP) | Uses vig-free market probabilities as q")

with st.expander("How to use (click to expand)", expanded=False):
    st.markdown("""
    1. **Cloud Mode:** toggle below to auto-fetch today’s board (needs ODDS_API_KEY in Secrets).  
    2. Or upload a DK CSV (sample in repo).  
    3. (Optional) Upload features CSV for true projections.  
    4. Build **Safety** and **Value** parlays.
    """)

# Cloud Mode
st.subheader("Cloud Mode (Auto-Fetch)")
cloud_on = st.toggle("Enable Cloud Mode (auto-fetch today)")
if cloud_on:
    try:
        fetch_date = pd.to_datetime('today').strftime('%Y-%m-%d')
        cmd = [sys.executable, "etl/fetch_and_build.py"]
        # run via module call
        import importlib.util
        spec = importlib.util.spec_from_file_location("fab","etl/fetch_and_build.py")
        fab = importlib.util.module_from_spec(spec); spec.loader.exec_module(fab)
        odds_csv, feat_csv = fab.run(fetch_date)
        st.success(f"Fetched DraftKings board for {fetch_date}.")
        # Download buttons
        if os.path.exists(odds_csv):
            with open(odds_csv,"rb") as f:
                st.download_button("⬇️ Download DK CSV", f, file_name=odds_csv, mime="text/csv")
        if os.path.exists(feat_csv):
            with open(feat_csv,"rb") as f:
                st.download_button("⬇️ Download Features CSV", f, file_name=feat_csv, mime="text/csv")
    except Exception as e:
        st.error(f"Cloud fetch failed: {e} — Make sure ODDS_API_KEY is set in Streamlit Secrets.")

# Upload odds CSV
uploaded = st.file_uploader("Upload DK Markets CSV", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to begin. See sample_data/sample_dk_markets.csv in the repo.")
    st.stop()

df = pd.read_csv(uploaded)
required_cols = {"date","game_id","market_type","side","team","player_id","player_name","alt_line","american_odds"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Missing columns in CSV: {missing}")
    st.stop()

df['alt_line'] = pd.to_numeric(df['alt_line'], errors='coerce')
df['american_odds'] = pd.to_numeric(df['american_odds'], errors='coerce')

dfq = compute_vig_free_probs(df)
dfq['q'] = dfq['vigfree_p']
dfq['description'] = dfq.apply(describe_row, axis=1)

st.subheader("Candidate Legs (vig-free)")
st.dataframe(dfq[["date","game_id","market_type","side","team","player_name","alt_line","american_odds","decimal_odds","q","description"]].sort_values(["q","decimal_odds"], ascending=[False, False]), use_container_width=True, height=420)

with st.sidebar:
    st.header("Parlay Settings")
    target_decimal_odds = st.slider("Target Parlay Decimal Odds", min_value=3.0, max_value=15.0, value=6.0, step=0.1)
    min_legs = st.slider("Min legs", 2, 8, 4, 1)
    max_legs = st.slider("Max legs", 3, 10, 7, 1)
    market_types_all = ["MONEYLINE","RUN_LINE","ALT_RUN_LINE","PITCHER_WIN","PITCHER_KS","PITCHER_OUTS","PITCHER_WALKS"]
    selected_types = st.multiselect("Market types to include", market_types_all, default=market_types_all)

filtered = dfq[dfq["market_type"].isin(selected_types)].copy()

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Build Safety Parlay")
    picks_safety = build_parlay_greedy(filtered, target_decimal_odds=target_decimal_odds, min_legs=min_legs, max_legs=max_legs, mode="SAFETY")
    if not picks_safety.empty:
        st.write(f"**Decimal Odds:** {picks_safety.attrs.get('total_decimal_odds', float('nan')):.3f}")
        st.write(f"**Est. Hit Prob (independence):** {picks_safety.attrs.get('est_hit_prob', float('nan')):.3f}")
        st.dataframe(picks_safety[["game_id","market_type","side","team","player_name","alt_line","american_odds","decimal_odds","q","description"]], use_container_width=True, height=320)
    else:
        st.info("No safety parlay built. Try relaxing filters or reducing target odds.")

with col2:
    st.markdown("### Build Value Parlay")
    picks_value = build_parlay_greedy(filtered, target_decimal_odds=target_decimal_odds, min_legs=min_legs, max_legs=max_legs, mode="VALUE")
    if not picks_value.empty:
        st.write(f"**Decimal Odds:** {picks_value.attrs.get('total_decimal_odds', float('nan')):.3f}")
        st.write(f"**Est. Hit Prob (independence):** {picks_value.attrs.get('est_hit_prob', float('nan')):.3f}")
        st.dataframe(picks_value[["game_id","market_type","side","team","player_name","alt_line","american_odds","decimal_odds","q","description"]], use_container_width=True, height=320)
    else:
        st.info("No value parlay built. Try relaxing filters or reducing target odds.")
