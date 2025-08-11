# app.py -- MLB Parlay Picker (mobile-friendly)
# --------------------------------------------
import os, sys, time, subprocess
import numpy as np
import pandas as pd
import streamlit as st

# local libs
from parlay.odds import compute_vig_free_probs, implied_prob_from_american
from parlay.builder import build_parlay_greedy
from parlay.describe import describe_row
from parlay.projections import apply_projections  # uses features file to compute q_proj

st.set_page_config(page_title="MLB Parlay Picker -- MVP", layout="wide")
st.title("MLB Parlay Picker -- MVP")
st.caption("DraftKings + Alternate Lines | Cross‑game only | Vig‑free market probabilities + optional true projections")

# --- show if secret exists (non-blocking)
try:
    _ = st.secrets["ODDS_API_KEY"]
    st.sidebar.success("Secret detected ✔︎")
except Exception:
    st.sidebar.info("No ODDS_API_KEY in Secrets. You can still upload CSVs manually.")

# ======================================================================
# Cloud Mode (auto-fetch)
# ======================================================================
st.subheader("Cloud Mode (Auto‑Fetch)")
cloud_on = st.toggle("Enable Cloud Mode (auto‑fetch today)")

if cloud_on:
    try:
        fetch_date = pd.to_datetime("today").strftime("%Y-%m-%d")
        # Import the fetcher directly (Option B: per‑event odds endpoint)
        import importlib.util
        spec = importlib.util.spec_from_file_location("fab", "etl/fetch_and_build.py")
        fab = importlib.util.module_from_spec(spec); spec.loader.exec_module(fab)
        odds_csv, feat_csv = fab.run(fetch_date)

        st.success(f"Fetched DraftKings board for {fetch_date}.")
        # Download buttons
        if os.path.exists(odds_csv):
            with open(odds_csv, "rb") as f:
                st.download_button("⬇️ Download DK CSV", f, file_name=odds_csv, mime="text/csv")
        if os.path.exists(feat_csv):
            with open(feat_csv, "rb") as f:
                st.download_button("⬇️ Download Features CSV", f, file_name=feat_csv, mime="text/csv")
    except Exception as e:
        st.error(f"Cloud fetch failed: {e}")
        st.caption("Tip: ensure your Secrets has ODDS_API_KEY = \"...\" (straight quotes).")

# ======================================================================
# Uploads
# ======================================================================
uploaded = st.file_uploader("Upload DK Markets CSV", type=["csv"])
features_up = st.file_uploader("Upload Features CSV (optional, enables true projections)", type=["csv"], key="feat")

if uploaded is None:
    st.info("Upload a DK CSV to begin. (A sample lives in sample_data/sample_dk_markets.csv)")
    st.stop()

# ======================================================================
# Read & normalize odds
# ======================================================================
df = pd.read_csv(uploaded)
req_cols = {"date","game_id","market_type","side","team","player_id","player_name","alt_line","american_odds"}
missing = req_cols - set(df.columns)
if missing:
    st.error(f"Missing columns in CSV: {missing}")
    st.stop()

df["american_odds"] = pd.to_numeric(df["american_odds"], errors="coerce")
df["alt_line"] = pd.to_numeric(df["alt_line"], errors="coerce")

dfq = compute_vig_free_probs(df)  # adds implied_p, decimal_odds, vigfree_p
dfq["description"] = dfq.apply(describe_row, axis=1)

# True projections (if features uploaded)
if features_up is not None:
    try:
        feats = pd.read_csv(features_up)
        dfq = apply_projections(dfq, feats)  # adds q_proj and helpful projection columns
        st.success("Applied true projections (q_proj).")
    except Exception as e:
        st.warning(f"Could not apply projections: {e}")

# Choose probability column for modeling
# Prefer projections (q_proj), else vig-free market prob (vigfree_p)
if "q_proj" in dfq.columns:
    dfq["q_model"] = dfq["q_proj"].fillna(dfq.get("vigfree_p"))
else:
    dfq["q_model"] = dfq.get("vigfree_p")

# Fallback if decimal not present for any reason
if "decimal_odds" not in dfq.columns or dfq["decimal_odds"].isna().any():
    def american_to_decimal(a):
        a = float(a)
        return 1.0 + (a/100.0 if a > 0 else 100.0/abs(a))
    dfq["decimal_odds"] = dfq["american_odds"].apply(american_to_decimal)

# Market (vig-free) prob and value metrics
dfq["p_market"] = dfq.get("vigfree_p").fillna(1.0/dfq["decimal_odds"].clip(lower=1.0000001))
dfq["edge"] = dfq["q_model"] - dfq["p_market"]
dfq["ev"]   = dfq["q_model"] * dfq["decimal_odds"] - 1.0

# Handy category tags for filtering
def market_category(mt: str) -> str:
    mt = str(mt)
    if mt == "MONEYLINE":
        return "Moneyline"
    if mt in {"RUN_LINE","ALT_RUN_LINE"}:
        return "Run Line"
    if mt == "PITCHER_KS":
        return "Pitcher Strikeouts"
    if mt == "PITCHER_OUTS":
        return "Pitcher Outs"
    if mt == "PITCHER_WALKS":
        return "Pitcher Walks"
    if mt == "PITCHER_WIN":
        return "Pitcher To Win"
    return mt.title()
dfq["category"] = dfq["market_type"].map(market_category)

# ======================================================================
# Sidebar: global filters
# ======================================================================
with st.sidebar:
    st.header("Filters & Parlay Settings")

    # Global minimum odds (≥), e.g., -500 means keep -500, -400, -300, ... +100 etc.
    min_american = st.slider("Minimum American odds to include (≥)", -700, 700, value=-500, step=25)

    # Minimum parlay decimal odds (7.0 = +600)
    min_parlay_decimal = st.slider("Minimum parlay decimal odds", 3.0, 25.0, value=7.0, step=0.1)

    # Fixed leg counts for presets (we will build for 4,5,6,8)
    min_legs = 4
    max_legs = st.slider("Max legs (for manual builds, if used)", 4, 20, 12, 1)

    # Category selector
    all_cats = ["Moneyline","Run Line","Pitcher Strikeouts","Pitcher Outs","Pitcher Walks","Pitcher To Win"]
    selected_cats = st.multiselect("Categories", all_cats, default=all_cats)

    # Quick presets
    st.markdown("**Quick filters**")
    pitcher_props_only = st.checkbox("Pitcher props only", value=False)
    moneyline_only = st.checkbox("Moneyline only", value=False)
    runline_only = st.checkbox("Run lines only", value=False)

    # Game filter / search
    games_all = sorted(dfq["game_id"].dropna().unique().tolist())
    games_selected = st.multiselect("Games", games_all, default=games_all)
    text_search = st.text_input("Search team/player contains (optional)", value="").strip()

# Apply global filters
pool = dfq.copy()
pool = pool[pool["american_odds"] >= min_american]
pool = pool[pool["category"].isin(selected_cats)]
pool = pool[pool["game_id"].isin(games_selected)]

if pitcher_props_only:
    pool = pool[pool["category"].str.startswith("Pitcher")]
if moneyline_only:
    pool = pool[pool["category"] == "Moneyline"]
if runline_only:
    pool = pool[pool["category"] == "Run Line"]

if text_search:
    s = text_search.lower()
    pool = pool[(pool["team"].fillna("").str.lower().str.contains(s)) |
                (pool["player_name"].fillna("").str.lower().str.contains(s)) |
                (pool["description"].fillna("").str.lower().str.contains(s))]

# ======================================================================
# Candidate table (post-filter)
# ======================================================================
st.subheader("Candidate Legs (post‑filter)")
cols_show = ["date","game_id","category","market_type","side","team","player_name","alt_line",
             "american_odds","decimal_odds","q_model","p_market","edge","ev","description"]
st.dataframe(
    pool[cols_show].sort_values(["q_model","edge","decimal_odds"], ascending=[False,False,False]),
    use_container_width=True, height=460
)

# ======================================================================
# Top 20 Bets -- ≥60% model hit rate, odds ≥ -350, ranked by EV (with rationale)
# ======================================================================
st.subheader("Top 20 Bets (≥60% hit, odds ≥ −350, ranked by EV)")

top = pool.copy()
# enforce odds >= -350 regardless of global slider (but if slider is stricter, it already applies)
min_top_odds = max(min_american, -350)
top = top[top["american_odds"] >= min_top_odds].copy()
top = top[top["q_model"] >= 0.60].copy()

# Rank by EV; tie-break by q then by payout
top["rank_score"] = top["ev"]
top = top.sort_values(["rank_score","q_model","decimal_odds"], ascending=[False,False,False]).head(20)

def rationale(r: pd.Series) -> str:
    bits = []
    bits.append(f"Model {r['q_model']:.1%} vs market {r['p_market']:.1%} → edge {r['edge']:+.1%}. "
                f"Price {int(r['american_odds'])} (dec {r['decimal_odds']:.2f}), EV {r['ev']:+.2f} per unit.")
    mt = str(r.get("market_type",""))
    if mt == "PITCHER_KS" and pd.notna(r.get("mu_ks", np.nan)):
        bits.append(f" Ks μ≈{r['mu_ks']:.2f} vs line {r.get('alt_line')}.")
    if mt == "PITCHER_OUTS" and pd.notna(r.get("expected_ip", np.nan)):
        bits.append(f" Expected IP≈{r['expected_ip']:.2f} (outs mean ≈ {r['expected_ip']*3:.1f}) vs line {r.get('alt_line')}.")
    if mt == "PITCHER_WALKS" and pd.notna(r.get("mu_bb", np.nan)):
        bits.append(f" Walks μ≈{r['mu_bb']:.2f} vs line {r.get('alt_line')}.")
    if mt in {"RUN_LINE","ALT_RUN_LINE"} and pd.notna(r.get("alt_line", np.nan)):
        bits.append(f" Run line {r['alt_line']} selected for safety/value.")
    if pd.notna(r.get("pitcher_k_rate", np.nan)):
        bits.append(f" K% {r['pitcher_k_rate']:.0%} vs opp K% {r.get('opp_k_rate', np.nan):.0%}.")
    if pd.notna(r.get("pitcher_bb_rate", np.nan)):
        bits.append(f" BB% {r['pitcher_bb_rate']:.0%} vs opp BB% {r.get('opp_bb_rate', np.nan):.0%}.")
    if pd.notna(r.get("leash_bias", np.nan)) and r["leash_bias"] > 0:
        bits.append(" Longer leash expected.")
    return " ".join(bits)

if top.empty:
    st.info("No legs meet the ≥60% + odds ≥ −350 threshold under the current filters. Loosen odds or upload features.")
else:
    for i, (_, row) in enumerate(top.iterrows(), start=1):
        st.markdown(f"**{i}. {row['description']}**  \n"
                    f"Odds: {int(row['american_odds'])} | Model q: {row['q_model']:.1%} | Market: {row['p_market']:.1%} "
                    f"| Edge: {row['edge']:+.1%} | EV: {row['ev']:+.2f}")
        st.caption(rationale(row))

# ======================================================================
# Parlay presets -- exact 4, 5, 6, 8 legs; Low / Medium / High
# ======================================================================
st.subheader("Parlay Presets (exact 4, 5, 6, 8 legs)")

# Builder expects column "q" → feed it model prob
pool_for_builder = pool.copy()
pool_for_builder["q"] = pool_for_builder["q_model"]

def build_fixed_parlay(subpool: pd.DataFrame, leg_count: int, mode: str, q_cut: float):
    """Pick exactly leg_count by using the greedy builder with a huge target."""
    p = subpool[subpool["q_model"] >= q_cut].copy()
    if p.empty:
        return None
    # Huge target so it stops at max_legs
    picks = build_parlay_greedy(p, target_decimal_odds=1e9, min_legs=leg_count, max_legs=leg_count, mode=mode)
    return picks if not picks.empty else None

def summarize_parlay(name: str, leg_count: int, picks: pd.DataFrame):
    if picks is None or picks.empty:
        return {"Legs": leg_count, "Type": name, "Decimal": np.nan, "Est Hit": np.nan, "Meets +600?": False, "Picks": "--"}
    dec = float(picks.attrs.get("total_decimal_odds", float("nan")))
    est = float(picks.attrs.get("est_hit_prob", float("nan")))
    meets = bool(dec >= min_parlay_decimal)
    picks_desc = " | ".join(picks["description"].tolist())
    return {"Legs": leg_count, "Type": name, "Decimal": round(dec,3), "Est Hit": round(est,3), "Meets +600?": meets, "Picks": picks_desc}

rows = []
risk_defs = [
    ("Low",    "SAFETY", 0.65),
    ("Medium", "SAFETY", 0.60),
    ("High",   "VALUE",  0.55),
]
for leg_count in [4,5,6,8]:
    for (label, mode, q_cut) in risk_defs:
        picks = build_fixed_parlay(pool_for_builder, leg_count, mode, q_cut)
        rows.append(summarize_parlay(label, leg_count, picks))

parlays_tbl = pd.DataFrame(rows).sort_values(["Legs","Type"])
st.dataframe(parlays_tbl, use_container_width=True)

# ======================================================================
# Locks -- ≥85% model probability
# ======================================================================
st.subheader("Locks (model ≥85% to hit)")
locks = pool[pool["q_model"] >= 0.85].copy().sort_values(["q_model","edge"], ascending=[False,False]).head(12)
if locks.empty:
    st.info("No 85%+ locks under current filters.")
else:
    for i, (_, row) in enumerate(locks.iterrows(), start=1):
        st.markdown(f"**{i}. {row['description']}** -- "
                    f"Odds {int(row['american_odds'])} | q={row['q_model']:.1%} | EV {row['ev']:+.2f}")
        # optional rationale reuse:
        # st.caption(rationale(row))

st.caption("Note: Model probabilities assume approximate independence between legs; real‑world correlation may reduce combined hit rates. This tool is for research/entertainment.")