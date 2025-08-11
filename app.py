# app.py -- MLB Parlay Picker (executive UI)
# -------------------------------------------------
import os, sys, time, subprocess
import numpy as np
import pandas as pd
import streamlit as st

# local libs
from parlay.odds import compute_vig_free_probs, implied_prob_from_american
from parlay.builder import build_parlay_greedy
from parlay.describe import describe_row
from parlay.projections import apply_projections  # enables q_proj when features uploaded

st.set_page_config(page_title="MLB Parlay Picker -- MVP", layout="wide")
st.markdown("""
<style>
/* subtle, readable tables */
div[data-testid="stDataFrame"] td { font-size: 0.95rem; line-height: 1.3; }
div[data-testid="stDataFrame"] th { font-size: 0.90rem; background: #f6f7fb; }
.block-title { font-weight: 700; font-size: 1.05rem; margin-top: .5rem; }
.small-note { color: #6b7280; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

st.title("MLB Parlay Picker -- MVP")
st.caption("DraftKings + Alternate Lines | Cross‑game only | Vig‑free market probabilities + optional true projections")

# --- quick hint if secret exists
try:
    _ = st.secrets["ODDS_API_KEY"]
    st.sidebar.success("Secret detected ✔︎")
except Exception:
    st.sidebar.info("No ODDS_API_KEY in Secrets. Use Cloud Mode only if key is set, otherwise upload CSVs manually.")

# ======================================================
# Cloud Mode (auto‑fetch via your Odds API key)
# ======================================================
st.subheader("Cloud Mode (Auto‑Fetch)")
cloud_on = st.toggle("Enable Cloud Mode (auto‑fetch today)")

if cloud_on:
    try:
        fetch_date = pd.to_datetime("today").strftime("%Y-%m-%d")
        import importlib.util
        spec = importlib.util.spec_from_file_location("fab", "etl/fetch_and_build.py")
        fab = importlib.util.module_from_spec(spec); spec.loader.exec_module(fab)
        odds_csv, feat_csv = fab.run(fetch_date)

        st.success(f"Fetched DraftKings board for {fetch_date}.")
        if os.path.exists(odds_csv):
            with open(odds_csv, "rb") as f:
                st.download_button("⬇️ Download DK CSV", f, file_name=odds_csv, mime="text/csv")
        if os.path.exists(feat_csv):
            with open(feat_csv, "rb") as f:
                st.download_button("⬇️ Download Features CSV", f, file_name=feat_csv, mime="text/csv")
    except Exception as e:
        st.error(f"Cloud fetch failed: {e}")
        st.caption("Tip: ensure Secrets has ODDS_API_KEY = \"...\" (straight quotes).")

# ======================================================
# Uploads
# ======================================================
uploaded = st.file_uploader("Upload DK Markets CSV", type=["csv"])
features_up = st.file_uploader("Upload Features CSV (optional, enables true projections)", type=["csv"], key="feat")

if uploaded is None:
    st.info("Upload a DK CSV to begin. (Sample: sample_data/sample_dk_markets.csv)")
    st.stop()

# ======================================================
# Read & normalize odds
# ======================================================
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

# True projections (if features uploaded) -> q_proj
if features_up is not None:
    try:
        feats = pd.read_csv(features_up)
        dfq = apply_projections(dfq, feats)
        st.success("Applied true projections (q_proj).")
    except Exception as e:
        st.warning(f"Could not apply projections: {e}")

# Choose model probability: prefer projections, else vig‑free p
if "q_proj" in dfq.columns:
    dfq["q_model"] = dfq["q_proj"].fillna(dfq.get("vigfree_p"))
else:
    dfq["q_model"] = dfq.get("vigfree_p")

# Ensure decimal odds available
if "decimal_odds" not in dfq.columns or dfq["decimal_odds"].isna().any():
    def american_to_decimal(a):
        a = float(a)
        return 1.0 + (a/100.0 if a > 0 else 100.0/abs(a))
    dfq["decimal_odds"] = dfq["american_odds"].apply(american_to_decimal)

# Market prob & value metrics
dfq["p_market"] = dfq.get("vigfree_p").fillna(1.0/dfq["decimal_odds"].clip(lower=1.0000001))
dfq["edge"] = dfq["q_model"] - dfq["p_market"]
dfq["ev"]   = dfq["q_model"] * dfq["decimal_odds"] - 1.0

# Human category tags
def market_category(mt: str) -> str:
    mt = str(mt)
    if mt == "MONEYLINE": return "Moneyline"
    if mt in {"RUN_LINE","ALT_RUN_LINE"}: return "Run Line"
    if mt == "PITCHER_KS": return "Pitcher Strikeouts"
    if mt == "PITCHER_OUTS": return "Pitcher Outs"
    if mt == "PITCHER_WALKS": return "Pitcher Walks"
    if mt == "PITCHER_WIN": return "Pitcher To Win"
    return mt.title()
dfq["category"] = dfq["market_type"].map(market_category)

# ======================================================
# Sidebar: global filters (applied everywhere)
# ======================================================
with st.sidebar:
    st.header("Filters & Parlay Settings")

    # Global min odds (≥). Example: -500 keeps -500, -400, -300, ... +100 etc.
    min_american = st.slider("Minimum American odds to include (≥)", -700, 700, value=-500, step=25)

    # Parlay minimum decimal odds (7.0 ≈ +600)
    min_parlay_decimal = st.slider("Minimum parlay decimal odds", 3.0, 25.0, value=7.0, step=0.1)

    # We’ll build fixed presets for 4/5/6/8 legs; this max is for any manual/greedy calls
    max_legs = st.slider("Max legs (for manual/greedy builds)", 4, 20, 12, 1)

    # Category filter
    all_cats = ["Moneyline","Run Line","Pitcher Strikeouts","Pitcher Outs","Pitcher Walks","Pitcher To Win"]
    selected_cats = st.multiselect("Categories", all_cats, default=all_cats)

    # Quick toggles
    st.markdown("**Quick filters**")
    pitcher_props_only = st.checkbox("Pitcher props only", value=False)
    moneyline_only = st.checkbox("Moneyline only", value=False)
    runline_only = st.checkbox("Run lines only", value=False)

    # Game filter & text search
    games_all = sorted(dfq["game_id"].dropna().unique().tolist())
    games_selected = st.multiselect("Games", games_all, default=games_all)
    text_search = st.text_input("Search team/player contains", value="").strip()

# Apply global filters to build the working pool
pool = dfq.copy()
pool = pool[pool["american_odds"] >= min_american]
pool = pool[pool["category"].isin(selected_cats)]
pool = pool[pool["game_id"].isin(games_selected)]
if pitcher_props_only: pool = pool[pool["category"].str.startswith("Pitcher")]
if moneyline_only:     pool = pool[pool["category"] == "Moneyline"]
if runline_only:       pool = pool[pool["category"] == "Run Line"]
if text_search:
    s = text_search.lower()
    pool = pool[(pool["team"].fillna("").str.lower().str.contains(s)) |
                (pool["player_name"].fillna("").str.lower().str.contains(s)) |
                (pool["description"].fillna("").str.lower().str.contains(s))]

# Helpers for pretty tables
def add_pct_cols(df):
    out = df.copy()
    out["Model q %"] = out["q_model"] * 100.0
    out["Market %"]  = out["p_market"] * 100.0
    out["Edge %"]    = out["edge"] * 100.0
    return out

colcfg = {
    "american_odds": st.column_config.NumberColumn("Odds", format="%d"),
    "decimal_odds":  st.column_config.NumberColumn("Dec",  format="%.2f"),
    "Model q %":     st.column_config.NumberColumn("Model q %", format="%.1f"),
    "Market %":      st.column_config.NumberColumn("Market %",  format="%.1f"),
    "Edge %":        st.column_config.NumberColumn("Edge %",    format="+%.1f"),
    "ev":            st.column_config.NumberColumn("EV / unit", format="+%.2f"),
}

# ======================================================
# TABS
# ======================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Candidates",
    "Top 20 Bets",
    "Parlay Presets",
    "ML Winners & Alt RL Locks",
    "Locks (≥85%)"
])

# ---------------------------------- Candidates
with tab1:
    st.markdown('<div class="block-title">Candidate Legs (post‑filter)</div>', unsafe_allow_html=True)
    show_cols = ["date","game_id","category","market_type","side","team","player_name","alt_line",
                 "american_odds","decimal_odds","q_model","p_market","edge","ev","description"]
    cand = add_pct_cols(pool[show_cols])
    show_cols_disp = ["date","game_id","category","side","team","player_name","alt_line",
                      "american_odds","decimal_odds","Model q %","Market %","Edge %","ev","description"]
    st.dataframe(
        cand[show_cols_disp].sort_values(["q_model","edge","decimal_odds"], ascending=[False,False,False]),
        use_container_width=True, height=500, column_config=colcfg, hide_index=True
    )
    st.caption("Note: Values reflect current filters (odds slider, categories, games, search).")

# ---------------------------------- Top 20 Bets
with tab2:
    st.markdown('<div class="block-title">Top 20 Bets (≥60% hit, odds ≥ −350, ranked by EV)</div>', unsafe_allow_html=True)
    top = pool.copy()
    min_top_odds = max(min_american, -350)          # enforce odds ≥ −350 regardless of a looser global slider
    top = top[(top["american_odds"] >= min_top_odds) & (top["q_model"] >= 0.60)].copy()
    top["rank_score"] = top["ev"]
    top = top.sort_values(["rank_score","q_model","decimal_odds"], ascending=[False,False,False]).head(20)

    if top.empty:
        st.info("No legs meet the ≥60% & odds ≥ −350 threshold under current filters.")
    else:
        # Structured list with written rationale
        for i, (_, r) in enumerate(top.iterrows(), start=1):
            st.markdown(f"**{i}. {r['description']}**")
            st.markdown(
                f"- **Odds**: {int(r['american_odds'])} | **Model q**: {r['q_model']:.1%} | "
                f"**Market**: {r['p_market']:.1%} | **Edge**: {r['edge']:+.1%} | **EV**: {r['ev']:+.2f}"
            )
            blips = []
            mt = str(r.get("market_type",""))
            if mt == "PITCHER_KS" and pd.notna(r.get("mu_ks", np.nan)):
                blips.append(f"Ks μ≈{r['mu_ks']:.2f} vs line {r.get('alt_line')}.")
            if mt == "PITCHER_OUTS" and pd.notna(r.get("expected_ip", np.nan)):
                blips.append(f"Expected IP≈{r['expected_ip']:.2f} (outs mean≈{r['expected_ip']*3:.1f}) vs line {r.get('alt_line')}.")
            if mt == "PITCHER_WALKS" and pd.notna(r.get("mu_bb", np.nan)):
                blips.append(f"Walks μ≈{r['mu_bb']:.2f} vs line {r.get('alt_line')}.")
            if mt in {"RUN_LINE","ALT_RUN_LINE"} and pd.notna(r.get("alt_line", np.nan)):
                blips.append(f"Run line {r['alt_line']} chosen for safety/value tradeoff.")
            if pd.notna(r.get("pitcher_k_rate", np.nan)):
                blips.append(f"K% {r['pitcher_k_rate']:.0%} vs opp K% {r.get('opp_k_rate', np.nan):.0%}.")
            if pd.notna(r.get("pitcher_bb_rate", np.nan)):
                blips.append(f"BB% {r['pitcher_bb_rate']:.0%} vs opp BB% {r.get('opp_bb_rate', np.nan):.0%}.")
            if pd.notna(r.get("leash_bias", np.nan)) and r["leash_bias"] > 0:
                blips.append("Longer leash expected.")
            if not blips:
                blips.append("Ranked primarily by EV (model q × payout vs market).")
            st.caption(" ".join(blips))

# ---------------------------------- Parlay Presets (exact 4, 5, 6, 8 legs; Low/Med/High)
with tab3:
    st.markdown('<div class="block-title">Parlay Presets (exact 4, 5, 6, 8 legs)</div>', unsafe_allow_html=True)

    # Builder expects column "q" → feed it model prob
    pool_for_builder = pool.copy()
    pool_for_builder["q"] = pool_for_builder["q_model"]

    def build_fixed_parlay(subpool: pd.DataFrame, leg_count: int, mode: str, q_cut: float):
        p = subpool[subpool["q_model"] >= q_cut].copy()
        if p.empty: return None
        # Force exactly leg_count by giving a huge target and bounding max_legs = leg_count
        picks = build_parlay_greedy(p, target_decimal_odds=1e9, min_legs=leg_count, max_legs=leg_count, mode=mode)
        return picks if not picks.empty else None

    def summarize_parlay(label: str, leg_count: int, picks: pd.DataFrame):
        if picks is None or picks.empty:
            return {"Legs": leg_count, "Type": label, "Decimal": np.nan, "Est Hit": np.nan, "Meets +600?": False, "Picks": "--"}
        dec = float(picks.attrs.get("total_decimal_odds", float("nan")))
        est = float(picks.attrs.get("est_hit_prob", float("nan")))
        meets = bool(dec >= min_parlay_decimal)
        picks_desc = " | ".join(picks["description"].tolist())
        return {"Legs": leg_count, "Type": label, "Decimal": round(dec,3), "Est Hit": round(est,3), "Meets +600?": meets, "Picks": picks_desc}

    rows = []
    risk_defs = [("Low", "SAFETY", 0.65), ("Medium", "SAFETY", 0.60), ("High", "VALUE", 0.55)]
    for leg_count in [4,5,6,8]:
        for (label, mode, q_cut) in risk_defs:
            picks = build_fixed_parlay(pool_for_builder, leg_count, mode, q_cut)
            rows.append(summarize_parlay(label, leg_count, picks))

    parlays_tbl = pd.DataFrame(rows).sort_values(["Legs","Type"])
    st.dataframe(parlays_tbl, use_container_width=True, hide_index=True)

# ---------------------------------- ML Winners & Alt RL Locks
with tab4:
    st.markdown('<div class="block-title">Moneyline Winners (by game)</div>', unsafe_allow_html=True)
    ml = pool[pool["category"]=="Moneyline"].copy()
    # pick the side with highest model probability per game
    if ml.empty:
        st.info("No moneyline markets in the current pool.")
    else:
        ml_best = (ml.sort_values(["game_id","q_model","ev"], ascending=[True,False,False])
                      .groupby("game_id", as_index=False).head(1))
        ml_best["ML Lock?"] = ml_best["q_model"] >= 0.65   # strong ML confidence threshold
        ml_disp = add_pct_cols(ml_best[[
            "game_id","side","team","american_odds","decimal_odds","q_model","p_market","edge","ev","description","ML Lock?"
        ]])
        ml_show = ["game_id","team","side","american_odds","decimal_odds","Model q %","Market %","Edge %","ev","ML Lock?","description"]
        st.dataframe(ml_disp[ml_show].sort_values(["ML Lock?","q_model","ev"], ascending=[False,False,False]),
                     use_container_width=True, hide_index=True, column_config=colcfg)
        st.caption("Picks are the most probable moneyline side in each game (model q). ML Lock? = q ≥ 65%.")

    st.markdown('<div class="block-title" style="margin-top:10px;">Alternate Run Line -- Strong Candidates</div>', unsafe_allow_html=True)
    arl = pool[pool["category"]=="Run Line"].copy()
    if arl.empty:
        st.info("No run line markets in the current pool.")
    else:
        # top ARL per game by model prob (favor safer alt lines automatically if offered)
        arl_best = (arl.sort_values(["game_id","q_model","ev"], ascending=[True,False,False])
                       .groupby("game_id", as_index=False).head(1))
        arl_best["ARL Lock?"] = arl_best["q_model"] >= 0.70
        arl_disp = add_pct_cols(arl_best[[
            "game_id","team","side","alt_line","american_odds","decimal_odds","q_model","p_market","edge","ev","description","ARL Lock?"
        ]])
        arl_show = ["game_id","team","side","alt_line","american_odds","decimal_odds","Model q %","Market %","Edge %","ev","ARL Lock?","description"]
        st.dataframe(arl_disp[arl_show].sort_values(["ARL Lock?","q_model","ev"], ascending=[False,False,False]),
                     use_container_width=True, hide_index=True, column_config=colcfg)
        st.caption("Strong ARL candidates: best run line per game by model q (ARL Lock? = q ≥ 70%).")

# ---------------------------------- Locks (≥85%)
with tab5:
    st.markdown('<div class="block-title">Locks (model ≥ 85% to hit)</div>', unsafe_allow_html=True)
    locks = pool[pool["q_model"] >= 0.85].copy().sort_values(["q_model","edge"], ascending=[False,False]).head(20)
    if locks.empty:
        st.info("No 85%+ locks under current filters.")
    else:
        locks_disp = add_pct_cols(locks[[
            "category","description","american_odds","decimal_odds","q_model","p_market","edge","ev"
        ]])
        lock_show = ["category","description","american_odds","decimal_odds","Model q %","Market %","Edge %","ev"]
        st.dataframe(locks_disp[lock_show],
                     use_container_width=True, hide_index=True, column_config=colcfg)
        st.caption("Reminder: correlations and late news can change true hit rates. Use responsibly.")