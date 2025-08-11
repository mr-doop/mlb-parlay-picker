# app.py -- Clean, resilient Streamlit app with real features, fixed run-lines, and Apple-like UI cues
from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

# Internal modules (already in your repo)
from parlay.odds import american_to_decimal
from parlay.builder import build_presets, build_one_tap
from parlay.describe import describe_row, matchup
from parlay.ui import inject_css, parlay_card

# NEW helpers we added
from parlay.feature_join import normalize_features, attach_projections

# Updated ETL that now builds real features and fixes run-line alt_line
from etl.fetch_and_build import run as etl_run
from etl.sources.opp_rates import load_opp_rates

# --------------------------- Page & Style ---------------------------
st.set_page_config(page_title="MLB Parlay Picker -- MVP", layout="wide", page_icon="⚾")
inject_css()  # keep your minimal Apple-like CSS in parlay/ui.py

# --------------------------- Cache ---------------------------
@st.cache_data(ttl=60 * 30)
def _cache_etl(date_str: str):
    """Fetch DK board, schedule, pitchers, and auto-generated features."""
    return etl_run(date_str)

@st.cache_data(ttl=60 * 60)
def _cache_opp_rates():
    """Opp 21-day rolling K%/BB% by opponent team."""
    return load_opp_rates()

# --------------------------- Utilities ---------------------------
def _safe_decimal(x):
    try:
        return american_to_decimal(int(x))
    except Exception:
        return 1.0

def _player_key(name) -> str:
    """A.B. -> aplayerkey (first initial + last name), robust to NaN."""
    if not isinstance(name, (str, np.str_)):
        return ""
    s = name.replace(".", "").strip()
    if not s or s.lower() == "nan":
        return ""
    parts = [p for p in s.split() if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return (parts[0][:1] + parts[0]).lower()
    return (parts[0][:1] + parts[-1]).lower()

def build_pool(dk_df: pd.DataFrame,
               schedule_df: pd.DataFrame,
               pitchers_df: pd.DataFrame,
               opp_rates: pd.DataFrame) -> pd.DataFrame:
    """Assemble the working pool:
       - DK board (or schedule fallback)
       - Join schedule for matchup, teams
       - Map pitcher -> team via probable pitchers
       - Add opponent 21d K%/BB%
       - Baseline q, EV/Edge
       - Friendly description text
    """
    if dk_df is None or dk_df.empty:
        # Fallback to schedule to avoid "missing games"
        base = schedule_df.assign(
            market_type="MONEYLINE",
            team=schedule_df["home_abbr"],
            side="YES",
            alt_line=np.nan,
            american_odds=0,
            player_id=None,
            player_name=None,
        )[["date","game_id","market_type","team","side","alt_line","american_odds","player_id","player_name"]]
        df = base.copy()
    else:
        df = dk_df.copy()

    # Odds math
    if "decimal_odds" not in df.columns:
        df["decimal_odds"] = df["american_odds"].apply(_safe_decimal)
    if "p_market" not in df.columns:
        df["p_market"] = (1.0 / df["decimal_odds"]).replace([np.inf, -np.inf], np.nan).clip(0.01, 0.99).fillna(0.50)

    # Join schedule for authoritative matchup & home/away teams
    sch_cols = ["home_abbr", "away_abbr", "matchup"]
    df = df.merge(schedule_df[["game_id"] + sch_cols], on="game_id", how="left")

    # Player -> team via probable pitchers (fixes missing team on players)
    if "player_name" in df.columns:
        df["player_key"] = df["player_name"].apply(_player_key)
        pi = pitchers_df[["game_id","player_key","team_abbr"]].drop_duplicates()
        df = df.merge(pi, on=["game_id","player_key"], how="left")
        # Team fallback: if not mapped, use the DK "team" or the home team
        df["team_abbr"] = df["team_abbr"].fillna(df.get("team","")).replace("", np.nan)
        df["team_abbr"] = df["team_abbr"].fillna(df["home_abbr"])
    else:
        df["team_abbr"] = df.get("team","").fillna(df["home_abbr"])

    # Opponent rates (21-day rolling), merged by opponent team
    df["opp_abbr"] = np.where(df["team_abbr"] == df["home_abbr"], df["away_abbr"], df["home_abbr"])
    rates = opp_rates.rename(columns={"team_abbr": "opp_abbr"})
    df = df.merge(rates, on="opp_abbr", how="left")

    # Ensure run-line alt_line displays cleanly (DK sometimes omits point)
    df.loc[(df["market_type"] == "RUN_LINE") & (df["alt_line"].isna()), "alt_line"] = 1.5

    # Baseline model (prior blended with market; projections can override)
    df["q_model"] = (0.6 * df["p_market"] + 0.4 * 0.50).clip(0.01, 0.99)

    # Economics
    df["edge"] = (df["q_model"] - df["p_market"]).fillna(0.0)
    df["ev"]   = (df["q_model"] * df["decimal_odds"] - 1.0).fillna(0.0)

    # Category & description
    def _cat(mkt):
        return {
            "PITCHER_KS": "Pitcher Ks",
            "PITCHER_OUTS": "Pitcher Outs",
            "PITCHER_WALKS": "Pitcher BB",
            "PITCHER_WIN": "Pitcher Win",
            "MONEYLINE": "Moneyline",
            "RUN_LINE": "Run Line",
            "ALT_RUN_LINE": "Alt Run Line",
        }.get(mkt, mkt)
    df["category"] = df["market_type"].apply(_cat)

    # Human-readable description for cards/tables
    df["description"] = df.apply(describe_row, axis=1)

    # Convenience columns
    df["game_code"] = df["matchup"]
    try:
        df["american_odds"] = df["american_odds"].astype(int)
    except Exception:
        pass

    return df

# --------------------------- Header & Help ---------------------------
st.title("MLB Parlay Picker -- MVP")
st.caption("Full schedule coverage · Correct team mapping on players · Opponent 21‑day K%/BB% · "
           "Projections attach via CSV · Clean run‑line text · Apple‑like UI.")

with st.expander("How to use (quick)"):
    st.markdown(
        "1) Toggle **Cloud Mode** to fetch today’s DK board and schedule.\n"
        "2) (Optional) Upload a **Features CSV** with probabilities (`q`/`prob`) at player or team level; "
        "the app normalizes and attaches them progressively (player+line → player → team+line → team).\n"
        "3) Use Filters, then review **Candidates**, **Top 20**, **Parlay Presets**, **Alt Line Safety Board**, **One‑Tap Ticket**, **ML Locks**.\n"
        "4) Download the DK & Features CSVs directly after Cloud fetch."
    )

# --------------------------- Fetch / Upload ---------------------------
today = dt.date.today().strftime("%Y-%m-%d")
cloud_on = st.toggle("Enable Cloud Mode (auto‑fetch today)", value=False)

odds_df = None
schedule_df = None
pitchers_df = None
feat_df_raw = pd.DataFrame()

if cloud_on:
    try:
        odds_df, feat_df_auto, odds_csv, feat_csv, schedule_df, pitchers_df = _cache_etl(today)
        st.success(f"Fetched DraftKings & schedule for **{today}**.")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Download DK CSV", data=odds_csv, file_name=f"dk_{today}.csv",
                               mime="text/csv", use_container_width=True)
        with c2:
            st.download_button("Download Features CSV", data=feat_csv, file_name=f"features_{today}.csv",
                               mime="text/csv", use_container_width=True)
        # Preload the features built from DK board (team-level ML/RL) so features is never empty
        feat_df_raw = feat_df_auto.copy()
    except Exception as e:
        st.error(f"Cloud fetch failed: {e}")

up_odds = st.file_uploader("Upload DK Markets CSV", type=["csv"], key="dk_csv")
if up_odds is not None:
    odds_df = pd.read_csv(up_odds)

up_feat = st.file_uploader("Upload Features CSV (optional; attaches q_proj)", type=["csv"], key="feat_csv")
if up_feat is not None:
    feat_df_raw = pd.read_csv(up_feat)

# Fallback schedule build if manual-only
if schedule_df is None:
    if odds_df is not None and "game_id" in odds_df.columns:
        gids = odds_df["game_id"].dropna().unique().tolist()
        schedule_df = pd.DataFrame({
            "game_id": gids,
            "home_abbr": ["" for _ in gids],
            "away_abbr": ["" for _ in gids],
            "matchup": [matchup(g) for g in gids],
            "date": today,
        })
    else:
        st.info("Upload a DK CSV or enable Cloud Mode.")
        st.stop()

if pitchers_df is None:
    pitchers_df = pd.DataFrame(columns=["game_id", "player_key", "team_abbr"])

opp_rates = _cache_opp_rates()

# --------------------------- Build Pool & Apply Projections ---------------------------
pool_base = build_pool(odds_df, schedule_df, pitchers_df, opp_rates)

feat_norm = normalize_features(feat_df_raw)
pool, applied = attach_projections(pool_base, feat_norm)

if applied > 0:
    st.success(f"Applied true projections to **{applied}** legs.")
elif up_feat is not None:
    st.warning("Uploaded features did not match any legs. Check columns like player/team, market, side, line, "
               "and that probabilities are 0–1 (e.g., 0.63).")

# --------------------------- Filters ---------------------------
st.divider()
st.subheader("Filters")

c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

with c1:
    games = sorted(pool["game_id"].dropna().unique().tolist())
    # Map game_id -> matchup label
    game_labels = pool.drop_duplicates("game_id")[["game_id", "matchup"]].set_index("game_id")["matchup"].to_dict()
    sel_all_g = st.checkbox("Select all games", value=True)
    games_sel = games if sel_all_g else st.multiselect(
        "Games", options=games, default=games, format_func=lambda k: game_labels.get(k, k)
    )

with c2:
    teams = sorted([t for t in pool["team_abbr"].dropna().unique().tolist() if t])
    sel_all_t = st.checkbox("Select all teams", value=True)
    teams_sel = teams if sel_all_t else st.multiselect("Teams", options=teams, default=teams)

with c3:
    cats_all = sorted(pool["category"].dropna().unique().tolist())
    cats_sel = st.multiselect("Categories", options=cats_all, default=cats_all)

with c4:
    odds_min, odds_max = st.slider("American odds", -700, 700, (-700, 700), step=25)

# Apply filters
mask = (pool["game_id"].isin(games_sel)) & (pool["category"].isin(cats_sel)) & (pool["american_odds"].between(odds_min, odds_max))
if teams_sel:
    mask &= pool["team_abbr"].isin(teams_sel)

poolf = pool[mask].copy()

with st.expander("Coverage (from MLB schedule) -- games detected"):
    st.write(", ".join(sorted(pool.drop_duplicates("game_id")["matchup"].tolist())))

if poolf.empty:
    st.warning("No legs after filters. Try widening the odds range or selecting more categories.")
    st.stop()

# --------------------------- Tabs ---------------------------
tab1, tab2, tab3, tabSafety, tabOneTap, tab4, tab5 = st.tabs(
    ["Candidates", "Top 20", "Parlay Presets", "Alt Line Safety Board", "One‑Tap Ticket", "ML Winners & Alt RL Locks", "Locks (≥85%)"]
)

# ---- Candidates ----
with tab1:
    show = poolf[[
        "description","category","american_odds","decimal_odds","q_model","p_market",
        "edge","ev","matchup","team_abbr","opp_k_rate","opp_bb_rate"
    ]].copy()
    show.rename(columns={
        "american_odds":"Odds","decimal_odds":"Dec","q_model":"q","p_market":"Market",
        "edge":"Edge","ev":"EV","matchup":"Game","team_abbr":"Team",
        "opp_k_rate":"OppK","opp_bb_rate":"OppBB"
    }, inplace=True)
    st.dataframe(show.sort_values(["q","Edge","Dec"], ascending=[False, False, False]),
                 use_container_width=True, hide_index=True)

# ---- Top 20 ----
with tab2:
    st.markdown("**Top 20 Bets (q ≥ 60%, odds ≥ −350)**")
    top = poolf[(poolf["q_model"] >= 0.60) & (poolf["american_odds"] >= -350)].copy()
    top = top.sort_values(["ev","q_model","decimal_odds"], ascending=[False,False,False]).head(20)
    if top.empty:
        st.info("No legs meet thresholds.")
    else:
        for i,(_,r) in enumerate(top.iterrows(), start=1):
            st.markdown(f"**{i}. {r['description']}**")
            st.markdown(f"- **Odds** {int(r['american_odds'])} · **q** {r['q_model']:.1%} · "
                        f"**Market** {r['p_market']:.1%} · **Edge** {r['edge']:+.1%} · **EV** {r['ev']:+.2f}")
            parts = []
            if pd.notna(r.get("opp_k_rate")):
                parts.append(f"OppK {r['opp_k_rate']:.0%}")
            if pd.notna(r.get("opp_bb_rate")):
                parts.append(f"OppBB {r['opp_bb_rate']:.0%}")
            st.caption(" • ".join(parts) or "--")

# ---- Presets ----
with tab3:
    st.markdown("**Parlay Presets (4 · 5 · 6 · 8 legs)**")
    presets = build_presets(poolf, legs_list=(4,5,6,8), min_parlay_am="+600",
                            odds_min=odds_min, odds_max=odds_max)
    for legs in (4,5,6,8):
        st.markdown(f"#### {legs}‑Leg")
        cols = st.columns(3)
        for j, tier in enumerate(["Low", "Medium", "High"]):
            pr = presets.get((legs, tier), {"legs": [], "decimal": 1.0, "q_est": 0.0, "meets_min": False})
            with cols[j]:
                parlay_card(tier, pr["legs"], pr["decimal"], pr["q_est"], pr["meets_min"])

# ---- Alt Line Safety ----
with tabSafety:
    st.markdown("**Alt Line Safety Board**")
    q_thr = st.slider("Min q", 0.55, 0.95, 0.70, 0.01)
    safe = poolf[(poolf["q_model"] >= q_thr) &
                 (poolf["category"].isin(["Pitcher Ks", "Pitcher Outs", "Pitcher BB",
                                          "Alt Run Line", "Run Line", "Moneyline"]))].copy()
    show = safe[["description","american_odds","q_model","p_market","edge","ev","matchup","team_abbr"]].rename(
        columns={"american_odds":"Odds","q_model":"q","p_market":"Market","edge":"Edge","ev":"EV",
                 "matchup":"Game","team_abbr":"Team"})
    st.dataframe(show.sort_values(["q","Edge","EV"], ascending=[False,False,False]),
                 use_container_width=True, hide_index=True)

# ---- One‑Tap Ticket ----
with tabOneTap:
    st.markdown("**One‑Tap Ticket**")
    cA, cB = st.columns(2)
    legs = cA.selectbox("Legs", [4,5,6,8], index=0)
    risk = cB.selectbox("Risk", ["Low","Medium","High"], index=0)
    one = build_one_tap(poolf, legs=legs, risk=risk, odds_min=odds_min, odds_max=odds_max)
    parlay_card(f"{legs}-Leg {risk}", one.get("legs", []), one.get("decimal", 1.0),
                one.get("q_est", 0.0), one.get("meets_min", False))

# ---- ML Winners & Alt RL Locks ----
with tab4:
    st.markdown("**Moneyline Picks & Alt RL Locks**")
    ml = poolf[poolf["category"] == "Moneyline"].copy()
    if ml.empty:
        st.info("No ML markets in current pool.")
    else:
        ml_best = ml.sort_values(["q_model","ev"], ascending=[False,False]).drop_duplicates("game_id")
        ml_best["ML Lock?"] = (ml_best["q_model"] >= 0.65)
        show = ml_best[["description","american_odds","decimal_odds","q_model","p_market","edge","ev","matchup"]].rename(
            columns={"american_odds":"Odds","decimal_odds":"Dec","q_model":"q","p_market":"Market",
                     "edge":"Edge","ev":"EV","matchup":"Game"})
        st.dataframe(show.sort_values(["ML Lock?","q","EV"], ascending=[False,False,False]),
                     use_container_width=True, hide_index=True)

# ---- Locks (≥85%) ----
with tab5:
    st.markdown("**Locks (q ≥ 85%)**")
    locks = poolf[poolf["q_model"] >= 0.85].copy()
    show = locks[["description","american_odds","q_model","ev","category","matchup"]].rename(
        columns={"american_odds":"Odds","q_model":"q","ev":"EV","matchup":"Game"})
    st.dataframe(show.sort_values(["q","EV"], ascending=[False,False]),
                 use_container_width=True, hide_index=True)

st.caption("If some games still look empty, the book may not have posted those markets yet. "
           "Widen odds range, or check back later. Downloads are available after Cloud fetch.")