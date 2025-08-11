# app.py -- Clean, optimized, with schedule+pitchers+opp rates and downloads
import os, io, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

from parlay.odds import american_to_decimal
from parlay.builder import build_presets, build_one_tap
from parlay.describe import describe_row, matchup, compact_player
from parlay.ui import inject_css, parlay_card

from etl.fetch_and_build import run as etl_run
from etl.sources.opp_rates import load_opp_rates

st.set_page_config(page_title="MLB Parlay Picker -- MVP", layout="wide", page_icon="⚾")
inject_css()

@st.cache_data(ttl=60*30)
def _cache_etl(date_str: str):
    return etl_run(date_str)

@st.cache_data(ttl=60*60)
def _cache_opp_rates():
    return load_opp_rates()

def add_missing(df: pd.DataFrame, cols, val=np.nan):
    d = df.copy()
    for c in cols:
        if c not in d.columns: d[c] = val
    return d

def _player_key(name: str) -> str:
    name = (name or "").replace(".", "").strip()
    if not name: return ""
    parts = [p for p in name.split() if p]
    if len(parts) == 1: return (parts[0][:1] + parts[0]).lower()
    return (parts[0][:1] + parts[-1]).lower()

def build_pool(dk_df: pd.DataFrame, feat_df: pd.DataFrame,
               schedule_df: pd.DataFrame, pitchers_df: pd.DataFrame,
               opp_rates: pd.DataFrame) -> pd.DataFrame:
    # Ensure decimals and market probabilities exist
    df = dk_df.copy()

    # If DK missing, at least provide schedule rows for filters (no odds)
    if df.empty:
        df = schedule_df.assign(market_type="MONEYLINE", team=schedule_df["home_abbr"],
                                side="YES", alt_line=None, american_odds=0, player_id=None, player_name=None)[
            ["date","game_id","market_type","team","side","alt_line","american_odds","player_id","player_name"]
        ]

    if "decimal_odds" not in df.columns:
        df["decimal_odds"] = df["american_odds"].astype(float).apply(lambda x: american_to_decimal(int(x)) if x != 0 else 1.0)
    if "p_market" not in df.columns:
        df["p_market"] = (1.0/df["decimal_odds"]).replace([np.inf,-np.inf], np.nan).clip(0.01,0.99).fillna(0.5)

    # Hydrate schedule columns for every row (home/away abbr, matchup)
    base_cols = ["home_abbr","away_abbr","matchup"]
    df = df.merge(schedule_df[["game_id"]+base_cols], on="game_id", how="left")

    # Attach pitcher team via MLB probable pitchers
    if "player_name" in df.columns:
        df["player_key"] = df["player_name"].apply(_player_key)
        pi = pitchers_df[["game_id","player_key","team_abbr"]].drop_duplicates()
        df = df.merge(pi, on=["game_id","player_key"], how="left", suffixes=("",""))
        # fallback: if still missing, use team column (moneylines), else home_abbr
        df["team_abbr"] = df["team_abbr"].fillna(df.get("team","")).replace("", np.nan)
        df["team_abbr"] = df["team_abbr"].fillna(df["home_abbr"])
    else:
        df["team_abbr"] = df.get("team","").fillna(df["home_abbr"])

    # Opponent team for pitcher props: if row team is home, opponent=away; else opponent=home
    df["opp_abbr"] = np.where(df["team_abbr"] == df["home_abbr"], df["away_abbr"], df["home_abbr"])

    # Attach opponent K%/BB% by team abbr (21d Statcast, cached) -- varied per team
    rates = opp_rates.rename(columns={"team_abbr":"opp_abbr"}).copy()
    df = df.merge(rates, on="opp_abbr", how="left")
    # If some teams missing from statcast (rare), fallback by mapping dictionary
    # (Already handled by load_opp_rates fallback.)

    # Model probability -- blend if user projections present
    if "q_proj" in feat_df.columns and not feat_df.empty:
        # Join on player_id when present, else on player_key if both exist
        if "player_id" in df.columns and "player_id" in feat_df.columns:
            m = df.merge(feat_df[["player_id","q_proj"]], on="player_id", how="left")
        elif "player_key" in df.columns and "player_key" in feat_df.columns:
            m = df.merge(feat_df[["player_key","q_proj"]], on="player_key", how="left")
        else:
            m = df.copy()
        df = m
        df["q_model"] = df["q_proj"].where(df["q_proj"].notna(), 0.6*df["p_market"] + 0.4*0.50)
    else:
        df["q_model"] = (0.6*df["p_market"] + 0.4*0.50)
    df["q_model"] = df["q_model"].clip(0.01,0.99)

    # Economics
    df["edge"] = (df["q_model"] - df["p_market"]).fillna(0.0)
    df["ev"]   = (df["q_model"] * df["decimal_odds"] - 1.0).fillna(0.0)

    # Compact category and description
    def cat(r):
        return {"PITCHER_KS":"Pitcher Ks","PITCHER_OUTS":"Pitcher Outs","PITCHER_WALKS":"Pitcher BB",
                "PITCHER_WIN":"Pitcher Win","MONEYLINE":"Moneyline","RUN_LINE":"Run Line","ALT_RUN_LINE":"Alt Run Line"}.get(r["market_type"], r["market_type"])
    df["category"] = df.apply(cat, axis=1)
    df["description"] = df.apply(describe_row, axis=1)
    df["game_code"] = df["matchup"]  # already "AWAY@HOME"
    df["american_odds"] = df["american_odds"].astype(int, errors="ignore")

    return df

# ---------- UI ----------
st.title("MLB Parlay Picker -- MVP")
st.caption("Clean UI · Full game coverage via MLB schedule · Correct teams on players · True opponent K%/BB% · Downloadable CSVs.")

with st.expander("How to use"):
    st.markdown("1) Toggle **Cloud Mode** to auto‑fetch today. 2) Optional: upload **Features CSV** with column `q_proj`. 3) Filter games/teams/odds. 4) Review **Top‑20**, **Presets**, **Safety**, **One‑Tap**, **ML Locks**.")

today = dt.date.today().strftime("%Y-%m-%d")
cloud_on = st.toggle("Enable Cloud Mode (auto‑fetch today)", value=False)

odds_df = None; feat_df = pd.DataFrame({"player_id":[], "q_proj":[]})
schedule_df = None; pitchers_df = None
if cloud_on:
    try:
        odds_df, feat_df0, odds_csv, feat_csv, schedule_df, pitchers_df = _cache_etl(today)
        st.success(f"Fetched data for {today}.")
        c1,c2 = st.columns(2)
        with c1:
            st.download_button("Download DK CSV", data=odds_csv, file_name=f"dk_{today}.csv", mime="text/csv", use_container_width=True)
        with c2:
            st.download_button("Download Features CSV", data=feat_csv, file_name=f"features_{today}.csv", mime="text/csv", use_container_width=True)
        feat_df = feat_df0.copy()
    except Exception as e:
        st.error(f"Cloud fetch failed: {e}")

# Manual uploads (still supported)
up_odds = st.file_uploader("Upload DK Markets CSV", type=["csv"], key="dk_csv")
if up_odds is not None:
    odds_df = pd.read_csv(up_odds)

up_feat = st.file_uploader("Upload Features CSV (optional, adds q_proj)", type=["csv"], key="feat_csv")
if up_feat is not None:
    feat_df = pd.read_csv(up_feat)

if odds_df is None or schedule_df is None or pitchers_df is None:
    st.info("Cloud fetch recommended. If using manual, please also fetch schedule/pitchers. (Cloud Mode handles this automatically.)")
    # Minimal schedule if missing: derive from odds_df game_ids
    if odds_df is not None:
        tmp = odds_df["game_id"].dropna().unique().tolist()
        schedule_df = pd.DataFrame({"game_id": tmp, "home_abbr":"","away_abbr":"","matchup": [matchup(g) for g in tmp]})
        pitchers_df = pd.DataFrame(columns=["game_id","player_key","team_abbr"])

# Opponent rates (statcast cached or fallback)
opp_rates = _cache_opp_rates()

# Build pool
pool_raw = build_pool(odds_df, feat_df, schedule_df, pitchers_df, opp_rates)

# ---------- Filters ----------
st.divider()
st.subheader("Filters")

f1,f2,f3,f4 = st.columns([1,1,1,1])
with f1:
    games = sorted(pool_raw["game_id"].dropna().unique().tolist())
    gmap = dict(zip(games, pool_raw.drop_duplicates("game_id")[["game_id","game_code"]].set_index("game_id")["game_code"]))
    sel_all_g = st.checkbox("Select all games", value=True)
    games_sel = games if sel_all_g else st.multiselect("Games", options=games, default=games, format_func=lambda k: gmap.get(k,k))
with f2:
    teams = sorted([t for t in pool_raw["team_abbr"].dropna().unique().tolist() if t])
    sel_all_t = st.checkbox("Select all teams", value=True)
    teams_sel = teams if sel_all_t else st.multiselect("Teams", options=teams, default=teams)
with f3:
    cats_all = sorted(pool_raw["category"].dropna().unique().tolist())
    cats = st.multiselect("Categories", options=cats_all, default=cats_all)
with f4:
    odds_min, odds_max = st.slider("American odds range", -700, 700, (-700, 700), 25)

mask = (pool_raw["game_id"].isin(games_sel)) & (pool_raw["category"].isin(cats)) & (pool_raw["american_odds"].between(odds_min, odds_max))
if teams_sel: mask &= pool_raw["team_abbr"].isin(teams_sel)
pool = pool_raw[mask].copy()

with st.expander("Coverage (from MLB schedule)"):
    st.write(", ".join(sorted(pool_raw.drop_duplicates("game_id")["game_code"].tolist())))

if pool.empty:
    st.warning("No legs after filters.")
    st.stop()

# ---------- Tabs ----------
tab1, tab2, tab3, tabSafety, tabOneTap, tab4, tab5 = st.tabs(
    ["Candidates", "Top 20", "Parlay Presets", "Alt Line Safety Board", "One‑Tap Ticket", "ML Winners & Alt RL Locks", "Locks (≥85%)"]
)

with tab1:
    show = pool[["description","category","american_odds","decimal_odds","q_model","p_market","edge","ev","game_code","team_abbr","opp_k_rate","opp_bb_rate"]].copy()
    show.rename(columns={"american_odds":"Odds","decimal_odds":"Dec","q_model":"q","p_market":"Market","edge":"Edge","ev":"EV","game_code":"Game","team_abbr":"Team","opp_k_rate":"OppK","opp_bb_rate":"OppBB"}, inplace=True)
    st.dataframe(show.sort_values(["q","Edge","Dec"], ascending=[False,False,False]), use_container_width=True, hide_index=True)

with tab2:
    st.markdown("**Top 20 Bets (q ≥ 60%, odds ≥ −350)**")
    top = pool[(pool["q_model"] >= 0.60) & (pool["american_odds"] >= -350)].copy()
    top = top.sort_values(["ev","q_model","decimal_odds"], ascending=[False,False,False]).head(20)
    if top.empty: st.info("No legs meet thresholds.")
    else:
        for i,(_,r) in enumerate(top.iterrows(), start=1):
            st.markdown(f"**{i}. {r['description']}**")
            st.markdown(f"- **Odds** {int(r['american_odds'])} · **q** {r['q_model']:.1%} · **Market** {r['p_market']:.1%} · **Edge** {r['edge']:+.1%} · **EV** {r['ev']:+.2f}")
            note_parts = []
            if pd.notna(r.get("opp_k_rate")): note_parts.append(f"OppK {r['opp_k_rate']:.0%}")
            if pd.notna(r.get("opp_bb_rate")): note_parts.append(f"OppBB {r['opp_bb_rate']:.0%}")
            st.caption(" • ".join(note_parts) or "--")

with tab3:
    st.markdown("**Parlay Presets (4 · 5 · 6 · 8 legs)**")
    presets = build_presets(pool, legs_list=(4,5,6,8), min_parlay_am="+600", odds_min=odds_min, odds_max=odds_max)
    for legs in (4,5,6,8):
        st.markdown(f"#### {legs}‑Leg")
        cols = st.columns(3)
        for j,tier in enumerate(["Low","Medium","High"]):
            pr = presets.get((legs,tier), {"legs": [], "decimal": 1.0, "q_est": 0.0, "meets_min": False})
            with cols[j]:
                parlay_card(tier, pr["legs"], pr["decimal"], pr["q_est"], pr["meets_min"])

with tabSafety:
    st.markdown("**Alt Line Safety Board**")
    q_thr = st.slider("Min q", 0.55, 0.95, 0.70, 0.01)
    safe = pool[(pool["q_model"] >= q_thr) & (pool["category"].isin(["Pitcher Ks","Pitcher Outs","Pitcher BB","Alt Run Line","Run Line","Moneyline"]))].copy()
    show = safe[["description","american_odds","q_model","p_market","edge","ev","game_code","team_abbr"]].rename(
        columns={"american_odds":"Odds","q_model":"q","p_market":"Market","edge":"Edge","ev":"EV","game_code":"Game","team_abbr":"Team"})
    st.dataframe(show.sort_values(["q","Edge","EV"], ascending=[False,False,False]), use_container_width=True, hide_index=True)

with tabOneTap:
    st.markdown("**One‑Tap Ticket**")
    cA,cB = st.columns(2)
    legs = cA.selectbox("Legs", [4,5,6,8], index=0)
    risk = cB.selectbox("Risk", ["Low","Medium","High"], index=0)
    one = build_one_tap(pool, legs=legs, risk=risk, odds_min=odds_min, odds_max=odds_max)
    parlay_card(f"{legs}-Leg {risk}", one.get("legs", []), one.get("decimal", 1.0), one.get("q_est", 0.0), one.get("meets_min", False))

with tab4:
    st.markdown("**Moneyline Picks & Alt RL Locks**")
    ml = pool[pool["category"]=="Moneyline"].copy()
    if ml.empty: st.info("No ML markets in current pool.")
    else:
        ml_best = ml.sort_values(["q_model","ev"], ascending=[False,False]).drop_duplicates("game_id")
        ml_best["ML Lock?"] = (ml_best["q_model"] >= 0.65)
        show = ml_best[["description","american_odds","decimal_odds","q_model","p_market","edge","ev","game_code"]].rename(
            columns={"american_odds":"Odds","decimal_odds":"Dec","q_model":"q","p_market":"Market","edge":"Edge","ev":"EV","game_code":"Game"})
        st.dataframe(show.sort_values(["ML Lock?","q","EV"], ascending=[False,False,False]), use_container_width=True, hide_index=True)

with tab5:
    st.markdown("**Locks (q ≥ 85%)**")
    locks = pool[pool["q_model"] >= 0.85].copy()
    show = locks[["description","american_odds","q_model","ev","category","game_code"]].rename(
        columns={"american_odds":"Odds","q_model":"q","ev":"EV","game_code":"Game"})
    st.dataframe(show.sort_values(["q","EV"], ascending=[False,False]), use_container_width=True, hide_index=True)

st.caption("All games are shown via MLB schedule. If odds are not yet posted for a matchup, you’ll still see it in filters/coverage.")