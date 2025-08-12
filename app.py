# app.py
from __future__ import annotations

import io
import math
from datetime import datetime, date
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ---- Local modules (already provided in earlier steps) ----
# - etl.fetch_and_build.run(date_str) returns:
#   odds_df, feat_df_auto, dk_csv_bytes, feat_csv_bytes, schedule_df, pitchers_df
from etl.fetch_and_build import run as cloud_fetch
from parlay.feature_join import normalize_features, attach_projections

# --------------- Small odds utils (self-contained) ----------------

def american_to_decimal(a):
    if pd.isna(a):
        return np.nan
    try:
        a = int(a)
    except Exception:
        return np.nan
    if a < 0:
        return 1 + (100 / abs(a))
    else:
        return 1 + (a / 100.0)

def implied_from_decimal(d):
    try:
        d = float(d)
        if d <= 1:
            return np.nan
        return 1.0 / d
    except Exception:
        return np.nan

def fair_two_way(dec_a, dec_b):
    """Return vig-free probabilities for two outcomes from decimal odds."""
    pa = implied_from_decimal(dec_a)
    pb = implied_from_decimal(dec_b)
    if any(pd.isna([pa, pb])) or (pa + pb) <= 0:
        return np.nan, np.nan
    s = pa + pb
    return pa / s, pb / s

# ---------- Formatting helpers (names, descriptions, chips) ----------

CAT_DISPLAY = {
    "MONEYLINE": "Moneyline",
    "RUN_LINE": "Run Line",
    "ALT_RUN_LINE": "Run Line (Alt)",
    "PITCHER_KS": "Pitcher Ks",
    "PITCHER_OUTS": "Pitcher Outs",
    "PITCHER_WALKS": "Pitcher Walks",
    "PITCHER_WIN": "Pitcher Win",
}

def short_name(full: str) -> str:
    if not isinstance(full, str) or not full.strip():
        return ""
    parts = full.replace(".", "").split()
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0][0]}. {parts[-1]}"

def fmt_team(t: str) -> str:
    return (t or "").upper()[:3]

def make_matchup(away: str, home: str) -> str:
    a = fmt_team(away)
    h = fmt_team(home)
    if not a and not h:
        return ""
    return f"{a}@{h}"

def describe_leg(row: pd.Series) -> str:
    cat = CAT_DISPLAY.get(row["market_type"], row["market_type"])
    team = fmt_team(row.get("team_abbr", ""))
    matchup = row.get("matchup", "")
    side = row.get("side", "")
    alt = row.get("alt_line", np.nan)
    player = row.get("player_name", "")
    pshort = short_name(player)
    if row["market_type"] == "MONEYLINE":
        return f"{team} ML ({matchup})"
    if row["market_type"] in {"RUN_LINE", "ALT_RUN_LINE"}:
        if pd.notna(alt):
            sgn = "+" if alt > 0 else ""
            return f"{team} {sgn}{alt:g} ({matchup})"
        return f"{team} RL ({matchup})"
    # Player props
    if row["market_type"] == "PITCHER_KS":
        return f"{pshort} ({team}) {side[0]}{alt:g} Ks"
    if row["market_type"] == "PITCHER_OUTS":
        return f"{pshort} ({team}) {side[0]}{alt:g} Outs"
    if row["market_type"] == "PITCHER_WALKS":
        return f"{pshort} ({team}) {side[0]}{alt:g} BB"
    if row["market_type"] == "PITCHER_WIN":
        lab = "Win YES" if side == "YES" else "Win NO"
        return f"{pshort} ({team}) {lab}"
    return f"{cat}: {team or pshort}"

# ------------------- Pool builder (robust) -------------------

def build_pool(
    odds_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    pitchers_df: pd.DataFrame,
) -> pd.DataFrame:
    """Unify the DK board (ML/RL + pitchers) into the app's working table."""
    if odds_df is None or odds_df.empty:
        return pd.DataFrame(columns=[
            "date","game_id","market_type","side","alt_line","american_odds","decimal_odds",
            "p_market","q_model","team_abbr","player_name","player_id","player_key",
            "home_abbr","away_abbr","matchup","description"
        ])

    df = odds_df.copy()

    # Basic types
    for c in ["american_odds","alt_line"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Ensure decimal odds
    if "decimal_odds" not in df.columns:
        df["decimal_odds"] = df["american_odds"].apply(american_to_decimal)

    # Attach schedule for matchup text
    if schedule_df is not None and not schedule_df.empty:
        sch = schedule_df[["game_id","home_abbr","away_abbr"]].drop_duplicates()
        df = df.merge(sch, on="game_id", how="left")
    if "matchup" not in df.columns:
        df["matchup"] = df.apply(lambda r: make_matchup(r.get("away_abbr",""), r.get("home_abbr","")), axis=1)

    # Attach team to players from probable pitchers (stable)
    if pitchers_df is not None and not pitchers_df.empty:
        pcols = [c for c in ["player_key","team_abbr","game_id"] if c in pitchers_df.columns]
        if "player_key" in df.columns and pcols:
            df = df.merge(pitchers_df[pcols].drop_duplicates(), on="player_key", how="left", suffixes=("","_pp"))
            # Prefer pitchers_df team when present
            df["team_abbr"] = df["team_abbr"].fillna(df.get("team_abbr_pp"))

    # Safe team format
    if "team_abbr" in df.columns:
        df["team_abbr"] = df["team_abbr"].astype(str).str.upper().str[:3]

    # --- Build fair (vig-free) market probabilities ---
    df["p_market"] = np.nan

    # Moneyline (two-way per game)
    ml = df["market_type"] == "MONEYLINE"
    if ml.any():
        for gid, grp in df[ml].groupby("game_id"):
            # Expect two rows (home/away)
            if len(grp) >= 2:
                decs = grp["decimal_odds"].values[:2]
                p0, p1 = fair_two_way(decs[0], decs[1])
                idxs = grp.index[:2]
                df.loc[idxs[0], "p_market"] = p0
                df.loc[idxs[1], "p_market"] = p1

    # Run line / alt run line (two-way per game & alt_line)
    rl = df["market_type"].isin(["RUN_LINE","ALT_RUN_LINE"])
    if rl.any():
        for (gid, alt), grp in df[rl].groupby(["game_id","alt_line"], dropna=False):
            if len(grp) >= 2:
                decs = grp["decimal_odds"].values[:2]
                p0, p1 = fair_two_way(decs[0], decs[1])
                idxs = grp.index[:2]
                df.loc[idxs[0], "p_market"] = p0
                df.loc[idxs[1], "p_market"] = p1

    # Player props (two-way per player & line)
    is_prop = df["market_type"].isin(["PITCHER_KS","PITCHER_OUTS","PITCHER_WALKS"])
    if is_prop.any():
        for (gid, pkey, alt, mkt), grp in df[is_prop].groupby(["game_id","player_key","alt_line","market_type"], dropna=False):
            if "OVER" in grp["side"].values and "UNDER" in grp["side"].values:
                g_over = grp[grp["side"] == "OVER"].iloc[0]
                g_under = grp[grp["side"] == "UNDER"].iloc[0]
                p_o, p_u = fair_two_way(g_over["decimal_odds"], g_under["decimal_odds"])
                df.loc[g_over.name, "p_market"] = p_o
                df.loc[g_under.name, "p_market"] = p_u

    # Pitcher win (YES/NO)
    is_win = df["market_type"] == "PITCHER_WIN"
    if is_win.any():
        for (gid, pkey, mkt), grp in df[is_win].groupby(["game_id","player_key","market_type"], dropna=False):
            if "YES" in grp["side"].values and "NO" in grp["side"].values:
                y = grp[grp["side"] == "YES"].iloc[0]
                n = grp[grp["side"] == "NO"].iloc[0]
                p_y, p_n = fair_two_way(y["decimal_odds"], n["decimal_odds"])
                df.loc[y.name, "p_market"] = p_y
                df.loc[n.name, "p_market"] = p_n

    # Defaults if still NaN
    df["p_market"] = df["p_market"].fillna(df["decimal_odds"].apply(implied_from_decimal)).clip(0.01, 0.99)

    # Start model q as market prob (will be overridden by features)
    df["q_model"] = df["p_market"].astype(float)

    # Display columns
    if "player_name" not in df.columns:
        df["player_name"] = ""
    df["description"] = df.apply(describe_leg, axis=1)

    # De-duplicate noisy rows (same leg multiple times)
    dedup_keys = ["game_id","team_abbr","player_key","market_type","side","alt_line","american_odds","description"]
    df = df.drop_duplicates(subset=dedup_keys).reset_index(drop=True)

    return df


# ------------------ UI helpers ------------------

def css_inject():
    st.markdown(
        """
        <style>
          .stApp { font-family: -apple-system, BlinkMacSystemFont, Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
          section[data-testid="stSidebar"] { width: 360px !important; }
          div[data-testid="stMetricValue"] { font-weight: 600; }
          .note { color: #6b7280; font-size: 0.87rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def chips_multiselect(label: str, options: List[str], default: List[str]) -> List[str]:
    return st.multiselect(label, options, default=default, label_visibility="visible")


def odds_slider():
    return st.slider(
        "American odds",
        min_value=-700, max_value=700, value=(-700, 700), step=1
    )

# ------------------ Parlay presets (variety) ------------------

def choose_presets(pool: pd.DataFrame) -> pd.DataFrame:
    """
    Build Low/Medium/High presets across 4,5,6,8 legs with basic variety:
      - Avoid duplicate players.
      - Limit to one leg per game for ML/RL.
      - Prefer high q, safer odds for Low; mix for Medium; go spicy for High.
    """
    if pool.empty:
        return pd.DataFrame(columns=["Legs","Type","Decimal","~Est Hit","Meets +600?","Picks"])

    def pick_rows(df, legs, flavor):
        if df.empty:
            return []
        df = df.copy()
        # Scoring different by flavor
        if flavor == "Low":
            df["score"] = df["q_model"] * (1.2 - df["decimal_odds"].clip(1.0, 2.5) / 2.5)
        elif flavor == "Medium":
            df["score"] = df["q_model"] * 0.8 + (df["decimal_odds"] - 1.0) * 0.2
        else:  # High
            df["score"] = (df["decimal_odds"] - 1.0) * 0.65 + df["q_model"] * 0.35

        # Sort by score descending
        df = df.sort_values("score", ascending=False)

        picks = []
        seen_players = set()
        seen_games = set()

        for _, r in df.iterrows():
            # Avoid repeats:
            if r.get("player_key"):
                if r["player_key"] in seen_players:
                    continue
            if r["market_type"] in {"MONEYLINE","RUN_LINE","ALT_RUN_LINE"}:
                gid = r.get("game_id")
                if pd.notna(gid) and gid in seen_games:
                    continue

            picks.append(r)
            if r.get("player_key"):
                seen_players.add(r["player_key"])
            if r["market_type"] in {"MONEYLINE","RUN_LINE","ALT_RUN_LINE"} and pd.notna(r.get("game_id")):
                seen_games.add(r["game_id"])

            if len(picks) == legs:
                break

        return picks

    rows = []
    legs_options = [4,5,6,8]
    types = ["Low","Medium","High"]
    for L in legs_options:
        for t in types:
            subset = pool.copy()
            if t == "Low":
                subset = subset[subset["q_model"] >= 0.60]
                subset = subset[subset["american_odds"] <= -100]  # safer
            elif t == "Medium":
                subset = subset[subset["q_model"] >= 0.55]
            else:
                subset = subset[subset["american_odds"] >= -300]  # allow plus money and mild minus

            picks = pick_rows(subset, L, t)
            if not picks or len(picks) < L:
                continue

            picks_df = pd.DataFrame(picks)
            dec = float(np.prod(picks_df["decimal_odds"]))
            hit = float(np.prod(picks_df["q_model"]))
            meets = dec >= 7.0  # ≈ +600 American
            rows.append({
                "Legs": L,
                "Type": t,
                "Decimal": round(dec, 3),
                "~Est Hit": round(hit, 3),
                "Meets +600?": meets,
                "Picks": " | ".join(picks_df["description"].tolist())
            })

    return pd.DataFrame(rows)


# --------------------------- App ---------------------------

st.set_page_config(page_title="MLB Parlay Picker -- MVP", page_icon="⚾", layout="wide")
css_inject()
st.title("MLB Parlay Picker -- MVP")
st.caption("DraftKings lines • Moneyline / Run Lines / Pitcher Props • Vig‑free market probs + optional true projections")

# Sidebar -- Cloud mode & Upload
with st.expander("How Cloud Mode works (tap to expand)", expanded=False):
    st.markdown(
        """
        **Cloud Mode** pulls today's board directly from DraftKings:
        - Game lines (Moneyline, Run Line)
        - Pitcher props (Ks, Outs, Walks, Win)

        You can also upload:
        1) a DK markets CSV (from Cloud Mode's **Download DK CSV**), and  
        2) an optional **Features CSV** to apply your projections (columns like
           `player_name/team_abbr/market_type/side/alt_line/q_proj`).
        """
    )

# ---- Controls row ----
cloud = st.toggle("Enable Cloud Mode (auto‑fetch today)", value=False)

today_str = date.today().strftime("%Y-%m-%d")
odds_df = pd.DataFrame()
feat_auto = pd.DataFrame()
schedule_df = pd.DataFrame()
pitchers_df = pd.DataFrame()
dk_bytes = b""
feat_bytes = b""

if cloud:
    try:
        odds_df, feat_auto, dk_bytes, feat_bytes, schedule_df, pitchers_df = cloud_fetch(today_str)
        st.success(f"Fetched DraftKings board for {today_str}.")
        colA, colB, colC = st.columns([1,1,3])
        with colA:
            st.download_button("Download DK CSV", data=dk_bytes, file_name=f"dk_board_{today_str}.csv", type="primary")
        with colB:
            st.download_button("Download Features CSV (auto)", data=feat_bytes, file_name=f"features_{today_str}.csv")
    except Exception as e:
        st.error(f"Cloud fetch failed: {e}")

# Uploads
st.subheader("Upload (optional)")
c1, c2 = st.columns(2)
with c1:
    dk_csv = st.file_uploader("Upload DK Markets CSV", type=["csv"], help="Use the 'Download DK CSV' above as a template")
with c2:
    feat_csv = st.file_uploader("Upload Features CSV (enables true projections)", type=["csv"])

# Load uploaded DK CSV if present (overrides Cloud board)
if dk_csv is not None:
    try:
        odds_df = pd.read_csv(dk_csv)
        st.info("Loaded DK CSV from upload.")
    except Exception as e:
        st.error(f"Failed to read DK CSV: {e}")

# Load features (uploaded or auto)
feat_df_raw = pd.DataFrame()
if feat_csv is not None:
    try:
        feat_df_raw = pd.read_csv(feat_csv)
        st.info("Loaded Features CSV from upload.")
    except Exception as e:
        st.error(f"Failed to read Features CSV: {e}")
elif not feat_auto.empty:
    feat_df_raw = feat_auto.copy()

# Build base pool
pool_base = build_pool(odds_df, feat_df_raw, schedule_df, pitchers_df)

# Attach projections (robust)
feat_norm = normalize_features(feat_df_raw)
pool, applied = attach_projections(pool_base, feat_norm)
st.success(f"Applied true projections to {applied:,} legs.")

# -------- Filters (Apple-simple) ---------
st.header("Filters")
# Select all games
all_games = sorted(pool["matchup"].dropna().unique().tolist())
sel_all_games = st.checkbox("Select all games", value=True)
games_selected = all_games if sel_all_games else st.multiselect("Games", all_games, default=all_games)

# Select all teams
all_teams = sorted(pd.unique(pool["team_abbr"].dropna().astype(str).str.upper()))
sel_all_teams = st.checkbox("Select all teams", value=True)
teams_selected = all_teams if sel_all_teams else st.multiselect("Teams", all_teams, default=all_teams)

# Category chips
all_cats = [CAT_DISPLAY.get(c, c) for c in sorted(pool["market_type"].dropna().unique())]
default_cats = all_cats[:]  # show all by default
cat_disp_selected = chips_multiselect("Categories", all_cats, default_cats)
cat_selected = {k for k, v in CAT_DISPLAY.items() if v in cat_disp_selected} | {
    c for c in pool["market_type"].unique() if CAT_DISPLAY.get(c, c) in cat_disp_selected
}

# Odds range
amin, amax = odds_slider()

# ---------- Apply filters ----------
filt = (
    pool["matchup"].isin(games_selected) &
    pool["team_abbr"].astype(str).str.upper().isin(teams_selected) &
    pool["market_type"].isin(cat_selected) &
    pool["american_odds"].between(amin, amax)
)
cand = pool[filt].copy().reset_index(drop=True)

with st.expander("Coverage (from MLB schedule) -- games detected", expanded=False):
    games_debug = ", ".join(all_games)
    st.caption(games_debug or "No games detected.")

# ---------- Tabs ----------
tabs = st.tabs(["Candidates", "Top 20", "Parlay Presets", "Alt Line Safety Board", "One‑Tap Ticket", "ML Winners & Alt RL Locks"])

# Candidates
with tabs[0]:
    if cand.empty:
        st.info("No legs match the current filters.")
    else:
        show_cols = ["description","market_type","american_odds","decimal_odds","q_model","p_market","edge","ev","matchup","team_abbr","player_name","alt_line"]
        colcfg = {
            "description": st.column_config.TextColumn("Leg"),
            "market_type": st.column_config.TextColumn("Category"),
            "american_odds": st.column_config.NumberColumn("Odds"),
            "decimal_odds": st.column_config.NumberColumn("Dec"),
            "q_model": st.column_config.NumberColumn("q (model)", format="%.3f"),
            "p_market": st.column_config.NumberColumn("q (market)", format="%.3f"),
            "edge": st.column_config.NumberColumn("Edge", format="%.3f"),
            "ev": st.column_config.NumberColumn("EV", format="%.3f"),
            "matchup": st.column_config.TextColumn("Game"),
            "team_abbr": st.column_config.TextColumn("Team"),
            "player_name": st.column_config.TextColumn("Player"),
            "alt_line": st.column_config.NumberColumn("Line"),
        }
        cand = cand.assign(
            edge=(cand["q_model"] - cand["p_market"]).round(4),
            ev=(cand["q_model"] * cand["decimal_odds"] - 1.0).round(4),
        )
        st.dataframe(
            cand[show_cols].sort_values(["q_model","edge","decimal_odds"], ascending=[False,False,True]),
            use_container_width=True, height=520, column_config=colcfg
        )

# Top 20
with tabs[1]:
    if cand.empty:
        st.info("No legs available for ranking.")
    else:
        # Rank by (high q, then value edge, then safer odds)
        top = cand.copy()
        top["rank_score"] = top["q_model"] * 1.1 + (top["q_model"] - top["p_market"]) * 0.7 - (top["decimal_odds"] - 1.0) * 0.05
        top = top.sort_values("rank_score", ascending=False).head(20)

        for i, row in enumerate(top.itertuples(index=False), start=1):
            odds = f"{int(row.american_odds):+d}" if pd.notna(row.american_odds) else "--"
            ev = f"{row.ev:+.2f}"
            edge = f"{(row.q_model - row.p_market):+.1%}"
            qline = f"**Odds:** {odds}  |  **Model q:** {row.q_model:.1%}  |  **Market:** {row.p_market:.1%}  |  **Edge:** {edge}  |  **EV:** {ev}"
            st.markdown(f"**{i}. {row.description}**")
            st.caption(qline)
            st.markdown("---")

# Parlay presets
with tabs[2]:
    presets = choose_presets(cand)
    if presets.empty:
        st.info("No presets available under current filters.")
    else:
        st.dataframe(presets, use_container_width=True, height=460)

# Alt Line Safety Board (very high q, lower payout)
with tabs[3]:
    safety = cand.copy()
    safety = safety[
        (safety["q_model"] >= 0.65) &
        (safety["american_odds"] <= -150)
    ].sort_values(["q_model","decimal_odds"], ascending=[False,True]).head(40)
    if safety.empty:
        st.info("No high‑confidence alt lines at the moment.")
    else:
        show_cols = ["description","american_odds","decimal_odds","q_model","p_market","edge","ev"]
        colcfg = {
            "american_odds": st.column_config.NumberColumn("Odds"),
            "decimal_odds": st.column_config.NumberColumn("Dec"),
            "q_model": st.column_config.NumberColumn("q", format="%.3f"),
            "p_market": st.column_config.NumberColumn("Market", format="%.3f"),
            "edge": st.column_config.NumberColumn("Edge", format="%.3f"),
            "ev": st.column_config.NumberColumn("EV", format="%.3f"),
        }
        st.dataframe(
            safety.assign(
                edge=(safety["q_model"] - safety["p_market"]).round(4),
                ev=(safety["q_model"] * safety["decimal_odds"] - 1.0).round(4),
            )[show_cols],
            use_container_width=True, height=460, column_config=colcfg
        )

# One‑Tap Ticket (builds a single diversified 4–6 leg ticket)
with tabs[4]:
    base = cand.copy()
    base = base.sort_values(["q_model","decimal_odds"], ascending=[False,True])
    ticket = choose_presets(base)
    ticket = ticket[ticket["Legs"].isin([4,5,6])].sort_values(["~Est Hit","Decimal"], ascending=[False,True]).head(1)
    if ticket.empty:
        st.info("No one‑tap ticket currently meets the filters.")
    else:
        t = ticket.iloc[0]
        st.metric("One‑Tap Ticket", f"{int(t['Legs'])} legs · {t['Type']}")
        st.caption(f"≈Hit {t['~Est Hit']:.1%} · Dec {t['Decimal']:.2f} · Meets +600? {'✅' if t['Meets +600?'] else '❌'}")
        st.write(t["Picks"])

# ML Winners & Alt RL Locks
with tabs[5]:
    ml = cand[cand["market_type"] == "MONEYLINE"].copy()
    if ml.empty:
        st.info("No moneylines under current filters.")
    else:
        # Best side per game by q_model
        ml["game_rank"] = ml.groupby("game_id")["q_model"].rank(ascending=False, method="first")
        winners = ml[ml["game_rank"] == 1.0].copy()
        winners["Lock?"] = winners["q_model"] >= 0.65
        winners_disp = winners[["description","american_odds","decimal_odds","q_model","p_market","Lock?","matchup","team_abbr"]]
        colcfg = {
            "american_odds": st.column_config.NumberColumn("Odds"),
            "decimal_odds": st.column_config.NumberColumn("Dec"),
            "q_model": st.column_config.NumberColumn("q", format="%.3f"),
            "p_market": st.column_config.NumberColumn("Market", format="%.3f"),
            "Lock?": st.column_config.CheckboxColumn("ML Lock"),
        }
        st.dataframe(winners_disp.sort_values(["Lock?","q_model","decimal_odds"], ascending=[False,False,True]),
                     use_container_width=True, height=460, column_config=colcfg)