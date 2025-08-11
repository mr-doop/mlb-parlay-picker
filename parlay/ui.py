# parlay/ui.py
from __future__ import annotations
from typing import List
import streamlit as st

GLOBAL_CSS = """
<style>
:root {
  --card-bg: #ffffff;
  --card-border: #e9edf3;
  --chip-bg: #f3f6fb;
  --chip-tx: #233143;
}
.block-title { font-weight: 700; font-size: 1.05rem; margin: 4px 0 8px 0; }
.card { border: 1px solid var(--card-border); border-radius: 14px; padding: 10px 12px;
        background: var(--card-bg); box-shadow: 0 1px 2px rgba(0,0,0,0.03); margin-bottom: 8px; }
.row { display: flex; gap: 8px; flex-wrap: wrap; }
.chip { display: inline-block; padding: 2px 8px; border-radius: 999px;
        background: var(--chip-bg); color: var(--chip-tx); font-size: 12px; margin-right: 6px; }
.pick { font-weight: 600; }
.sub { color: #5a6b7f; font-size: 12px; }
.small-muted { color: #7a8898; font-size: 12px; }
.parlay-title { font-weight: 700; margin: 2px 0 6px 0; }
.parlay-row { border: 1px solid var(--card-border); border-radius: 12px; padding: 8px 10px; margin-bottom: 8px; }
.parlay-meta { display:flex; gap:8px; flex-wrap: wrap; margin-top: 6px; }
</style>
"""

def inject_css():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

def chip(text: str) -> str:
    return f'<span class="chip">{text}</span>'

def mini_leg_label(row) -> str:
    odds = int(row.get("american_odds", 0))
    q = float(row.get("q_model", row.get("p_market", 0.5)))
    return f'<div class="pick">{row.get("description","")}</div><div class="sub">Odds {odds} • q {q:.0%}</div>'

def parlay_card(title: str, picks: List, decimal: float, q_est: float, meets: bool):
    am = int(round((decimal - 1) * 100))
    checks = chip(f"Dec {decimal:.2f}") + chip(f"~Hit {q_est:.0%}") + chip(f"{'Meets +600' if meets else 'Below +600'}")
    items = "".join([f'<div class="parlay-row">{mini_leg_label(p)}</div>' for p in picks])
    html = f"""
    <div class="card">
      <div class="parlay-title">{title}</div>
      <div class="parlay-meta">{checks}</div>
      <div style="height:6px"></div>
      {items}
      <div class="small-muted">Est. American ≈ {'+' if am>=0 else ''}{am}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)