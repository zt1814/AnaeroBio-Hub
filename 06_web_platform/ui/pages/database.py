# -*- coding: utf-8 -*-
"""Module 4 — Benchmark database browser."""
import sqlite3

import pandas as pd
import streamlit as st

from model_engine import DB_PATH
from ui import common as C

C.page_header(
    "Benchmark Database",
    "The 546 unique curated molecules behind AnBioGate: experimental anaerobic degradation, DT50, lag, BIOWIN and physicochemical descriptors.",
)

COLS = ("CAS_RN, Chemical_Name, SMILES, Mean_Anaerobic_Degrad_Pct, Mean_Anaerobic_DT50_d, Lit_Lag_days, "
        "Biowin7_Anaerobic_Prob, Phys_logKow, Best_Acidic_pKa, Tiers")


@st.cache_data(show_spinner=False, ttl=600)
def load_master():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    try:
        return pd.read_sql(f"SELECT {COLS} FROM unified_molecules_master", conn)
    finally:
        conn.close()


df = load_master()

c1, c2, c3 = st.columns([2, 1.2, 1.2], gap="large")
with c1:
    kw = st.text_input("Search CAS, name or SMILES", max_chars=100, placeholder="e.g. phenol, 106-44-5, c1ccccc1")
with c2:
    outcome = st.selectbox("Outcome", ["All", "Readily biodegradable (≥ 60%)", "Partial (0–60%)", "Inhibitory / none (< 0%)"])
with c3:
    only_dt50 = st.checkbox("Has measured DT50", value=False)

view = df
if kw.strip():
    hay = (view["CAS_RN"].astype(str) + " " + view["Chemical_Name"].astype(str) + " " + view["SMILES"].astype(str))
    view = view[hay.str.contains(kw.strip(), case=False, na=False, regex=False)]
pct = view["Mean_Anaerobic_Degrad_Pct"]
if outcome.startswith("Readily"):
    view = view[pct >= 60]
elif outcome.startswith("Partial"):
    view = view[(pct >= 0) & (pct < 60)]
elif outcome.startswith("Inhibitory"):
    view = view[pct < 0]
if only_dt50:
    view = view[view["Mean_Anaerobic_DT50_d"].notna()]

C.kpi_row([
    ("Showing", f"{len(view)}", f"of {len(df)} molecules", ""),
    ("Median % ThGP", f"{view['Mean_Anaerobic_Degrad_Pct'].median():.1f}%" if len(view) and view["Mean_Anaerobic_Degrad_Pct"].notna().any() else "—", "measured", ""),
    ("With DT50", f"{int(view['Mean_Anaerobic_DT50_d'].notna().sum())}", "measured half-lives", ""),
    ("With lag", f"{int(view['Lit_Lag_days'].notna().sum())}", "measured lag times", ""),
])

C.show_df(
    view,
    column_config={
        "Mean_Anaerobic_Degrad_Pct": st.column_config.NumberColumn("Degradation (% ThGP)", format="%.1f"),
        "Mean_Anaerobic_DT50_d": st.column_config.NumberColumn("DT50 (d)", format="%.1f"),
        "Lit_Lag_days": st.column_config.NumberColumn("Lag (d)", format="%.1f"),
        "Biowin7_Anaerobic_Prob": st.column_config.NumberColumn("BIOWIN7", format="%.3f"),
        "Phys_logKow": st.column_config.NumberColumn("log Kow", format="%.2f"),
        "Best_Acidic_pKa": st.column_config.NumberColumn("pKa", format="%.2f"),
    },
)
st.download_button("Download filtered table (CSV)", view.to_csv(index=False).encode("utf-8-sig"), "AnaeroBio_benchmark.csv", "text/csv")

with st.expander("Distribution of measured degradation"):
    vals = df["Mean_Anaerobic_Degrad_Pct"].dropna()
    bins = pd.cut(vals, bins=[-101, 0, 20, 40, 60, 80, 101], labels=["< 0", "0–20", "20–40", "40–60", "60–80", "80–100+"])
    st.bar_chart(bins.value_counts().reindex(["< 0", "0–20", "20–40", "40–60", "60–80", "80–100+"]))
