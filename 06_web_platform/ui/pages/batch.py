# -*- coding: utf-8 -*-
"""Module 2 — High-throughput batch screening."""
import pandas as pd
import streamlit as st

from model_engine import predict_batch_fast
from ui import common as C

C.page_header(
    "Batch Screening",
    f"Screen up to {C.MAX_BATCH} chemicals at once. Paste CAS numbers or SMILES (one per line) or upload a .txt/.csv file; "
    "results can be downloaded as CSV.",
)

DEFAULT = "\n".join(["106-44-5", "108-39-4", "84-74-2", "1163-19-5", "84852-53-9", "110-00-9", "638-16-4", "10595-49-0"])
ENVS = {"Digested sludge · 35 °C": 1, "Sediment & soil · 20 °C": 0}

with st.container(border=True):
    c1, c2 = st.columns([1.6, 1], gap="large")
    with c1:
        txt = st.text_area("Chemicals (one per line)", value=DEFAULT, height=190)
        up = st.file_uploader("…or upload a file (first column or one query per line)", type=["txt", "csv"])
    with c2:
        env = st.radio("Anaerobic environment", list(ENVS))
        dur = st.slider("Incubation duration (days)", 7, 180, 56, 7)
        run = st.button("Run screening", type="primary")

if run:
    raw = txt.splitlines()
    if up is not None:
        try:
            raw = up.getvalue().decode("utf-8-sig", errors="ignore").splitlines()
        except Exception:
            st.error("The uploaded file could not be read.")
            st.stop()
    queries = []
    for line in raw:
        line = line.strip().split(",")[0].strip().strip('"')
        if line and line.lower() not in ("smiles", "cas", "cas_rn", "query"):
            queries.append(line)
    if not queries:
        st.warning("No chemicals found.")
        st.stop()
    if len(queries) > C.MAX_BATCH:
        st.warning(f"Only the first {C.MAX_BATCH} of {len(queries)} entries are screened.")
        queries = queries[: C.MAX_BATCH]
    bad = []
    for q in queries:
        try:
            C.clean_query(q)
        except ValueError:
            bad.append(q[:40])
    if bad:
        st.error(f"{len(bad)} entr{'y' if len(bad) == 1 else 'ies'} not valid (e.g. {bad[0]!r}). Remove spaces / unsupported characters and retry.")
        st.stop()
    with st.spinner(f"Screening {len(queries)} molecules…"):
        try:
            results = predict_batch_fast(queries, float(dur), ENVS[env])
        except Exception as e:
            st.error("Screening failed for this input.")
            C.report_error(e)
            st.stop()
    rows = [
        {
            "Query": r["query"],
            "CAS_RN": r["cas_rn"],
            "Chemical_Name": r["chemical_name"],
            "SMILES": r["canonical_smiles"],
            "Pred_Degrad_pct": r["predicted_degrad_pct"],
            "Exp_pct": r["experimental_degrad_pct"],
            "Pred_DT50_d": r["predicted_dt50_days"],
            "Pred_Lag_d": r["predicted_lag_days"],
            "BIOWIN7": r["epa_biowin7_prob"],
            "Outcome": r["classification_outcome"],
            "Applicability_domain": r["applicability_domain"],
            "Physics_gates": "; ".join(r["physics_gates_triggered"]),
        }
        for r in results
    ]
    st.session_state["batch_df"] = pd.DataFrame(rows)
    st.session_state["batch_meta"] = f"{env}, {dur} d"

df = st.session_state.get("batch_df")
if df is not None:
    n = len(df)
    ready = int((df["Pred_Degrad_pct"] >= 60).sum())
    inhib = int((df["Pred_Degrad_pct"] < 0).sum())
    out_dom = int((~df["Applicability_domain"].astype(str).str.lower().str.contains("in-domain")).sum())
    C.kpi_row([
        ("Screened", str(n), st.session_state.get("batch_meta", ""), ""),
        ("Readily biodegradable", str(ready), "≥ 60% ThGP", "ok"),
        ("Inhibition alerts", str(inhib), "negative % ThGP", "bad" if inhib else ""),
        ("Outside domain", str(out_dom), "interpret with care", "warn" if out_dom else ""),
    ])
    cfg = {
        "Pred_Degrad_pct": st.column_config.NumberColumn("Pred. % ThGP", format="%.1f"),
        "Exp_pct": st.column_config.NumberColumn("Exp. %", format="%.1f"),
        "Pred_DT50_d": st.column_config.NumberColumn("DT50 (d)", format="%.1f"),
        "Pred_Lag_d": st.column_config.NumberColumn("Lag (d)", format="%.1f"),
        "BIOWIN7": st.column_config.NumberColumn("BIOWIN7", format="%.3f"),
    }
    C.show_df(df, column_config=cfg)
    st.download_button("Download report (CSV)", df.to_csv(index=False).encode("utf-8-sig"), "AnaeroBio_batch_report.csv", "text/csv")
