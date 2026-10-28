# -*- coding: utf-8 -*-
"""Module 3 — MicroPath Explorer (observed microbial transformation products)."""
import pandas as pd
import streamlit as st

from model_engine import METAB_PATH, predict_batch_fast
from ui import common as C

C.page_header(
    "MicroPath Explorer",
    "Observed microbial transformation products of a parent compound. Each metabolite is run through AnBioGate so you can "
    "see whether degradation leads to more or less biodegradable products.",
)


@st.cache_data(show_spinner=False)
def load_pathways():
    df = pd.read_csv(METAB_PATH)
    df = df.dropna(subset=["parent_smiles", "child_smiles"])
    df["remark"] = df["remark"].fillna("") if "remark" in df.columns else ""
    return df[["parent_smiles", "child_smiles", "remark"]].drop_duplicates()


df = load_pathways()
top = df["parent_smiles"].value_counts().head(40)
options = [f"{smi}  ({n} products)" for smi, n in top.items()]

c1, c2 = st.columns([1, 1], gap="large")
with c1:
    pick = st.selectbox("Most-documented parent compounds", options, index=min(4, len(options) - 1))
with c2:
    parent = st.text_input("…or enter a parent SMILES", value=pick.split("  (")[0], max_chars=C.MAX_QUERY_LEN)

parent = parent.strip()
hits = df[df["parent_smiles"] == parent].drop_duplicates(subset=["child_smiles"])
st.markdown(f"**{len(hits)}** observed transformation product(s) for `{parent[:120]}`")

if hits.empty:
    st.info("No observed products for this exact SMILES. Use the canonical SMILES from the Predictor page.")
    st.stop()

hits = hits.head(C.MAX_BATCH)
kids = hits["child_smiles"].tolist()
with st.spinner("Predicting metabolite fate…"):
    try:
        preds = predict_batch_fast(kids, 56.0, 1)
    except Exception as e:
        st.error("Could not evaluate these metabolites.")
        C.report_error(e)
        st.stop()

out = pd.DataFrame(
    {
        "Metabolite_SMILES": kids,
        "Name": [p["chemical_name"] for p in preds],
        "Pred_Degrad_pct": [p["predicted_degrad_pct"] for p in preds],
        "DT50_d": [p["predicted_dt50_days"] for p in preds],
        "BIOWIN7": [p["epa_biowin7_prob"] for p in preds],
        "Status": [p["classification_outcome"] for p in preds],
        "Remark": hits["remark"].tolist(),
    }
)
try:
    parent_pred = C.predict(C.clean_query(parent), 56.0, 1, shap=False)
except Exception:
    parent_pred = None
if parent_pred:
    C.kpi_row([
        ("Parent % ThGP", f"{parent_pred['predicted_degrad_pct']:.1f}%", parent_pred["chemical_name"], C.tone_for(parent_pred["predicted_degrad_pct"])),
        ("Products", str(len(out)), "observed", ""),
        ("Median product % ThGP", f"{out['Pred_Degrad_pct'].median():.1f}%", "predicted", C.tone_for(float(out["Pred_Degrad_pct"].median()))),
        ("Products ≥ 60%", str(int((out["Pred_Degrad_pct"] >= 60).sum())), "readily biodegradable", "ok"),
    ])
C.show_df(
    out,
    column_config={
        "Pred_Degrad_pct": st.column_config.NumberColumn("Pred. % ThGP", format="%.1f"),
        "DT50_d": st.column_config.NumberColumn("DT50 (d)", format="%.1f"),
        "BIOWIN7": st.column_config.NumberColumn("BIOWIN7", format="%.3f"),
    },
)
st.download_button("Download table (CSV)", out.to_csv(index=False).encode("utf-8-sig"), "AnaeroBio_micropath.csv", "text/csv")
