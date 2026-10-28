#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import urllib.parse
import sqlite3
import datetime
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import model_engine
importlib.reload(model_engine)
from model_engine import predict_molecule, train_and_save_bundle, load_bundle, DB_PATH, METAB_PATH

st.set_page_config(
    page_title="AnaeroBio-Hub v4.0 | Multi-Task Physics-Guided AI Platform",
    page_icon="🧪",
    layout="wide"
)

bundle = load_bundle()

st.markdown("""
<style>
.main-banner {
    background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
    color: white;
    padding: 20px 28px;
    border-radius: 10px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="main-banner">
    <h2 style="margin:0; color:white;">🧪 AnaeroBio-Hub v4.0: Multi-Task Physics-Guided AI Platform for Anaerobic Fate & Inhibition</h2>
    <p style="margin:6px 0 0 0; opacity:0.92; font-size:15px;">
        <b>546 Golden Benchmark Molecules</b> • <b>78,530 Indexed EPI Priors</b> • <b>55,956 Exp. pKa/Hydrolysis</b> • <b>28,088 Microbial Pathways</b> | Active Model: <code>{bundle['version']}</code>
    </p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "🔍 1. Multi-Task Inspector & SHAP Waterfall",
    "📦 2. Batch High-Throughput Screening",
    "🧬 3. MicroPath Metabolite Explorer",
    "📚 4. 546-Molecule Golden Database",
    "🌐 5. Sandbox Crowdsourcing & OpenAPI Hub"
])

with tabs[0]:
    col_l, col_r = st.columns([1.0, 1.5])
    with col_l:
        st.subheader("⚙️ Input Chemical & Reactor Regime")
        preset_examples = {
            "p-Cresol (CAS 106-44-5 | BIOWIN7 False Negative -> Benzoyl-CoA Active)": "106-44-5",
            "m-Cresol (CAS 108-39-4 | BIOWIN7 False Negative -> Benzoyl-CoA Active)": "108-39-4",
            "Dibutyl phthalate (CAS 84-74-2 | Ester Hydrolysis -> Methanogenesis)": "84-74-2",
            "Decabromodiphenyl ether (CAS 1163-19-5 | BIOWIN7 False Positive -> Steric Cutoff)": "1163-19-5",
            "Furan (CAS 110-00-9 | Methanogenic Coenzyme F420 Inhibitor < 0%)": "110-00-9",
            "Triazine-trithione TMT (CAS 638-16-4 | Ni2+/Co2+ F430 Chelator < 0%)": "638-16-4",
            "Lauramidopropyl trimethylammonium (CAS 10595-49-0 | Cleavable Quat Surfactant)": "10595-49-0",
            "Custom Input (Type any SMILES or CAS below)": ""
        }
        sel = st.selectbox("Select Showcase Benchmark Molecule or Custom:", list(preset_examples.keys()))
        default_q = preset_examples[sel] if preset_examples[sel] else "106-44-5"
        query_input = st.text_input("Enter SMILES or CAS Registry Number:", value=default_q)
        env_choice = st.radio(
            "Select Anaerobic Environment Modality:",
            ["Digested Sludge (OECD 311 / ISO 11734 / BMP, 35°C High Biomass)", "Aquatic Sediment & Soil (OECD 307 / 308, 20°C)"],
            index=0
        )
        is_sludge = 1 if "Sludge" in env_choice else 0
        dur_days = st.slider("Incubation / Hydraulic Retention Duration (Days):", min_value=7, max_value=180, value=56, step=7)

        if query_input:
            res_tmp = predict_molecule(query_input, float(dur_days), is_sludge)
            smi_enc = urllib.parse.quote(res_tmp["canonical_smiles"])
            img_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smi_enc}/PNG?image_size=320x220"
            st.markdown("#### 🔬 2D Structure & Applicability Domain (AD)")
            c_img, c_ad = st.columns([1.05, 1.0])
            with c_img:
                st.image(img_url, caption=f"{res_tmp['chemical_name']} ({res_tmp['cas_rn'] or 'Custom'})")
            with c_ad:
                st.success(f"**AD Confidence:**\n{res_tmp['applicability_domain']}")
                st.info(f"**Prior DB Hit:**\n{'✅ 78.5k Indexed DB' if res_tmp['hit_90k_priors_db'] else '⚡ First-Principles Calc'}")

    with col_r:
        if query_input:
            res = predict_molecule(query_input, float(dur_days), is_sludge)
            st.subheader(f"📊 Multi-Task Diagnosis: `{res['chemical_name']}`")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Pred. Biodeg (% ThGP)", f"{res['predicted_degrad_pct']:.1f}%",
                      delta=f"Exp: {res['experimental_degrad_pct']:.1f}%" if res['experimental_degrad_pct'] is not None else "Custom")
            m2.metric("Pred. Half-Life DT50", f"{res['predicted_dt50_days']:.1f} d",
                      delta=f"Exp: {res['experimental_dt50_days']:.1f}d" if res['experimental_dt50_days'] is not None else None)
            m3.metric("Pred. Acclimation Lag", f"{res['predicted_lag_days']:.1f} d",
                      delta=f"Exp: {res['experimental_lag_days']:.1f}d" if res['experimental_lag_days'] is not None else None)
            m4.metric("US EPA BIOWIN 7", f"{res['epa_biowin7_prob']:.3f}")
            m5.metric("pH 7.2 Neutral Frac", f"{res['key_Modality_features']['Neutral_Frac_pH72']*100:.1f}%")

            if res["predicted_degrad_pct"] >= 60:
                st.success(f"**Classification Verdict:** {res['classification_outcome']} | **Canonical SMILES:** `{res['canonical_smiles']}`")
            elif res["predicted_degrad_pct"] < 0:
                st.error(f"**Classification Verdict:** {res['classification_outcome']} | **Canonical SMILES:** `{res['canonical_smiles']}`")
            else:
                st.warning(f"**Classification Verdict:** {res['classification_outcome']} | **Canonical SMILES:** `{res['canonical_smiles']}`")

            st.markdown("#### 🛡️ First-Principles Physics Gates & Mechanistic Alerts")
            if res["physics_gates_triggered"]:
                for g in res["physics_gates_triggered"]:
                    st.warning(f"⚡ **{g}**")
            else:
                st.success("✅ Standard anaerobic enzymatic mineralization regime.")

            # 使用出版级双联 Matplotlib 并排绘制：左图 SHAP 模态瀑布图 + 右图 Monod-Gompertz 矿化动力学曲线
            st.markdown("#### 🧩 Mechanistic Modality Attribution & Time-Course Mineralization Kinetics")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.4), dpi=160)
            plt.subplots_adjust(wspace=0.28)

            # 左图：模态归因瀑布图
            contribs = res["shap_modality_contributions"]
            sorted_items = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=False)
            labels = [k for k, _ in sorted_items]
            vals = [v for _, v in sorted_items]
            colors = ["#1abc9c" if v >= 0 else "#e74c3c" for v in vals]
            ax1.barh(labels, vals, color=colors, edgecolor="k", height=0.62, linewidth=0.5)
            ax1.axvline(0, color="#2c3e50", linewidth=0.9)
            ax1.set_xlabel("Modality Shift (% ThGP from Baseline)", fontsize=9, fontweight="bold")
            ax1.set_title("(a) Modality Attribution Waterfall", fontsize=9.5, fontweight="bold")
            ax1.tick_params(axis="both", labelsize=8)
            ax1.grid(True, axis="x", linestyle=":", alpha=0.5)

            # 右图：时间动力学曲线 (标注 Lag Time 与 60% 易降解阈值线)
            d_grid = [3, 7, 14, 21, 28, 42, 56, 70, 90, 120, 150, 180]
            k_preds = [predict_molecule(query_input, float(d), is_sludge)["predicted_degrad_pct"] for d in d_grid]
            ax2.plot(d_grid, k_preds, color="#16a085" if res["predicted_degrad_pct"] >= 0 else "#c0392b", marker="o", markersize=4, linewidth=2.0, label="Predicted Trajectory")
            ax2.axhline(60.0, color="#2980b9", linestyle="--", linewidth=0.9, label="Readily Biodeg. (60%)")
            ax2.axhline(0.0, color="#7f8c8d", linestyle=":", linewidth=0.8)
            ax2.axvline(res["predicted_lag_days"], color="#e67e22", linestyle="-.", linewidth=1.0, label=f"Lag = {res['predicted_lag_days']:.1f} d")
            ax2.scatter([dur_days], [res["predicted_degrad_pct"]], color="#e74c3c", s=55, zorder=5, label=f"Current ({dur_days}d)")
            ax2.set_xlim(0, 185)
            ax2.set_ylim(-35, 108)
            ax2.set_xlabel("Incubation Duration (Days)", fontsize=9, fontweight="bold")
            ax2.set_ylabel("Mineralization (% ThGP)", fontsize=9, fontweight="bold")
            ax2.set_title("(b) Simulated Mineralization Kinetics", fontsize=9.5, fontweight="bold")
            ax2.tick_params(axis="both", labelsize=8)
            ax2.legend(loc="lower right", fontsize=7.5, frameon=True)
            ax2.grid(True, linestyle=":", alpha=0.5)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

with tabs[1]:
    st.subheader("📦 High-Throughput Batch Multi-Task Screening (%ThGP + DT50 + Lag)")
    default_batch = "\n".join([
        "106-44-5", "108-39-4", "84-74-2", "1163-19-5",
        "84852-53-9", "110-00-9", "638-16-4", "10595-49-0"
    ])
    batch_txt = st.text_area("Paste CAS Registry Numbers or SMILES (one per line):", value=default_batch, height=150)
    if st.button("⚡ Execute Multi-Task Batch Screening", type="primary"):
        q_list = [x.strip() for x in batch_txt.splitlines() if x.strip()]
        rows = []
        for q in q_list:
            r = predict_molecule(q, 56.0, 1)
            rows.append({
                "Query": r["query"],
                "CAS_RN": r["cas_rn"],
                "Chemical_Name": r["chemical_name"],
                "SMILES": r["canonical_smiles"],
                "Pred_Degrad (%)": r["predicted_degrad_pct"],
                "Exp_Truth (%)": r["experimental_degrad_pct"],
                "Pred_DT50 (d)": r["predicted_dt50_days"],
                "Pred_Lag (d)": r["predicted_lag_days"],
                "EPA_BIOWIN7": r["epa_biowin7_prob"],
                "Outcome_Category": r["classification_outcome"],
                "AD_Confidence": r["applicability_domain"],
                "Physics_Gates": "; ".join(r["physics_gates_triggered"])
            })
        df_b_res = pd.DataFrame(rows)
        st.dataframe(df_b_res, use_container_width=True)
        st.download_button("📥 Download Multi-Task Batch Report (CSV)", df_b_res.to_csv(index=False).encode("utf-8-sig"), "AnaeroBio_v4_Batch_Report.csv", "text/csv")

with tabs[2]:
    st.subheader("🧬 MicroPath Explorer: 28,088 Observed Microbial Metabolic Transformations")
    df_metab = pd.read_csv(METAB_PATH)
    sample_parents = df_metab["parent_smiles"].dropna().unique()[:30].tolist()
    sel_p = st.selectbox("Select Parent Molecule SMILES:", sample_parents, index=4 if len(sample_parents)>4 else 0)
    custom_p = st.text_input("Or Enter Custom Parent SMILES:", value=sel_p)
    hits = df_metab[df_metab["parent_smiles"] == custom_p.strip()][["parent_smiles", "child_smiles", "remark"]].dropna(subset=["child_smiles"]).drop_duplicates()
    st.write(f"Found **{len(hits)}** observed microbial transformation products for `{custom_p}`:")
    if len(hits) > 0:
        enriched_m = []
        for _, mrow in hits.iterrows():
            c_smi = mrow["child_smiles"]
            c_pred = predict_molecule(c_smi, 56.0, 1)
            enriched_m.append({
                "Parent_SMILES": mrow["parent_smiles"],
                "Metabolite_SMILES": c_smi,
                "Metabolite_Name": c_pred["chemical_name"],
                "Metabolite_Pred_Pct (%)": c_pred["predicted_degrad_pct"],
                "Metabolite_DT50 (d)": c_pred["predicted_dt50_days"],
                "Metabolite_BIOWIN7": c_pred["epa_biowin7_prob"],
                "Metabolite_Status": c_pred["classification_outcome"]
            })
        st.dataframe(pd.DataFrame(enriched_m), use_container_width=True)

with tabs[3]:
    st.subheader("📚 546-Molecule Unified Anaerobic Biodegradation Golden Benchmark Database")
    conn = sqlite3.connect(DB_PATH)
    df_db = pd.read_sql("SELECT CAS_RN, Chemical_Name, SMILES, Mean_Anaerobic_Degrad_Pct, Mean_Anaerobic_DT50_d, Lit_Lag_days, Biowin7_Anaerobic_Prob, Phys_logKow, Best_Acidic_pKa, Tiers FROM unified_molecules_master", conn)
    conn.close()
    search_kw = st.text_input("Filter Database by CAS, Chemical Name, or SMILES Substructure:")
    df_show = df_db[df_db.astype(str).apply(lambda x: x.str.contains(search_kw, case=False, na=False)).any(axis=1)] if search_kw else df_db
    st.write(f"Showing **{len(df_show)}** / {len(df_db)} unique benchmark molecules:")
    st.dataframe(df_show, use_container_width=True)
    st.download_button("📥 Export Filtered Golden Benchmark (CSV)", df_show.to_csv(index=False).encode("utf-8-sig"), "AnaeroBio_Golden_Benchmark.csv", "text/csv")

with tabs[4]:
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.subheader("🛡️ Two-Stage Sandbox Community Data Submission")
        st.caption("Submissions enter the `pending` sandbox by default to prevent model poisoning. Authorized curators may enter their Curator Token for immediate approval & hot-retraining.")
        with st.form("contrib_form"):
            c_cas = st.text_input("CAS Registry Number (e.g., 120-80-9):")
            c_name = st.text_input("Chemical Name (e.g., Catechol):")
            c_smi = st.text_input("Canonical SMILES (Required, e.g., Oc1ccccc1O):")
            c_pct = st.number_input("Experimental Anaerobic Biodegradation (% ThGP):", min_value=-100.0, max_value=120.0, value=88.0)
            c_dur = st.number_input("Incubation Duration (Days):", min_value=1.0, max_value=365.0, value=56.0)
            c_doi = st.text_input("Reference DOI / Source:")
            c_token = st.text_input("Curator Token (Optional, for authorized instant approval):", type="password")
            sub_btn = st.form_submit_button("📥 Submit Experimental Record")
            if sub_btn and c_smi.strip():
                status_val = "approved" if c_token.strip() == os.getenv("ANAERO_ADMIN_TOKEN", "ANAERO2026") else "pending"
                conn = sqlite3.connect(DB_PATH)
                cur = conn.cursor()
                now_s = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cur.execute("""
                    INSERT INTO community_submissions
                    (cas_rn, chemical_name, smiles, exp_degrad_pct, duration_days, assay_guideline, inoculum_system, reference_doi, contributor, status, submitted_at)
                    VALUES (?, ?, ?, ?, ?, 'OECD 311 / BMP', 'Digested Sludge', ?, 'Researcher', ?, ?)
                """, (c_cas, c_name, c_smi.strip(), float(c_pct), float(c_dur), c_doi, status_val, now_s))
                sid = cur.lastrowid
                conn.commit()
                conn.close()
                if status_val == "approved":
                    new_b = train_and_save_bundle(f"v4.0.{sid}-live")
                    st.success(f"✅ Approved & Hot-Retrained to `{new_b['version']}`!")
                else:
                    st.info(f"🛡️ Record #{sid} staged in `pending` sandbox awaiting curator review.")

    with col_c2:
        st.subheader("💻 OpenAPI 3.0 Status & Sandbox Queue")
        st.markdown("- **Root Welcome Card**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)\n- **Interactive Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)")
        conn = sqlite3.connect(DB_PATH)
        df_sub_q = pd.read_sql("SELECT id, cas_rn, chemical_name, exp_degrad_pct, status, submitted_at FROM community_submissions ORDER BY id DESC LIMIT 10", conn)
        df_ver = pd.read_sql("SELECT * FROM model_versions ORDER BY trained_at DESC LIMIT 6", conn)
        conn.close()
        st.markdown("#### ⏳ Community Sandbox Queue (`pending` vs `approved`)")
        st.dataframe(df_sub_q, use_container_width=True)
        st.markdown("#### 📜 Model Version Ledger")
        st.dataframe(df_ver, use_container_width=True)
