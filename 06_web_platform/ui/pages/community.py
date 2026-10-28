# -*- coding: utf-8 -*-
"""Module 5 — Community sandbox (two-stage review: pending → approved + hot retraining)."""
import datetime
import re
import sqlite3

import pandas as pd
import streamlit as st

from model_engine import DB_PATH, train_and_save_bundle
from ui import common as C

C.page_header(
    "Community Sandbox",
    "Contribute experimental anaerobic data. Records enter a pending queue and are merged into the production model only after curator review, "
    "so no single submission can change predictions.",
)

if not C.PUBLIC_SUBMISSION:
    st.info("Public data submission is closed during peer review. You can still browse the review queue and the model version ledger below.")

MAX_PER_SESSION = 5
_SMILES_OK = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#$%\\/.:*]+$")


def _insert(cas, name, smi, pct, dur, doi, status):
    conn = sqlite3.connect(DB_PATH, timeout=15)
    try:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO community_submissions
               (cas_rn, chemical_name, smiles, exp_degrad_pct, duration_days, assay_guideline, inoculum_system,
                reference_doi, contributor, status, submitted_at)
               VALUES (?, ?, ?, ?, ?, 'OECD 311 / BMP', 'Digested Sludge', ?, 'Researcher', ?, ?)""",
            (cas, name, smi, float(pct), float(dur), doi, status, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        sid = cur.lastrowid
        conn.commit()
        return sid
    finally:
        conn.close()


left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown("##### Submit an experimental record")
    with st.form("contrib_form", clear_on_submit=False):
        c_cas = st.text_input("CAS Registry Number", placeholder="120-80-9", max_chars=20)
        c_name = st.text_input("Chemical name", placeholder="Catechol", max_chars=120)
        c_smi = st.text_input("Canonical SMILES (required)", placeholder="Oc1ccccc1O", max_chars=C.MAX_QUERY_LEN)
        a, b = st.columns(2)
        c_pct = a.number_input("Measured biodegradation (% ThGP)", min_value=-100.0, max_value=120.0, value=88.0)
        c_dur = b.number_input("Incubation (days)", min_value=1.0, max_value=365.0, value=56.0)
        c_doi = st.text_input("Reference DOI / source (required)", placeholder="10.xxxx/xxxxx", max_chars=200)
        with st.expander("Curator tools (staff only)"):
            c_token = st.text_input("Curator token", type="password", help="Leave empty unless you are a curator.")
        ok = st.form_submit_button("Submit record", type="primary")

    if ok:
        smi, doi = c_smi.strip(), c_doi.strip()
        if st.session_state.get("n_sub", 0) >= MAX_PER_SESSION:
            st.warning("Submission limit reached for this session. Please come back later.")
        elif not smi or not _SMILES_OK.match(smi):
            st.error("Please enter a valid SMILES string (no spaces).")
        elif not doi:
            st.error("A reference DOI or source is required so the record can be verified.")
        else:
            approved = C.check_token(c_token)
            if not approved and not C.PUBLIC_SUBMISSION:
                st.error("Public data submission is currently closed. Please contact the curators.")
                st.stop()
            try:
                sid = _insert(c_cas.strip(), c_name.strip(), smi, c_pct, c_dur, doi, "approved" if approved else "pending")
                st.session_state["n_sub"] = st.session_state.get("n_sub", 0) + 1
                if approved:
                    with st.spinner("Approved — retraining the live model…"):
                        nb = train_and_save_bundle(f"v4.0.{sid}-live")
                    st.cache_data.clear()
                    st.success(f"Record #{sid} approved. Model retrained as {nb['version']}.")
                else:
                    st.success(f"Record #{sid} received and queued for curator review. Thank you!")
            except Exception as e:
                st.error("The record could not be saved right now. Please try again later.")
                C.report_error(e)

with right:
    st.markdown("##### How review works")
    st.markdown(
        "1. Your record is stored as **pending** and does not affect predictions.\n"
        "2. A curator checks the source (DOI) and the structure.\n"
        "3. Approved records are merged and the model is retrained and versioned."
    )
    try:
        conn = sqlite3.connect(DB_PATH, timeout=15)
        q = pd.read_sql("SELECT id, cas_rn, chemical_name, exp_degrad_pct, status, submitted_at FROM community_submissions ORDER BY id DESC LIMIT 10", conn)
        v = pd.read_sql("SELECT * FROM model_versions ORDER BY trained_at DESC LIMIT 6", conn)
        conn.close()
        st.markdown("##### Recent submissions")
        C.show_df(q)
        st.markdown("##### Model version ledger")
        C.show_df(v)
    except Exception:
        st.info("Submission queue and version ledger are not available yet.")
