#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AnaeroBio-Hub v4.0 portal — multi-module Streamlit entry point.
Run:  streamlit run web_portal.py            (the previous single-page app is kept as web_portal_v4_legacy.py)
Pages live in ui/pages/, shared theme + helpers in ui/common.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- Reviewer One-Click Diagnostic Presets ---
try:
    from platform_metadata import ONE_CLICK_DIAGNOSTIC_CASES
except ImportError:
    ONE_CLICK_DIAGNOSTIC_CASES = [
        {"label": "p-Cresol (Gate II Benzoyl-CoA)", "cas": "106-44-5", "smiles": "Cc1ccc(O)cc1", "exp_thgp": 75.3, "oof_thgp": 60.7, "biowin7": 0.163, "ad": "In-Domain"},
        {"label": "Lauramidopropyl betaine (Cleavable Surfactant)", "cas": "10595-49-0", "smiles": "CCCCCCCCCCCC(=O)NCCC[N+](C)(C)C", "exp_thgp": 72.5, "oof_thgp": 42.7, "biowin7": -0.547, "ad": "In-Domain"},
        {"label": "DecaBDE (Gate I Steric Barrier, OOD)", "cas": "1163-19-5", "smiles": "Brc1c(Br)c(Br)c(Oc2c(Br)c(Br)c(Br)c(Br)c2Br)c(Br)c1Br", "exp_thgp": 0.0, "oof_thgp": 10.4, "biowin7": 1.014, "ad": "Out-of-Domain"},
        {"label": "TMT Trithione (Gate III F430 Chelation)", "cas": "638-16-4", "smiles": "S=C1NC(=S)NC(=S)N1", "exp_thgp": -20.5, "oof_thgp": -9.0, "biowin7": -0.868, "ad": "Out-of-Domain"}
    ]

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="AnaeroBio-Hub | Anaerobic fate & inhibition",
    page_icon=":material/science:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from ui import common as C  # noqa: E402

PAGES = {
    "Predict": [
        st.Page("ui/pages/predictor.py", title="Predictor", icon=":material/science:", default=True),
        st.Page("ui/pages/batch.py", title="Batch screening", icon=":material/table_rows:", url_path="batch"),
    ],
    "Explore": [
        st.Page("ui/pages/micropath.py", title="MicroPath", icon=":material/hub:", url_path="micropath"),
        st.Page("ui/pages/database.py", title="Database", icon=":material/database:", url_path="database"),
    ],
    "Contribute & build": [
        st.Page("ui/pages/community.py", title="Community sandbox", icon=":material/verified_user:", url_path="community"),
        st.Page("ui/pages/api_hub.py", title="Open API", icon=":material/code:", url_path="api"),
    ],
}

try:
    nav = st.navigation(PAGES, position="top")
except TypeError:  # Streamlit < 1.46 has no top navigation
    nav = st.navigation(PAGES)

C.inject_css()
C.top_bar()
nav.run()
C.footer()
