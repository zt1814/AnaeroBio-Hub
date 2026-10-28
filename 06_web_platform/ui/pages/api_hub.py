# -*- coding: utf-8 -*-
"""Module 6 — Open API hub."""
import json

import streamlit as st

from model_engine import load_bundle
from ui import common as C

C.page_header(
    "Open REST API",
    "Use AnBioGate from your own scripts and pipelines. OpenAPI 3.0, JSON in / JSON out, no API key. "
    "Batch and contribution endpoints are rate-limited per IP.",
)

b = load_bundle()
C.kpi_row([
    ("Model version", b["version"], f"trained {str(b['trained_at'])[:10]}", ""),
    ("Benchmark molecules", f"{b['n_molecules']:,}", "unique SMILES", ""),
    ("Tasks", "3", "% ThGP · DT50 · lag", ""),
    ("Docs", "OpenAPI 3.0", "Swagger UI at /docs", ""),
])
st.link_button("Open interactive docs (Swagger)", C.API_DOCS_URL)

base = C.PUBLIC_BASE_URL.rstrip("/")
t1, t2, t3 = st.tabs(["curl", "Python", "Batch"])
with t1:
    st.code(
        f"""curl -X POST "{base}/api/v1/predict" \\
  -H "Content-Type: application/json" \\
  -d '{{"query": "106-44-5", "duration_days": 56, "is_sludge_assay": 1}}'""",
        language="bash",
    )
with t2:
    st.code(
        f'''import requests

r = requests.post(
    "{base}/api/v1/predict",
    json={{"query": "106-44-5", "duration_days": 56, "is_sludge_assay": 1}},
    timeout=60,
)
res = r.json()
print(res["predicted_degrad_pct"], res["predicted_dt50_days"], res["predicted_lag_days"])''',
        language="python",
    )
with t3:
    st.code(
        f'''import requests
queries = ["106-44-5", "84-74-2", "110-00-9"]          # up to 500
rows = requests.post("{base}/api/v1/predict_batch",
                     json={{"queries": queries, "duration_days": 56, "is_sludge_assay": 1}}, timeout=120).json()''',
        language="python",
    )

st.markdown("##### Endpoints")
st.markdown(
    "| Method | Path | Purpose |\n|---|---|---|\n"
    "| GET | `/api/v1/status` | Model version and tasks |\n"
    "| POST | `/api/v1/predict` | One molecule, full diagnosis |\n"
    "| POST | `/api/v1/predict_batch` | Up to 500 molecules |\n"
    "| GET | `/api/v1/metabolism?smiles=` | Observed metabolites of a parent |\n"
    "| POST | `/api/v1/contribute` | Submit a record (pending review) |"
)

with st.expander("Example response (p-cresol)"):
    try:
        ex = C.predict("106-44-5", 56.0, 1, shap=True)
        st.json(json.loads(json.dumps(ex, default=str)))
    except Exception:
        st.caption("Example unavailable.")
