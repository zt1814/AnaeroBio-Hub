#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AnaeroBio-Hub v4.0 OpenAPI 3.0 Server (含 / 根路由状态页 + Pending/Approved 两阶段沙箱防污染机制)
"""
import sqlite3
import datetime
from typing import List, Optional
import os
import hmac
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import importlib
import model_engine
from model_engine import predict_molecule, predict_batch_fast, train_and_save_bundle, load_bundle, DB_PATH, METAB_PATH

app = FastAPI(
    title="AnaeroBio-Hub v4.0 Open API (Multi-Task Anaerobic Fate, DT50, Lag & Methanogenic Inhibition)",
    description="Open-access RESTful API integrating 546 Curated Benchmark Molecules, 78,530 EPI Suite SQLite Priors, 28,088 Microbial Metabolic Pathways, and Two-Stage Sandbox Crowdsourcing.",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # public read API: no cookies/credentials needed; "*" origins must not be combined with credentials
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    query: str = Field(..., description="SMILES string or CAS Registry Number (e.g., '106-44-5' or 'Cc1ccc(O)cc1')")
    duration_days: float = Field(56.0, description="Incubation duration in days")
    is_sludge_assay: int = Field(1, description="1 for Digested Sludge (35°C), 0 for Aquatic Sediment/Soil (20°C)")

class BatchPredictRequest(BaseModel):
    queries: List[str]
    duration_days: float = 56.0
    is_sludge_assay: int = 1

class ContributeRequest(BaseModel):
    cas_rn: Optional[str] = ""
    chemical_name: str
    smiles: str
    exp_degrad_pct: float
    duration_days: float = 56.0
    assay_guideline: str = "OECD 311 / ISO 11734"
    inoculum_system: str = "Digested Sludge"
    reference_doi: str = ""
    contributor: str = "Community Researcher"
    admin_token: Optional[str] = Field("", description="Provide curator admin token to auto-approve and trigger immediate hot-retraining; otherwise enters 'pending' sandbox.")

@app.get("/")
def root_welcome():
    importlib.reload(model_engine)
    b = model_engine.load_bundle()
    return {
        "welcome": "AnaeroBio-Hub v4.0 Open RESTful API Server is running!",
        "interactive_swagger_docs": "http://127.0.0.1:8000/docs",
        "interactive_web_portal": "http://localhost:8501",
        "current_model_version": b["version"],
        "benchmark_molecules": b["n_molecules"],
        "endpoints": ["/api/v1/status", "/api/v1/predict", "/api/v1/predict_batch", "/api/v1/metabolism", "/api/v1/contribute"]
    }

@app.get("/api/v1/status")
def get_platform_status():
    b = load_bundle()
    return {
        "platform": "AnaeroBio-Hub",
        "model_version": b["version"],
        "trained_at": b["trained_at"],
        "total_unique_molecules": b["n_molecules"],
        "public_submission_enabled": _env_flag("ENABLE_PUBLIC_SUBMISSION"),
        "tasks_supported": ["Biodegradability (% ThGP)", "Half-life DT50 (days)", "Acclimation Lag Time (days)", "SHAP Modality Attribution"]
    }

@app.post("/api/v1/predict")
def api_predict_single(req: PredictRequest):
    return predict_molecule(req.query, req.duration_days, req.is_sludge_assay)

@app.post("/api/v1/predict_batch")
def api_predict_batch(req: BatchPredictRequest):
    return predict_batch_fast(req.queries, req.duration_days, req.is_sludge_assay)

@app.get("/api/v1/metabolism")
def api_lookup_metabolism(smiles: str):
    df_m = pd.read_csv(METAB_PATH)
    hits = df_m[df_m["parent_smiles"] == smiles.strip()][["parent_smiles", "child_smiles", "remark"]].dropna(subset=["child_smiles"]).fillna("").drop_duplicates()
    return {"parent_smiles": smiles, "n_metabolites_found": len(hits), "metabolites": hits.to_dict(orient="records")}

def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


@app.post("/api/v1/contribute")
def api_contribute_data(req: ContributeRequest):
    # Curator token comes ONLY from ANBIOGATE_ADMIN_TOKEN. Unset/empty => curator approval is disabled
    # (an empty request token must never match an empty secret).
    _secret = os.getenv("ANBIOGATE_ADMIN_TOKEN", "")
    is_approved = bool(_secret) and hmac.compare_digest((req.admin_token or "").encode("utf-8"), _secret.encode("utf-8"))
    # Public submissions are OFF by default (peer-review period). A valid curator token still works.
    if not is_approved and not _env_flag("ENABLE_PUBLIC_SUBMISSION"):
        raise HTTPException(status_code=403, detail="Public data submission is currently closed. Please contact the curators.")
    status_val = "approved" if is_approved else "pending"
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("""
        INSERT INTO community_submissions 
        (cas_rn, chemical_name, smiles, exp_degrad_pct, duration_days, assay_guideline, inoculum_system, reference_doi, contributor, status, submitted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (req.cas_rn, req.chemical_name, req.smiles.strip(), req.exp_degrad_pct, req.duration_days, req.assay_guideline, req.inoculum_system, req.reference_doi, req.contributor, status_val, now_str))
    sub_id = cur.lastrowid
    conn.commit()
    conn.close()

    if is_approved:
        b = train_and_save_bundle(f"v4.0.{sub_id}-live")
        return {"status": "approved_and_retrained", "submission_id": sub_id, "retrained_version": b["version"]}
    return {"status": "staged_in_pending_sandbox", "submission_id": sub_id, "note": "Awaiting curator approval before merging into production weights."}


# ── Local / single-process hosting of the static landing page (site/) at /portal/ ──────────────
# In production nginx serves site/ at "/" and this block is skipped (the Docker image has no ../site).
# index.html uses relative asset paths, so it works at "/" (nginx), "/portal/" (here) and file://.
import mimetypes
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
mimetypes.add_type("application/javascript", ".js")   # Windows registries sometimes map .js to text/plain
mimetypes.add_type("text/css", ".css")
_SITE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "site"))
if os.path.isdir(_SITE_DIR):
    @app.get("/portal", include_in_schema=False)
    def portal_redirect():
        return RedirectResponse(url="/portal/")   # trailing slash keeps relative ./assets/ links working
    app.mount("/portal", StaticFiles(directory=_SITE_DIR, html=True), name="portal")
