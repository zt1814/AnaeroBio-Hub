#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared UI helpers for the AnaeroBio-Hub Streamlit portal (theme, cards, validated + cached inference)."""
import hmac
import html
import inspect
import logging
import os
import re
import urllib.parse

import streamlit as st

from model_engine import predict_molecule, load_bundle

PORTAL_HOME = os.environ.get("PORTAL_HOME_URL", "/")
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "https://your-domain")
# Where the "API docs" link points. Production: /docs (nginx). Local dev: set by launch_platform.ps1.
API_DOCS_URL = os.environ.get("API_DOCS_URL", "/docs")
# ANAERO_DEBUG=1 shows the real traceback in the page (local development only; never on the public site).
DEBUG = os.environ.get("ANAERO_DEBUG", "") == "1"
# Structure image when RDKit is missing: "pubchem" (fetches a PNG from PubChem; the SMILES is sent to PubChem) or "none".
STRUCTURE_FALLBACK = os.environ.get("STRUCTURE_FALLBACK", "pubchem").lower()
log = logging.getLogger("anaerobio.ui")
# Curator token comes ONLY from ANBIOGATE_ADMIN_TOKEN. Unset or empty -> curator approval is disabled.
ADMIN_TOKEN = os.environ.get("ANBIOGATE_ADMIN_TOKEN", "")
# Public community submissions are OFF unless ENABLE_PUBLIC_SUBMISSION=true (default for the peer-review period).
PUBLIC_SUBMISSION = os.environ.get("ENABLE_PUBLIC_SUBMISSION", "false").strip().lower() in ("1", "true", "yes", "on")

MAX_QUERY_LEN = 300
MAX_BATCH = 200
esc = html.escape

# ───────────────────────────── validation / security ─────────────────────────────
_QUERY_RE = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#$%\\/.:*]+$")


def clean_query(q: str) -> str:
    """Validate a SMILES / CAS query from an untrusted visitor. Raises ValueError with a friendly message."""
    q = (q or "").strip()
    if not q:
        raise ValueError("Enter a SMILES string or a CAS Registry Number.")
    if len(q) > MAX_QUERY_LEN:
        raise ValueError(f"Query is too long (max {MAX_QUERY_LEN} characters).")
    if not _QUERY_RE.match(q):
        raise ValueError("Query contains characters that are not valid in a SMILES string or CAS number (spaces are not allowed).")
    return q


def check_token(token: str) -> bool:
    return bool(ADMIN_TOKEN) and hmac.compare_digest((token or "").strip().encode(), ADMIN_TOKEN.encode())


# ───────────────────────────── cached inference ─────────────────────────────
def model_version() -> str:
    return load_bundle()["version"]


_SUPPORTS_SHAP_FLAG = "include_shap" in inspect.signature(predict_molecule).parameters


def _engine_predict(q, dur, sludge, shap):
    """Call the engine; only pass include_shap when this engine version supports it."""
    if _SUPPORTS_SHAP_FLAG:
        return predict_molecule(q, dur, sludge, include_shap=bool(shap))
    return predict_molecule(q, dur, sludge)


@st.cache_data(show_spinner=False, max_entries=1024, ttl=3600)
def _predict_cached(q, dur, sludge, shap, version):  # `version` only keys the cache so retraining invalidates it
    return _engine_predict(q, dur, sludge, shap)


def predict(q: str, dur: float = 56.0, sludge: int = 1, shap: bool = True) -> dict:
    return _predict_cached(q, float(dur), int(sludge), bool(shap), model_version())


KINETIC_DAYS = [3, 7, 14, 21, 28, 42, 56, 70, 90, 120, 150, 180]


@st.cache_data(show_spinner=False, max_entries=256, ttl=3600)
def _kinetics_cached(q, sludge, version):
    return [_engine_predict(q, float(d), int(sludge), False)["predicted_degrad_pct"] for d in KINETIC_DAYS]


def kinetic_curve(q: str, sludge: int):
    return KINETIC_DAYS, _kinetics_cached(q, int(sludge), model_version())


def show_df(df, **kw):
    """st.dataframe at full width on both old (use_container_width) and new (width='stretch') Streamlit."""
    kw.setdefault("hide_index", True)
    try:
        return st.dataframe(df, width="stretch", **kw)
    except TypeError:
        return st.dataframe(df, use_container_width=True, **kw)


_PLACEHOLDER_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 280 150" width="260" role="img" '
                    'aria-label="Structure preview unavailable"><rect x="4" y="4" width="272" height="142" rx="12" fill="#f3f7f6" '
                    'stroke="#c9d8d3"/><polygon points="140,34 162,47 162,73 140,86 118,73 118,47" fill="none" stroke="#0f766e" '
                    'stroke-width="2.5"/><text x="140" y="116" text-anchor="middle" font-family="sans-serif" font-size="13" '
                    'fill="#506760">Structure preview unavailable</text></svg>')


def structure_svg(smiles: str):
    """优先使用本地 RDKit 绘制 SVG；若本机未安装 RDKit，则自动回退至 PubChem 2D PNG 高清分子图"""
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            drawer = rdMolDraw2D.MolDraw2DSVG(280, 180)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            return drawer.GetDrawingText()
    except Exception:
        pass
    if STRUCTURE_FALLBACK == "pubchem":
        enc = urllib.parse.quote(smiles.strip(), safe="")
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{enc}/PNG?record_type=2d&image_size=300x200"
        return (f'<img src="{esc(url)}" alt="Structure preview unavailable offline - see the SMILES below" '
                'style="max-width:260px;height:auto;display:block;margin:0 auto;" />')
    return _PLACEHOLDER_SVG


def report_error(e: Exception, message: str = None):
    """Log the real traceback on the server; show it in the page only when ANAERO_DEBUG=1."""
    log.exception("portal error: %s", type(e).__name__, exc_info=e)
    if message:
        st.error(message)
    if DEBUG:
        st.exception(e)
    else:
        st.caption(f"Reference: {type(e).__name__}")

CSS = """
<style>
/* Streamlit >= 1.4x draws a fixed top navigation bar; keep our brand bar fully below it */
.block-container,[data-testid="stMainBlockContainer"]{padding-top:5rem !important;padding-bottom:3rem;max-width:1220px}
footer{visibility:hidden}
:root{--ab-line:rgba(128,128,128,.28);--ab-soft:rgba(128,128,128,.07);--ab-brand:#0f766e;--ab-muted:rgba(128,128,128,.95)}
.ab-top{display:flex;align-items:center;gap:14px;padding:6px 2px 14px;border-bottom:1px solid var(--ab-line);margin-bottom:6px}
.ab-top a{text-decoration:none;color:inherit}
.ab-brand{display:flex;align-items:center;gap:10px;font-weight:700;font-size:1.05rem}
.ab-mark{width:28px;height:28px;border-radius:8px;background:var(--ab-brand);display:grid;place-items:center;color:#fff;font-size:15px}
.ab-ver{font-size:.72rem;font-weight:600;border:1px solid var(--ab-line);border-radius:99px;padding:1px 8px;color:var(--ab-muted)}
.ab-top .sp{flex:1}
.ab-top .lnk{font-size:.9rem;padding:6px 11px;border-radius:9px;border:1px solid var(--ab-line)}
.ab-top .lnk:hover{background:var(--ab-soft)}
.ab-head{margin:10px 0 18px}
.ab-head h1{font-size:1.9rem;letter-spacing:-.02em;margin:0 0 4px;padding:0}
.ab-head p{margin:0;color:var(--ab-muted);max-width:75ch}
.ab-pill{display:inline-block;font-size:.72rem;font-weight:700;padding:2px 9px;border-radius:99px;margin-left:10px;vertical-align:middle;color:#16a34a;background:rgba(22,163,74,.12);border:1px solid rgba(22,163,74,.35)}
.ab-kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:6px 0 14px}
.ab-kpi{border:1px solid var(--ab-line);background:var(--ab-soft);border-radius:14px;padding:14px 16px}
.ab-kpi .l{font-size:.78rem;color:var(--ab-muted);font-weight:600;letter-spacing:.02em;text-transform:uppercase}
.ab-kpi .v{font-size:1.9rem;font-weight:750;letter-spacing:-.02em;line-height:1.15;font-variant-numeric:tabular-nums}
.ab-kpi .s{font-size:.82rem;color:var(--ab-muted)}
.ab-kpi.ok .v{color:#16a34a}.ab-kpi.bad .v{color:#dc2626}.ab-kpi.warn .v{color:#d97706}
.ab-banner{border-radius:12px;padding:12px 16px;margin:2px 0 14px;border:1px solid;border-left-width:5px}
.ab-banner b{display:block}
.ab-banner.ok{background:rgba(22,163,74,.10);border-color:rgba(22,163,74,.45)}
.ab-banner.warn{background:rgba(217,119,6,.10);border-color:rgba(217,119,6,.45)}
.ab-banner.bad{background:rgba(220,38,38,.10);border-color:rgba(220,38,38,.45)}
.ab-banner.info{background:rgba(15,118,110,.10);border-color:rgba(15,118,110,.45)}
.ab-gate{border:1px solid var(--ab-line);border-left:4px solid #d97706;border-radius:10px;padding:8px 12px;margin:6px 0;font-size:.93rem}
.ab-gate.none{border-left-color:#16a34a}
.ab-chip{display:inline-block;font-size:.8rem;font-weight:600;padding:3px 10px;border-radius:99px;border:1px solid var(--ab-line);background:var(--ab-soft);margin:2px 6px 2px 0}
.ab-mol{background:#fff;border:1px solid var(--ab-line);border-radius:12px;padding:6px;text-align:center;margin-bottom:10px}
.ab-mol svg,.ab-mol img{max-width:100%;height:auto}
.ab-foot{margin-top:34px;padding-top:14px;border-top:1px solid var(--ab-line);color:var(--ab-muted);font-size:.82rem;display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}
.ab-foot a{color:inherit}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def top_bar():
    try:
        ver = esc(model_version())
    except Exception:
        ver = "v4"
    st.markdown(
        f'<div class="ab-top"><a class="ab-brand" href="{esc(PORTAL_HOME)}" target="_self">'
        f'<span class="ab-mark">⚗</span>AnaeroBio-Hub</a><span class="ab-ver">{ver}</span><span class="sp"></span>'
        f'<a class="lnk" href="{esc(PORTAL_HOME)}" target="_self">Home</a>'
        f'<a class="lnk" href="{esc(API_DOCS_URL)}" target="_self">API docs</a></div>',
        unsafe_allow_html=True,
    )


def footer():
    st.markdown(
        '<div class="ab-foot"><span>© AnaeroBio-Hub · AnBioGate. Predictions support research screening and do not replace '
        f'standardised tests or regulatory assessment.</span><span><a href="{esc(API_DOCS_URL)}" target="_self">API</a></span></div>',
        unsafe_allow_html=True,
    )


def page_header(title: str, desc: str, live: bool = True):
    pill = '<span class="ab-pill">Live</span>' if live else ""
    st.markdown(f'<div class="ab-head"><h1>{esc(title)}{pill}</h1><p>{esc(desc)}</p></div>', unsafe_allow_html=True)


def kpi_row(items):
    """items: list of (label, value, sub, tone) — all text is escaped."""
    cells = "".join(
        f'<div class="ab-kpi {esc(tone)}"><div class="l">{esc(label)}</div><div class="v">{esc(value)}</div>'
        f'<div class="s">{esc(sub)}</div></div>'
        for label, value, sub, tone in items
    )
    st.markdown(f'<div class="ab-kpis">{cells}</div>', unsafe_allow_html=True)


def banner(kind: str, title: str, body: str = ""):
    st.markdown(f'<div class="ab-banner {esc(kind)}"><b>{esc(title)}</b>{esc(body)}</div>', unsafe_allow_html=True)


def tone_for(pct: float) -> str:
    return "ok" if pct >= 60 else ("bad" if pct < 0 else "warn")
