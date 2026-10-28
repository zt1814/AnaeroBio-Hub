# -*- coding: utf-8 -*-
"""Module 1 — Single-molecule multi-task predictor (%ThGP + DT50 + Lag, physics gates, attribution, kinetics)."""
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from ui import charts
from ui import common as C

C.page_header(
    "Single-Molecule Predictor",
    "Biodegradation extent (% ThGP), half-life (DT50) and acclimation lag from a SMILES or CAS number, "
    "with the physics gates that fired and an attribution of what drove the result.",
)

EXAMPLES = {
    "p-Cresol · 106-44-5 — BIOWIN7 false negative, Benzoyl-CoA pathway active": "106-44-5",
    "m-Cresol · 108-39-4 — BIOWIN7 false negative, Benzoyl-CoA pathway active": "108-39-4",
    "Dibutyl phthalate · 84-74-2 — ester hydrolysis → methanogenesis": "84-74-2",
    "Decabromodiphenyl ether · 1163-19-5 — steric cut-off": "1163-19-5",
    "Furan · 110-00-9 — F420 coenzyme inhibitor (negative)": "110-00-9",
    "Triazine-trithione (TMT) · 638-16-4 — F430 chelator (negative)": "638-16-4",
    "Lauramidopropyl trimethylammonium · 10595-49-0 — cleavable quat surfactant": "10595-49-0",
    "Custom — type your own below": "",
}
ENVS = {
    "Digested sludge · 35 °C (OECD 311 / ISO 11734 / BMP)": 1,
    "Sediment & soil · 20 °C (OECD 307 / 308)": 0,
}

def _load_reviewer_examples():
    try:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reviewer_examples.json")
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {"note": "", "examples": []}


_REV = _load_reviewer_examples()
_REV_BY_CAS = {e["cas"]: e for e in _REV["examples"]}

st.session_state.setdefault("pred_q", "106-44-5")
if not st.session_state.get("_q_from_url"):          # landing-page deep links: /app/predictor?q=106-44-5
    st.session_state["_q_from_url"] = True
    _q = st.query_params.get("q")
    if _q:
        st.session_state["pred_q"] = str(_q)[: C.MAX_QUERY_LEN]


def _pick_example():
    val = EXAMPLES[st.session_state["pred_ex"]]
    if val:
        st.session_state["pred_q"] = val


def _set_q(cas):
    st.session_state["pred_q"] = cas


if _REV["examples"]:
    st.markdown("##### Reviewer quick-start — one click per diagnostic case")
    _cols = st.columns(len(_REV["examples"]))
    for _c, _e in zip(_cols, _REV["examples"]):
        _c.button(_e["name"], key=f"rev_{_e['cas']}", on_click=_set_q, args=(_e["cas"],), help=f"{_e['cas']} · {_e['why']}")

with st.container(border=True):
    c1, c2, c3 = st.columns([1.25, 1.15, 1.0], gap="large")
    with c1:
        st.selectbox("Showcase molecules", list(EXAMPLES), key="pred_ex", on_change=_pick_example)
        st.text_input("SMILES or CAS Registry Number", key="pred_q", max_chars=C.MAX_QUERY_LEN)
    with c2:
        env_label = st.radio("Anaerobic environment", list(ENVS), index=0)
    with c3:
        dur = st.slider("Incubation duration (days)", min_value=7, max_value=180, value=56, step=7)

sludge = ENVS[env_label]

try:
    query = C.clean_query(st.session_state.get("pred_q", ""))
except ValueError as e:
    st.info(str(e))
    st.stop()

try:
    with st.spinner("Running AnBioGate…"):
        res = C.predict(query, dur, sludge, shap=True)
        days, traj = C.kinetic_curve(query, sludge)
except Exception as e:  # never leak a stack trace to public visitors
    st.error("This structure could not be processed. Please check the SMILES / CAS and try again.")
    C.report_error(e)
    st.stop()

pct = float(res["predicted_degrad_pct"])
exp_pct, exp_dt, exp_lag = res.get("experimental_degrad_pct"), res.get("experimental_dt50_days"), res.get("experimental_lag_days")

C.kpi_row([
    ("Biodegradation", f"{pct:.1f}%", f"measured {exp_pct:.1f}%" if exp_pct is not None else "% ThGP · prediction", C.tone_for(pct)),
    ("Half-life DT50", f"{res['predicted_dt50_days']:.1f} d", f"measured {exp_dt:.1f} d" if exp_dt is not None else "prediction", ""),
    ("Acclimation lag", f"{res['predicted_lag_days']:.1f} d", f"measured {exp_lag:.1f} d" if exp_lag is not None else "prediction", ""),
    ("US EPA BIOWIN 7", f"{res['epa_biowin7_prob']:.3f}", "anaerobic probability", ""),
])
C.banner(C.tone_for(pct), f"{res['chemical_name']}" + (f" · CAS {res['cas_rn']}" if res.get("cas_rn") else ""), f" — {res['classification_outcome']}")

_ex = _REV_BY_CAS.get(str(res.get("cas_rn") or "").strip())
if _ex is not None and sludge == 1:
    with st.container(border=True):
        st.markdown(f"**Cross-validation check · {C.esc(_ex['name'])}** — {C.esc(_ex['why'])}")
        k1, k2, k3 = st.columns(3)
        k1.metric("Measured (literature)", f"{_ex['measured_pct']:.1f}%")
        _oof = _ex["oof_sludge348_pct"] if _ex["oof_sludge348_pct"] is not None else _ex["oof_full496_pct"]
        k2.metric("5-fold OOF prediction (blind)", f"{_oof:.1f}%", help="Made while this molecule was held out of training (Tier-1 sludge cohort, n = 348).")
        k3.metric("Online model (this page)", f"{pct:.1f}%", help="The deployed model was trained on ALL curated molecules including this one, so this value is in-sample and optimistic.")
        _extra = f" Full-benchmark OOF (n = 496): {_ex['oof_full496_pct']:.1f}%." if _ex["oof_full496_pct"] is not None else ""
        st.caption(
            "OOF = out-of-fold prediction from 5-fold cross-validation, the honest estimate of accuracy on a new chemical. "
            "The online value is in-sample (training molecule) and should not be read as accuracy." + _extra +
            f" Applicability domain in cross-validation: {C.esc(_ex['ad_status'])} (kNN distance {_ex['ad_knn_distance']})."
        )

left, right = st.columns([1, 1.45], gap="large")

with left:
    st.markdown("##### Structure & applicability domain")
    svg = C.structure_svg(res["canonical_smiles"])
    if svg:
        st.markdown(f'<div class="ab-mol">{svg}</div>', unsafe_allow_html=True)
    st.code(res["canonical_smiles"], language=None)
    ad_ok = "in-domain" in str(res["applicability_domain"]).lower()
    prior = "78.5k indexed EPI database" if res["hit_90k_priors_db"] else "first-principles estimate"
    st.markdown(
        f'<span class="ab-chip">{"✅" if ad_ok else "⚠️"} {C.esc(str(res["applicability_domain"]))}</span>'
        f'<span class="ab-chip">Prior: {C.esc(prior)}</span>'
        f'<span class="ab-chip">{"In benchmark set" if res["in_benchmark_db"] else "Not in benchmark set"}</span>',
        unsafe_allow_html=True,
    )
    kf = res["key_Modality_features"]
    desc = pd.DataFrame(
        [
            ("Molecular weight", f"{kf['Phys_MW']:.1f}"),
            ("log Kow", f"{kf['Phys_logKow']:.2f}"),
            ("Acidic pKa", f"{kf['Best_Acidic_pKa']:.2f}"),
            ("Neutral fraction, pH 7.2", f"{kf['Neutral_Frac_pH72'] * 100:.1f}%"),
            ("Steric bioavailability barrier", f"{kf['Steric_Bioavail_Barrier']:.2f}"),
            ("Hydrolysable bonds", f"{kf['Hydrolyzable_Bonds']}"),
            ("Halogen atoms", f"{kf['Halogen_Count']}"),
        ],
        columns=["Descriptor", "Value"],
    )
    C.show_df(desc)

with right:
    st.markdown("##### Physics gates & mechanistic alerts")
    gates = res.get("physics_gates_triggered") or []
    if gates:
        for g in gates:
            st.markdown(f'<div class="ab-gate">⚡ <b>{C.esc(str(g))}</b></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="ab-gate none">✅ Standard anaerobic enzymatic mineralisation regime — no gate fired.</div>', unsafe_allow_html=True)

    st.markdown("##### What drove the prediction")
    fig_a = charts.attribution_chart(res.get("shap_modality_contributions") or {}, res["shap_baseline_pct"], pct)
    if fig_a is not None:
        st.pyplot(fig_a)
        plt.close(fig_a)
    else:
        st.caption("Attribution is not available for this query.")

st.markdown("##### Mineralisation over time")
fig_k = charts.kinetics_chart(days, traj, float(res["predicted_lag_days"]), float(dur), pct)
st.pyplot(fig_k)
plt.close(fig_k)

st.download_button(
    "Download full result (JSON)",
    json.dumps(res, indent=2, default=str).encode("utf-8"),
    file_name=f"anaerobio_{res['cas_rn'] or 'query'}.json",
    mime="application/json",
)
with st.expander("How to read this"):
    st.markdown(
        "- **% ThGP** is mineralisation relative to the theoretical gas production; ≥ 60% is treated as readily biodegradable, "
        "and **negative values** flag inhibition of the methanogenic consortium.\n"
        "- **Physics gates** are mechanistic rules (Benzoyl-CoA pathway, hydrolysis, mass-transfer / steric / bond-energy barriers, "
        "F420 / F430 coenzyme inhibition) applied on top of the statistical ensemble.\n"
        "- **Attribution** groups the 42 features into modalities and shows how much each shifted the estimate from the training-set baseline.\n"
        "- Check the **applicability domain** badge: predictions outside the training chemistry are less reliable."
    )
