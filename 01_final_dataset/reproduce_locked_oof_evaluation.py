#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Canonical OOF Metrics Verification & Reproducibility Script
Directly reads the locked per-molecule OOF tables in 01_final_dataset/ and
04_figures/plot_ready_data/, verifies all regression and classification metrics
without NaN column-collision, and exports 01_final_dataset/verified_web_metrics.json.
"""
import os, json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score

BASE_DIR = r"D:\AI coding\pigntai_develop"
DATA_DIR = os.path.join(BASE_DIR, "01_final_dataset")
EXPORT_DIR = os.path.join(BASE_DIR, "04_figures", "plot_ready_data")

df01 = pd.read_csv(os.path.join(EXPORT_DIR, "01_Master_Curated_546_All_Features_and_OOF.csv"))
df_dt50 = pd.read_csv(os.path.join(DATA_DIR, "SI_OOF_Detail_DT50.csv"))
df_lag = pd.read_csv(os.path.join(DATA_DIR, "SI_OOF_Detail_Lag.csv"))

def exact_metrics(y_true, y_pred, is_pct=True):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    m = {
        "n": int(len(y_true)),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "Pearson_r": round(float(pearsonr(y_true, y_pred)[0]), 4),
        "Spearman_rho": round(float(spearmanr(y_true, y_pred)[0]), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
    }
    if is_pct:
        yb = (y_true >= 60.0).astype(int)
        if len(np.unique(yb)) > 1:
            m["ROC_AUC"] = round(float(roc_auc_score(yb, y_pred)), 4)
    return m

def extract_valid_pair(df_t, is_dt50=False):
    num_cols = [c for c in df_t.columns if pd.api.types.is_numeric_dtype(df_t[c]) and df_t[c].notna().sum() >= 50]
    exp_c = next((c for c in num_cols if ("exp" in c.lower() or "mean_" in c.lower() or "true" in c.lower() or "log10_dt50" in c.lower() or "lag" in c.lower()) and "pred" not in c.lower() and "oof" not in c.lower() and "res" not in c.lower() and "err" not in c.lower()), num_cols[0])
    prd_c = next((c for c in num_cols if ("oof" in c.lower() or "pred" in c.lower()) and c != exp_c), num_cols[1])
    yt = df_t[exp_c].values.astype(float)
    yp = df_t[prd_c].values.astype(float)
    if is_dt50 and np.nanmax(yt) > 20.0 and np.nanmax(yp) > 20.0:
        yt = np.log10(np.clip(yt, 0.5, 2000.0))
        yp = np.log10(np.clip(yp, 0.5, 2000.0))
    return yt, yp

# 1. Task 1: Tier-1 Sludge (n=348) All / In-Domain / Out-of-Domain
s348 = df01[df01["Sludge348_OOF_Pred_Degrad_Pct"].notna()].copy()
s348_in = s348[s348["Sludge348_AD_Status"] == "In-Domain"]
s348_out = s348[s348["Sludge348_AD_Status"] == "Out-of-Domain"]

# 2. Task 1: Full Benchmark (n=496)
s496 = df01[df01["Full496_OOF_Pred_Degrad_Pct"].notna()].copy()
s496_in = s496[s496["Full496_AD_Status"] == "In-Domain"]

# 3. Task 2 (DT50, n=85) & Task 3 (Lag, n=75) directly from their dedicated tables
yt_dt, yp_dt = extract_valid_pair(df_dt50, is_dt50=True)
yt_lg, yp_lg = extract_valid_pair(df_lag, is_dt50=False)

verified = {
    "Task1_Tier1_Sludge_All": exact_metrics(s348["Target_Clipped_Pct"], s348["Sludge348_OOF_Pred_Degrad_Pct"], is_pct=True),
    "Task1_Tier1_Sludge_InDomain": exact_metrics(s348_in["Target_Clipped_Pct"], s348_in["Sludge348_OOF_Pred_Degrad_Pct"], is_pct=True),
    "Task1_Tier1_Sludge_OutOfDomain": exact_metrics(s348_out["Target_Clipped_Pct"], s348_out["Sludge348_OOF_Pred_Degrad_Pct"], is_pct=True),
    "Task1_Full_Benchmark_All": exact_metrics(s496["Target_Clipped_Pct"], s496["Full496_OOF_Pred_Degrad_Pct"], is_pct=True),
    "Task1_Full_Benchmark_InDomain": exact_metrics(s496_in["Target_Clipped_Pct"], s496_in["Full496_OOF_Pred_Degrad_Pct"], is_pct=True),
    "Task2_Log10_DT50": exact_metrics(yt_dt, yp_dt, is_pct=False),
    "Task3_Acclimation_Lag": exact_metrics(yt_lg, yp_lg, is_pct=False),
}

json_path = os.path.join(DATA_DIR, "verified_web_metrics.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(verified, f, indent=2)

print("=== [RECOMPUTED & VERIFIED OOF METRICS FROM PER-MOLECULE OOF TABLES] ===")
for k, v in verified.items():
    print(f"  {k:32s}: {v}")
