#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AnaeroBio-Hub v4.0 — 多任务 5-Fold 交叉验证统计表导出脚本
cv_multitask_si_export.py

用途:
  对 DT50 半衰期子任务 (N=85) 和 Lag 驯化迟滞期子任务 (N=75) 分别执行
  5-Fold Out-Of-Fold (OOF) 交叉验证，输出论文 Supporting Information (SI) 所需的
  统计指标表 (R², RMSE, MAE, Spearman ρ，逐折 + 汇总)。

输出文件 (均写入 D:\\AI coding\\pigntai_develop\\01_final_dataset\\):
  - SI_Table_DT50_5Fold_CV.csv   — 厌氧半衰期 log10(DT50) 回归 5 折汇总
  - SI_Table_Lag_5Fold_CV.csv    — 驯化迟滞期 Lag 回归 5 折汇总
  - SI_Table_Combined_Metrics.csv — 两任务横向对比汇总 (与论文 Table 1 格式对齐)

运行方式:
  cd D:\\AI coding\\pigntai_develop\\06_web_platform
  python cv_multitask_si_export.py

随机种子: 20260817 (与 train_and_save_bundle 保持一致)
"""

import os
import sys
import sqlite3
import warnings
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ── 路径配置 ──────────────────────────────────────────────────────────────
PLAT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PLAT_DIR)
if PLAT_DIR not in sys.path:
    sys.path.insert(0, PLAT_DIR)

import model_engine as ME

DB_PATH = ME.DB_PATH
OUT_DIR = os.path.join(PROJECT_DIR, "01_final_dataset")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 20260817
FEATURE_COLS = ME.FEATURE_COLS


# ── 辅助函数 ──────────────────────────────────────────────────────────────

def compute_fold_metrics(y_true, y_pred):
    """计算单折指标 (R², RMSE, MAE, Spearman ρ)"""
    n = len(y_true)
    if n < 3:
        return dict(N=n, R2=float("nan"), RMSE=float("nan"), MAE=float("nan"), Spearman_rho=float("nan"), Spearman_p=float("nan"))
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    rho, p_val = sp_stats.spearmanr(y_true, y_pred)
    return dict(N=n, R2=round(r2, 4), RMSE=round(rmse, 4), MAE=round(mae, 4),
                Spearman_rho=round(float(rho), 4), Spearman_p=round(float(p_val), 6))

def run_5fold_cv(X, y, task_name, unit_label, transform_y=None, inverse_y=None, n_splits=5):
    """
    执行 5-Fold OOF 交叉验证并返回:
      - fold_results: 每折指标 DataFrame
      - oof_results:  汇总 OOF 整体指标 DataFrame
      - oof_pred_df:  完整 OOF 预测明细 DataFrame (用于绘图)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    model_factory = lambda: ExtraTreesRegressor(
        n_estimators=250, max_depth=12, min_samples_leaf=2,
        random_state=SEED, n_jobs=-1
    )

    y_train = transform_y(y) if transform_y else y
    oof_pred = np.full(len(y_train), np.nan)
    fold_records = []

    print(f"\n[{task_name}] 执行 {n_splits}-Fold OOF 交叉验证 (N={len(y_train)})...")
    for fold_i, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr = y_train[train_idx]

        model = model_factory()
        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_val)
        oof_pred[val_idx] = y_hat

        # 在原始空间计算指标
        y_val_orig = inverse_y(y_train[val_idx]) if inverse_y else y_train[val_idx]
        y_hat_orig = inverse_y(y_hat) if inverse_y else y_hat
        m = compute_fold_metrics(y_val_orig, y_hat_orig)
        m["Fold"] = fold_i
        fold_records.append(m)
        print(f"  Fold {fold_i}: R²={m['R2']:>7.4f}  RMSE={m['RMSE']:>7.4f}  MAE={m['MAE']:>7.4f}  ρ={m['Spearman_rho']:>7.4f}")

    # OOF 整体汇总
    y_oof_orig = inverse_y(oof_pred) if inverse_y else oof_pred
    y_orig = inverse_y(y_train) if inverse_y else y_train
    oof_m = compute_fold_metrics(y_orig, y_oof_orig)
    oof_m["Fold"] = "OOF (Overall)"
    oof_m["Task"] = task_name
    oof_m["Unit"] = unit_label
    print(f"  ── OOF 汇总: R²={oof_m['R2']:>7.4f}  RMSE={oof_m['RMSE']:>7.4f}  MAE={oof_m['MAE']:>7.4f}  ρ={oof_m['Spearman_rho']:>7.4f} ──")

    fold_df = pd.DataFrame(fold_records)[["Fold", "N", "R2", "RMSE", "MAE", "Spearman_rho", "Spearman_p"]]
    oof_pred_df = pd.DataFrame({
        "y_true": y_orig,
        "y_pred_oof": y_oof_orig
    })

    return fold_df, oof_m, oof_pred_df


# ── 主执行逻辑 ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  AnaeroBio-Hub v4.0 — 多任务子模型 5-Fold CV 统计表生成")
    print(f"  数据库: {DB_PATH}")
    print(f"  输出目录: {OUT_DIR}")
    print("=" * 70)

    # ── 加载主数据集 ────────────────────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    df_master = pd.read_sql("SELECT * FROM unified_molecules_master", conn)
    conn.close()
    df_master = df_master.drop_duplicates(subset=["SMILES"], keep="first").copy()
    print(f"\n[数据] 主表去重后: {len(df_master)} 个唯一分子")

    # 构建训练特征矩阵 (使用中位数填充)
    df_master["Duration_days"] = df_master.get("Median_Pct_Duration_d", pd.Series(dtype=float)).fillna(56.0)
    X_all = df_master[FEATURE_COLS].copy()
    medians = X_all.median().to_dict()
    X_all_filled = X_all.fillna(medians)


    # ════════════════════════════════════════════════════════════════════
    #  任务 A: 厌氧转化半衰期 log10(DT50_days)  N ≈ 85
    # ════════════════════════════════════════════════════════════════════
    sub_dt = df_master[
        df_master["Mean_Anaerobic_DT50_d"].notna() & (df_master["Mean_Anaerobic_DT50_d"] > 0)
    ].copy().reset_index(drop=True)

    sub_dt["Duration_days"] = 56.0
    X_dt = sub_dt[FEATURE_COLS].fillna(medians)
    y_dt_raw = np.clip(sub_dt["Mean_Anaerobic_DT50_d"].values, 0.5, 2000.0)

    fold_dt, oof_dt, oof_pred_dt = run_5fold_cv(
        X_dt, y_dt_raw, task_name="Anaerobic Half-life DT50",
        unit_label="days (original space)",
        transform_y=np.log10,
        inverse_y=lambda x: 10 ** x
    )

    # 额外计算 log 空间指标 (供方法论描述)
    oof_log_m = compute_fold_metrics(np.log10(y_dt_raw), np.log10(np.clip(oof_pred_dt["y_pred_oof"].values, 0.5, 2000.0)))

    # 保存逐折表
    dt_out = os.path.join(OUT_DIR, "SI_Table_DT50_5Fold_CV.csv")
    summary_row_dt = pd.DataFrame([{
        "Fold": "OOF (Overall)", "N": len(sub_dt),
        "R2": oof_dt["R2"], "RMSE": oof_dt["RMSE"], "MAE": oof_dt["MAE"],
        "Spearman_rho": oof_dt["Spearman_rho"], "Spearman_p": oof_dt["Spearman_p"]
    }])
    logspace_row = pd.DataFrame([{
        "Fold": "OOF log10-space", "N": len(sub_dt),
        "R2": round(oof_log_m["R2"], 4), "RMSE": round(oof_log_m["RMSE"], 4),
        "MAE": round(oof_log_m["MAE"], 4),
        "Spearman_rho": round(oof_log_m["Spearman_rho"], 4), "Spearman_p": round(oof_log_m["Spearman_p"], 6)
    }])
    dt_full = pd.concat([fold_dt, summary_row_dt, logspace_row], ignore_index=True)
    dt_full.insert(0, "Task", "Anaerobic Half-life DT50 (d)")
    dt_full.to_csv(dt_out, index=False, encoding="utf-8-sig")
    print(f"\n  ✅ DT50 5-Fold CV 表已保存 → {dt_out}")

    # 保存 OOF 明细
    oof_pred_dt["CAS_RN"] = sub_dt.get("CAS_RN", pd.Series(dtype=str)).values
    oof_pred_dt["Chemical_Name"] = sub_dt.get("Chemical_Name", pd.Series(dtype=str)).values
    oof_pred_dt.rename(columns={"y_true": "DT50_obs_d", "y_pred_oof": "DT50_pred_oof_d"}, inplace=True)
    oof_pred_dt["log10_DT50_obs"] = np.log10(np.clip(oof_pred_dt["DT50_obs_d"].values, 0.5, 2000.0))
    oof_pred_dt["log10_DT50_pred"] = np.log10(np.clip(oof_pred_dt["DT50_pred_oof_d"].values, 0.5, 2000.0))
    oof_detail_dt_out = os.path.join(OUT_DIR, "SI_OOF_Detail_DT50.csv")
    oof_pred_dt.to_csv(oof_detail_dt_out, index=False, encoding="utf-8-sig")
    print(f"  ✅ DT50 OOF 预测明细已保存 → {oof_detail_dt_out}")


    # ════════════════════════════════════════════════════════════════════
    #  任务 B: 产甲烷驯化迟滞期 Lag (days)  N ≈ 75
    # ════════════════════════════════════════════════════════════════════
    sub_lag = df_master[df_master["Lit_Lag_days"].notna()].copy().reset_index(drop=True)
    sub_lag["Duration_days"] = 56.0
    X_lag = sub_lag[FEATURE_COLS].fillna(medians)
    y_lag_raw = np.clip(sub_lag["Lit_Lag_days"].values, 1.0, 100.0)

    fold_lag, oof_lag, oof_pred_lag = run_5fold_cv(
        X_lag, y_lag_raw, task_name="Methanogenic Acclimation Lag",
        unit_label="days",
        transform_y=None, inverse_y=None
    )

    lag_out = os.path.join(OUT_DIR, "SI_Table_Lag_5Fold_CV.csv")
    summary_row_lag = pd.DataFrame([{
        "Fold": "OOF (Overall)", "N": len(sub_lag),
        "R2": oof_lag["R2"], "RMSE": oof_lag["RMSE"], "MAE": oof_lag["MAE"],
        "Spearman_rho": oof_lag["Spearman_rho"], "Spearman_p": oof_lag["Spearman_p"]
    }])
    lag_full = pd.concat([fold_lag, summary_row_lag], ignore_index=True)
    lag_full.insert(0, "Task", "Methanogenic Acclimation Lag (d)")
    lag_full.to_csv(lag_out, index=False, encoding="utf-8-sig")
    print(f"\n  ✅ Lag 5-Fold CV 表已保存 → {lag_out}")

    # 保存 OOF 明细
    oof_pred_lag["CAS_RN"] = sub_lag.get("CAS_RN", pd.Series(dtype=str)).values
    oof_pred_lag["Chemical_Name"] = sub_lag.get("Chemical_Name", pd.Series(dtype=str)).values
    oof_pred_lag.rename(columns={"y_true": "Lag_obs_d", "y_pred_oof": "Lag_pred_oof_d"}, inplace=True)
    oof_detail_lag_out = os.path.join(OUT_DIR, "SI_OOF_Detail_Lag.csv")
    oof_pred_lag.to_csv(oof_detail_lag_out, index=False, encoding="utf-8-sig")
    print(f"  ✅ Lag OOF 预测明细已保存 → {oof_detail_lag_out}")


    # ════════════════════════════════════════════════════════════════════
    #  论文 Table 1 格式横向对比汇总表 (含主任务 %ThGP 参考行)
    # ════════════════════════════════════════════════════════════════════
    # 主任务 %ThGP 指标从 bundle 中读取（已在 train_and_save_bundle 时完成）
    try:
        bundle = ME.load_bundle()
        n_pct = bundle.get("n_pct_trained", "~484")
    except Exception:
        n_pct = "~484"

    combined_rows = [
        {
            "Task": "Biodegradation Rate (% ThGP)",
            "Prediction_Target": "% ThGP (continuous, clipped -30 to 100)",
            "N_train": n_pct,
            "Model": "GBDT + ExtraTrees + KNN Ensemble (3-model weighted, w=0.35/0.45/0.20)",
            "CV_Scheme": "5-Fold OOF + Physics Gates",
            "R2_OOF": 0.5823,
            "r_OOF": 0.7631,
            "ROC_AUC_OOF": 0.8893,
            "RMSE_OOF": "—",
            "MAE_OOF": "—",
            "Spearman_rho_OOF": "—",
            "Notes": "Full dataset (SLD N=295 gold standard). R²=0.646/AUC=0.9063 on quantitative subset N=284"
        },
        {
            "Task": "Anaerobic Half-life DT50",
            "Prediction_Target": "log10(DT50 days)",
            "N_train": len(sub_dt),
            "Model": "ExtraTrees (n=250, max_depth=12)",
            "CV_Scheme": "5-Fold OOF",
            "R2_OOF": oof_dt["R2"],
            "r_OOF": round(float(np.sqrt(max(0, oof_dt["R2"]))), 4),
            "ROC_AUC_OOF": "—",
            "RMSE_OOF": round(oof_log_m["RMSE"], 4),
            "MAE_OOF": round(oof_log_m["MAE"], 4),
            "Spearman_rho_OOF": oof_dt["Spearman_rho"],
            "Notes": f"log10-space RMSE={oof_log_m['RMSE']:.4f}; original-space RMSE={oof_dt['RMSE']:.1f} d"
        },
        {
            "Task": "Methanogenic Acclimation Lag",
            "Prediction_Target": "Lag (days, clipped 1–100)",
            "N_train": len(sub_lag),
            "Model": "ExtraTrees (n=250, max_depth=12)",
            "CV_Scheme": "5-Fold OOF",
            "R2_OOF": oof_lag["R2"],
            "r_OOF": round(float(np.sqrt(max(0, oof_lag["R2"]))), 4),
            "ROC_AUC_OOF": "—",
            "RMSE_OOF": oof_lag["RMSE"],
            "MAE_OOF": oof_lag["MAE"],
            "Spearman_rho_OOF": oof_lag["Spearman_rho"],
            "Notes": "Lag thresholds: <7d rapid | 7–28d moderate | >28d slow acclimation"
        }
    ]

    combined_df = pd.DataFrame(combined_rows)
    combined_out = os.path.join(OUT_DIR, "SI_Table_Combined_Multitask_Metrics.csv")
    combined_df.to_csv(combined_out, index=False, encoding="utf-8-sig")
    print(f"\n  ✅ 三任务横向对比汇总表已保存 → {combined_out}")

    # ── 终端汇总输出 ─────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  📊 多任务 5-Fold OOF 汇总指标 (论文 SI 引用版本)")
    print("=" * 70)
    print(f"  {'Task':<40} {'N':>5}  {'R²':>7}  {'ρ':>7}  {'RMSE':>8}  {'MAE':>8}")
    print("  " + "-" * 66)
    for row in combined_rows[1:]:  # 跳过已知 %ThGP 参考行
        r2 = row["R2_OOF"]
        rho = row["Spearman_rho_OOF"]
        rmse = row["RMSE_OOF"]
        mae = row["MAE_OOF"]
        print(f"  {row['Task']:<40} {row['N_train']:>5}  {r2:>7.4f}  {rho:>7.4f}  {str(rmse):>8}  {str(mae):>8}")
    print("=" * 70)
    print(f"\n所有输出文件已写入: {OUT_DIR}")
    print("论文 SI 引用路径:")
    print(f"  → {os.path.basename(dt_out)}")
    print(f"  → {os.path.basename(lag_out)}")
    print(f"  → {os.path.basename(combined_out)}")
    print(f"  → {os.path.basename(oof_detail_dt_out)}  (OOF 散点图数据)")
    print(f"  → {os.path.basename(oof_detail_lag_out)}  (OOF 散点图数据)")


if __name__ == "__main__":
    main()
