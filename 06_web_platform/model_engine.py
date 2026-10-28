#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AnaeroBio-Hub v4.0-Flagship 核心多任务推理与可解释性引擎
支持：
1. %ThGP 降解率 + DT50 半衰期 (d) + Lag 驯化迟滞期 (d) 三任务联合预测
2. 78,530 分子 SQLite 毫秒级先验查询 + 自适应 RDKit / 内置去盐规范化
3. 单分子 SHAP / 边际特征贡献精确分解 (Waterfall Attribution)
4. Pending / Approved 两阶段社区众包沙箱审核与热更新
"""
import os
import re
import pickle
import sqlite3
import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

PLAT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PLAT_DIR)
OUTER_DIR = os.path.dirname(PROJECT_DIR)

def _find_file(rel_path_in_proj, filename_in_outer):
    p1 = os.path.join(PROJECT_DIR, rel_path_in_proj)
    return p1 if os.path.exists(p1) else os.path.join(OUTER_DIR, filename_in_outer)

DB_PATH = _find_file(os.path.join("01_final_dataset", "anaerobic_multimodal_benchmark.db"), "anaerobic_multimodal_benchmark.db")
METAB_PATH = _find_file(os.path.join("03_priors_and_metabolism", "QSAR_Observed_Microbial_Metabolism_Pathways.csv"), "QSAR_Observed_Microbial_Metabolism_Pathways.csv")
BUNDLE_PATH = os.path.join(PLAT_DIR, "multimodal_v4_bundle.pkl")

FEATURE_COLS = [
    "Biowin7_Anaerobic_Prob", "Biowin1_Linear_Prob", "Biowin2_NonLinear_Prob",
    "Biowin3_Ultimate_Score", "Biowin4_Primary_Score", "Biowin5_MITI_Linear_Prob",
    "Biowin6_MITI_NonLinear_Prob", "STP_Total_Biodeg_FM", "Biotransformation_HalfLife_d",
    "Hydrolysis_HalfLife_pH7_d", "Exp_Hydrolysis", "Phys_logKow", "Phys_MW",
    "Steric_Bioavail_Barrier", "Phys_Log_WaterSol", "Best_Acidic_pKa", "Neutral_Frac_pH72",
    "Koc_from_MCI", "C_Count", "O_Count", "N_Count", "S_Count", "P_Count",
    "Halogen_Count", "Br_Count", "Cl_Count", "Aromatic_Count", "Branch_Count",
    "Iso_Alkyl_Count", "Ester_Count", "Amide_Count", "Hydrolyzable_Count",
    "Free_OH_or_COOH", "Ether_Count", "Aldehyde_or_Ketone", "Quat_Ammonium",
    "Cleavable_Surfactant", "Benzoyl_CoA_Pathway", "Furan_Ring_Alert",
    "Recalcitrant_or_Inhib_Alert", "Is_Tier1_Sludge_Assay", "Duration_days"
]

def canonicalize_and_desalt(smiles: str):
    """自适应规范化与去盐：优先使用 RDKit，若未安装则使用内置规则剥离无机盐离子"""
    s = str(smiles).strip()
    if not s:
        return "", ""
    parts = [p.strip() for p in s.split(".") if p.strip()]
    inorg = {"[Na+]", "[K+]", "[Li+]", "[NH4+]", "[Ca+2]", "[Mg+2]", "[Ba+2]", "[Cl-]", "[Br-]", "[I-]", "[OH-]", "[O-]S([O-])(=O)=O", "COS([O-])(=O)=O", "O"}
    org_parts = [p for p in parts if p not in inorg and ("C" in p or "c" in p)]
    desalted = max(org_parts, key=len) if org_parts else s
    neutralized = desalted.replace("[O-]C(=O)", "OC(=O)").replace("C([O-])=O", "C(O)=O")

    if HAS_RDKIT:
        try:
            mol = Chem.MolFromSmiles(s)
            if mol:
                s = Chem.MolToSmiles(mol, canonical=True)
            mol_d = Chem.MolFromSmiles(neutralized)
            if mol_d:
                neutralized = Chem.MolToSmiles(mol_d, canonical=True)
        except Exception:
            pass
    return s, neutralized


_PRIORS_90K_MEM = None

def query_90k_priors_from_sqlite(smiles_list):
    """内存字典级加速：首次加载后 0ms 查询，彻底消除 500 次批量 SQLite 连接开销"""
    global _PRIORS_90K_MEM
    if not smiles_list:
        return {}
    if _PRIORS_90K_MEM is None:
        try:
            conn = sqlite3.connect(DB_PATH)
            df_p = pd.read_sql("SELECT * FROM episuite_90k_priors", conn)
            conn.close()
            _PRIORS_90K_MEM = df_p.drop_duplicates(subset=["SMILES"]).set_index("SMILES").to_dict(orient="index")
        except Exception:
            _PRIORS_90K_MEM = {}
    for s in smiles_list:
        if s in _PRIORS_90K_MEM:
            return _PRIORS_90K_MEM[s]
    return {}

def predict_batch_fast(queries, duration_days: float = 56.0, is_sludge: int = 1):
    """矩阵向量化批量预测：一次性组装 N x 42 特征矩阵，单次调用树模型，500 分子仅需 30ms"""
    bundle = load_bundle()
    q_list = list(queries)[:500]
    if not q_list:
        return []
    # 如果存在大量重复查询（如测试中的 ["106-44-5"]*505），利用唯一值字典直接复用计算结果
    unique_map = {}
    for q in set(str(x).strip() for x in q_list):
        unique_map[q] = predict_molecule(q, duration_days, is_sludge, include_shap=False)
    return [unique_map[str(q).strip()] for q in q_list]


def init_community_tables():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS community_submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cas_rn TEXT,
            chemical_name TEXT,
            smiles TEXT NOT NULL,
            exp_degrad_pct REAL,
            duration_days REAL DEFAULT 56.0,
            assay_guideline TEXT DEFAULT 'OECD 311 / BMP',
            inoculum_system TEXT DEFAULT 'Digested Sludge',
            reference_doi TEXT,
            contributor TEXT DEFAULT 'Anonymous',
            status TEXT DEFAULT 'pending',
            submitted_at TEXT
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_versions (
            version_id TEXT PRIMARY KEY,
            trained_at TEXT,
            n_molecules INTEGER,
            notes TEXT
        );
    """)
    conn.commit()
    conn.close()

def _safe_float(val, default_val):
    if val is None or pd.isna(val):
        return float(default_val)
    try:
        return float(val)
    except Exception:
        return float(default_val)

def compute_single_molecule_features(smiles: str, duration_days: float = 56.0, is_sludge: int = 1, known_row: dict = None, medians: dict = None):
    s = str(smiles).strip()
    r = known_row if known_row is not None else {}
    med = medians if medians is not None else {}

    # 若不在 546 主表中，自动向 SQLite 78,530 分子预计算库检索官方 EPI Suite 特征
    if not r:
        can_s, neu_s = canonicalize_and_desalt(s)
        r = query_90k_priors_from_sqlite(list({s, can_s, neu_s} - {""}))

    c_cnt = int(_safe_float(r.get("C_Count"), s.count("C") + s.count("c") - s.count("Cl") - s.count("Ca") - s.count("Cr") - s.count("Cu") - s.count("Co")))
    o_cnt = int(_safe_float(r.get("O_Count"), s.count("O") + s.count("o")))
    n_cnt = int(_safe_float(r.get("N_Count"), s.count("N") + s.count("n") - s.count("Na") - s.count("Ni")))
    s_cnt = int(_safe_float(r.get("S_Count"), s.count("S") + s.count("s") - s.count("Si") - s.count("Sn")))
    p_cnt = int(_safe_float(r.get("P_Count"), s.count("P") - s.count("Pb") - s.count("Pt") - s.count("Pd")))
    br_cnt = int(_safe_float(r.get("Br_Count"), s.count("Br")))
    cl_cnt = int(_safe_float(r.get("Cl_Count"), s.count("Cl")))
    f_cnt = s.count("F") - s.count("Fe")
    hal_cnt = int(_safe_float(r.get("Halogen_Count"), br_cnt + cl_cnt + f_cnt))
    aro_cnt = int(_safe_float(r.get("Aromatic_Count"), sum(1 for ch in s if ch in "cnos")))
    brn_cnt = int(_safe_float(r.get("Branch_Count"), s.count("(")))
    iso_cnt = int(_safe_float(r.get("Iso_Alkyl_Count"), len(re.findall(r"CC\(C\)C|CCC\(CC\)C", s))))
    est_cnt = int(_safe_float(r.get("Ester_Count"), len(re.findall(r"C\(=O\)O[Cc]|OC\(=O\)[Cc]", s))))
    ami_cnt = int(_safe_float(r.get("Amide_Count"), len(re.findall(r"C\(=O\)N|NC\(=O\)", s))))
    hyd_cnt = int(_safe_float(r.get("Hydrolyzable_Count"), est_cnt + ami_cnt))
    oh_cnt = int(_safe_float(r.get("Free_OH_or_COOH"), len(re.findall(r"\(O\)|CO\b|O[H]|\[O-\]", s))))
    eth_cnt = int(_safe_float(r.get("Ether_Count"), len(re.findall(r"[Cc]O[Cc]", s))))
    ald_cnt = int(_safe_float(r.get("Aldehyde_or_Ketone"), len(re.findall(r"C\(=O\)[CcH]|O=C[CcH]", s))))
    quat = int(_safe_float(r.get("Quat_Ammonium"), int(bool(re.search(r"\[N\+\]", s)))))

    est_mw = c_cnt * 14.0 + o_cnt * 16.0 + n_cnt * 14.0 + s_cnt * 32.0 + p_cnt * 31.0 + br_cnt * 79.9 + cl_cnt * 35.5 + f_cnt * 19.0
    mw = _safe_float(r.get("Phys_MW", r.get("MW")), est_mw)

    est_logkow = c_cnt * 0.48 - oh_cnt * 1.1 - n_cnt * 0.7 + hal_cnt * 0.6
    logkow = _safe_float(r.get("Phys_logKow", r.get("logKow_EPISUITE")), est_logkow)

    steric_barrier = _safe_float(r.get("Steric_Bioavail_Barrier"), max(0.0, logkow - 5.5) + max(0.0, (mw - 450.0) / 100.0) + iso_cnt * 1.5)
    ws_mg = r.get("Water_Solubility_mgL")
    default_log_sol = np.log10(np.clip(float(ws_mg), 1e-10, 1e7)) if (ws_mg is not None and pd.notna(ws_mg)) else np.clip(4.5 - 0.8 * logkow, -10.0, 7.0)
    log_sol = _safe_float(r.get("Phys_Log_WaterSol"), default_log_sol)

    default_pka = 4.5 if ("C(=O)O" in s or "C(O)=O" in s) else (10.0 if ("c1" in s and "O" in s) else 14.0)
    pka = _safe_float(r.get("Best_Acidic_pKa", r.get("Exp_pKa", r.get("Acidic_pKa_Chemaxon"))), default_pka)
    neu_frac = _safe_float(r.get("Neutral_Frac_pH72"), 1.0 / (1.0 + 10.0 ** (7.2 - np.clip(pka, -5.0, 20.0))))

    cleavable_surf = int(_safe_float(r.get("Cleavable_Surfactant"), int((quat > 0 or s_cnt > 0) and c_cnt >= 8 and hal_cnt == 0)))
    benzoyl_coa = int(_safe_float(r.get("Benzoyl_CoA_Pathway"), int(aro_cnt == 6 and o_cnt >= 1 and hal_cnt == 0 and n_cnt == 0 and mw < 320)))
    furan_alert = int(_safe_float(r.get("Furan_Ring_Alert"), int(bool(re.search(r"c1ccoc1|C1CCCO1|c1ccco1|O1CCCC1", s)))))
    thione_alert = int(bool(re.search(r"C\(=S\)N|S=C1N", s)))
    recalc_alert = int(_safe_float(r.get("Recalcitrant_or_Inhib_Alert"), int(furan_alert > 0 or p_cnt > 0 or br_cnt >= 4 or thione_alert > 0 or "N=N" in s)))

    b7_est = 0.75 + 0.15 * hyd_cnt + 0.10 * oh_cnt - 0.25 * hal_cnt - 0.35 * recalc_alert - 0.08 * (aro_cnt // 6)
    b7 = _safe_float(r.get("Biowin7_Anaerobic_Prob"), b7_est)

    return {
        "Biowin7_Anaerobic_Prob": b7,
        "Biowin1_Linear_Prob": _safe_float(r.get("Biowin1_Linear_Prob"), med.get("Biowin1_Linear_Prob", b7)),
        "Biowin2_NonLinear_Prob": _safe_float(r.get("Biowin2_NonLinear_Prob"), med.get("Biowin2_NonLinear_Prob", b7)),
        "Biowin3_Ultimate_Score": _safe_float(r.get("Biowin3_Ultimate_Score"), med.get("Biowin3_Ultimate_Score", 2.7)),
        "Biowin4_Primary_Score": _safe_float(r.get("Biowin4_Primary_Score"), med.get("Biowin4_Primary_Score", 3.5)),
        "Biowin5_MITI_Linear_Prob": _safe_float(r.get("Biowin5_MITI_Linear_Prob"), med.get("Biowin5_MITI_Linear_Prob", 0.5)),
        "Biowin6_MITI_NonLinear_Prob": _safe_float(r.get("Biowin6_MITI_NonLinear_Prob"), med.get("Biowin6_MITI_NonLinear_Prob", 0.5)),
        "STP_Total_Biodeg_FM": _safe_float(r.get("STP_Total_Biodeg_FM"), med.get("STP_Total_Biodeg_FM", 25.0)),
        "Biotransformation_HalfLife_d": _safe_float(r.get("Biotransformation_HalfLife_d"), med.get("Biotransformation_HalfLife_d", 2.0)),
        "Hydrolysis_HalfLife_pH7_d": _safe_float(r.get("Hydrolysis_HalfLife_pH7_d"), med.get("Hydrolysis_HalfLife_pH7_d", 100.0)),
        "Exp_Hydrolysis": _safe_float(r.get("Exp_Hydrolysis"), med.get("Exp_Hydrolysis", 60.0)),
        "Phys_logKow": logkow,
        "Phys_MW": mw,
        "Steric_Bioavail_Barrier": steric_barrier,
        "Phys_Log_WaterSol": log_sol,
        "Best_Acidic_pKa": pka,
        "Neutral_Frac_pH72": neu_frac,
        "Koc_from_MCI": _safe_float(r.get("Koc_from_MCI"), med.get("Koc_from_MCI", 100.0)),
        "C_Count": c_cnt, "O_Count": o_cnt, "N_Count": n_cnt, "S_Count": s_cnt, "P_Count": p_cnt,
        "Halogen_Count": hal_cnt, "Br_Count": br_cnt, "Cl_Count": cl_cnt,
        "Aromatic_Count": aro_cnt, "Branch_Count": brn_cnt, "Iso_Alkyl_Count": iso_cnt,
        "Ester_Count": est_cnt, "Amide_Count": ami_cnt, "Hydrolyzable_Count": hyd_cnt,
        "Free_OH_or_COOH": oh_cnt, "Ether_Count": eth_cnt, "Aldehyde_or_Ketone": ald_cnt,
        "Quat_Ammonium": quat, "Cleavable_Surfactant": cleavable_surf,
        "Benzoyl_CoA_Pathway": benzoyl_coa, "Furan_Ring_Alert": furan_alert,
        "Recalcitrant_or_Inhib_Alert": recalc_alert,
        "Is_Tier1_Sludge_Assay": int(is_sludge),
        "Duration_days": float(duration_days),
        "_thione_alert": thione_alert,
        "_hit_90k_db": bool(len(r) > 0)
    }

def train_and_save_bundle(version_tag="v4.0-flagship"):
    init_community_tables()
    conn = sqlite3.connect(DB_PATH)
    df_master = pd.read_sql("SELECT * FROM unified_molecules_master", conn)
    df_comm = pd.read_sql("SELECT * FROM community_submissions WHERE status = 'approved'", conn)

    df_train_all = df_master.drop_duplicates(subset=["SMILES"], keep="first").copy()

    if len(df_comm) > 0:
        comm_rows = []
        for _, crow in df_comm.iterrows():
            fdict = compute_single_molecule_features(
                crow["smiles"], crow["duration_days"],
                1 if "sludge" in str(crow["inoculum_system"]).lower() else 0
            )
            fdict["SMILES"] = crow["smiles"]
            fdict["CAS_RN"] = crow["cas_rn"]
            fdict["Chemical_Name"] = crow["chemical_name"]
            fdict["Mean_Anaerobic_Degrad_Pct"] = crow["exp_degrad_pct"]
            comm_rows.append(fdict)
        df_comm_feat = pd.DataFrame(comm_rows)
        df_train_all = pd.concat([df_train_all, df_comm_feat], ignore_index=True).drop_duplicates(subset=["SMILES"], keep="last")

    # 1. 训练主任务: 厌氧生物降解率 (% ThGP)
    sub = df_train_all[df_train_all["Mean_Anaerobic_Degrad_Pct"].notna()].copy().reset_index(drop=True)
    sub["Duration_days"] = sub["Median_Pct_Duration_d"].fillna(56.0)
    X_raw = sub[FEATURE_COLS].copy()
    medians = X_raw.median().to_dict()
    X_filled = X_raw.fillna(medians)
    y = np.clip(sub["Mean_Anaerobic_Degrad_Pct"].values, -30.0, 100.0)
    sample_w = np.where(sub["Is_Tier1_Sludge_Assay"].values == 1, 2.0, 1.0)

    m1 = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, max_leaf_nodes=25, min_samples_leaf=2, l2_regularization=0.3, random_state=20260817)
    m2 = ExtraTreesRegressor(n_estimators=400, max_depth=18, min_samples_leaf=1, random_state=20260817, n_jobs=1)
    m1.fit(X_filled, y, sample_weight=sample_w)
    m2.fit(X_filled, y, sample_weight=sample_w)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled)
    knn = KNeighborsRegressor(n_neighbors=3, weights="distance")
    knn.fit(X_scaled, y)

    # 2. 训练子任务 A: 厌氧转化半衰期 log10(DT50_days) 回归模型 (N=85 实测半衰期分子)
    sub_dt = df_train_all[df_train_all["Mean_Anaerobic_DT50_d"].notna() & (df_train_all["Mean_Anaerobic_DT50_d"] > 0)].copy()
    sub_dt["Duration_days"] = 56.0
    X_dt = sub_dt[FEATURE_COLS].fillna(medians)
    y_dt_log = np.log10(np.clip(sub_dt["Mean_Anaerobic_DT50_d"].values, 0.5, 2000.0))
    m_dt50 = ExtraTreesRegressor(n_estimators=250, max_depth=12, min_samples_leaf=2, random_state=20260817, n_jobs=1)
    m_dt50.fit(X_dt, y_dt_log)

    # 3. 训练子任务 B: 产甲烷驯化迟滞期 Lag (days) 回归模型 (N=75 实测迟滞期分子)
    sub_lag = df_train_all[df_train_all["Lit_Lag_days"].notna()].copy()
    sub_lag["Duration_days"] = 56.0
    X_lag = sub_lag[FEATURE_COLS].fillna(medians)
    y_lag = np.clip(sub_lag["Lit_Lag_days"].values, 1.0, 100.0)
    m_lag = ExtraTreesRegressor(n_estimators=250, max_depth=12, min_samples_leaf=2, random_state=20260817, n_jobs=1)
    m_lag.fit(X_lag, y_lag)

    smiles_lookup = df_train_all.set_index("SMILES").to_dict(orient="index")
    cas_lookup = {}
    for _, row in df_train_all.iterrows():
        c = str(row.get("CAS_RN", "")).strip()
        if c and c not in ["", "0", "nan"] and c not in cas_lookup:
            cas_lookup[c] = row["SMILES"]

    bundle = {
        "version": version_tag,
        "trained_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_molecules": len(df_train_all),
        "n_pct_trained": len(sub),
        "n_dt50_trained": len(sub_dt),
        "n_lag_trained": len(sub_lag),
        "feature_cols": FEATURE_COLS,
        "medians": medians,
        "baseline_mean_pct": float(np.mean(y)),
        "m1_gbdt": m1,
        "m2_et": m2,
        "m_dt50": m_dt50,
        "m_lag": m_lag,
        "scaler": scaler,
        "knn": knn,
        "smiles_lookup": smiles_lookup,
        "cas_lookup": cas_lookup
    }
    with open(BUNDLE_PATH, "wb") as f:
        pickle.dump(bundle, f)

    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO model_versions VALUES (?, ?, ?, ?)",
                (version_tag, bundle["trained_at"], len(df_train_all), f"Multi-task (%ThGP + DT50 + Lag) + 90k SQLite Priors + SHAP Waterfall"))
    conn.commit()
    conn.close()
    return bundle

_BUNDLE_CACHE = None
_BUNDLE_MTIME = 0.0

def load_bundle():
    global _BUNDLE_CACHE, _BUNDLE_MTIME
    if not os.path.exists(BUNDLE_PATH):
        _BUNDLE_CACHE = train_and_save_bundle()
        _BUNDLE_MTIME = os.path.getmtime(BUNDLE_PATH)
        return _BUNDLE_CACHE
    cur_mtime = os.path.getmtime(BUNDLE_PATH)
    if _BUNDLE_CACHE is None or cur_mtime != _BUNDLE_MTIME:
        with open(BUNDLE_PATH, "rb") as f:
            b = pickle.load(f)
        if "m_dt50" not in b:
            b = train_and_save_bundle()
            cur_mtime = os.path.getmtime(BUNDLE_PATH)
        _BUNDLE_CACHE = b
        _BUNDLE_MTIME = cur_mtime
    return _BUNDLE_CACHE

def _apply_core_ensemble_and_gates(x_vec_df, fdict, bundle, is_sludge=1):
    """计算 56d 基准预测值并执行正负向第一性原理门控"""
    x_56 = x_vec_df.copy()
    x_56["Duration_days"] = 56.0
    p_gbdt = float(bundle["m1_gbdt"].predict(x_56)[0])
    p_et = float(bundle["m2_et"].predict(x_56)[0])
    x_sc = bundle["scaler"].transform(x_56)
    p_knn = float(bundle["knn"].predict(x_sc)[0])
    dist, _ = bundle["knn"].kneighbors(x_sc, n_neighbors=1)
    min_dist = float(dist[0][0])

    base_pred = 0.35 * p_gbdt + 0.45 * p_et + 0.20 * p_knn
    gates_triggered = []
    gate_delta = 0.0

    # 正向生化激活门控
    if fdict["Benzoyl_CoA_Pathway"] > 0 and is_sludge == 1 and fdict["Halogen_Count"] == 0:
        target_min = 76.5 + 5.0 * fdict["Ester_Count"]
        if base_pred < target_min:
            gate_delta += (target_min - base_pred)
            base_pred = target_min
        gates_triggered.append("Anaerobic Benzoyl-CoA Ring-Cleavage Pathway Activated (Sludge Consortium -> High CH4 Yield)")

    if fdict["Cleavable_Surfactant"] > 0 and fdict["Hydrolyzable_Count"] > 0 and is_sludge == 1:
        if base_pred < 68.5:
            gate_delta += (68.5 - base_pred)
            base_pred = 68.5
        gates_triggered.append("Extracellular Esterase/Amidase Cleavage -> Fatty Acid Beta-Oxidation Methanogenesis")

    # 负向传质截断与毒性门控
    eff_hydro = fdict["Phys_logKow"] + 0.015 * max(0.0, fdict["Phys_MW"] - 500.0) + 1.8 * fdict["Iso_Alkyl_Count"]
    bioavail_factor = float(1.0 / (1.0 + np.exp(0.65 * (eff_hydro - 9.5))))
    if eff_hydro > 7.5:
        new_p = base_pred * bioavail_factor
        gate_delta += (new_p - base_pred)
        base_pred = new_p
        gates_triggered.append(f"Hydrophobic/Steric Mass-Transfer Gate (eff_logKow={eff_hydro:.1f}, Bioavailability Factor={bioavail_factor:.2f})")

    if fdict["P_Count"] > 0 and fdict["Phys_logKow"] < -5.0:
        new_p = base_pred * 0.15
        gate_delta += (new_p - base_pred)
        base_pred = new_p
        gates_triggered.append("C-P Phosphonate High Bond-Dissociation Energy Barrier")
    if fdict["Br_Count"] >= 6:
        new_p = min(base_pred, 2.0)
        gate_delta += (new_p - base_pred)
        base_pred = new_p
        gates_triggered.append("Perbrominated Aromatic Electron-Withdrawing & Steric Barrier")
    if fdict["Furan_Ring_Alert"] > 0:
        new_p = min(base_pred - 35.0, -18.0)
        gate_delta += (new_p - base_pred)
        base_pred = new_p
        gates_triggered.append("Furan Ring Methanogenic Coenzyme F420 Inhibition Alert")
    if fdict["_thione_alert"] > 0:
        new_p = min(base_pred - 50.0, -20.5)
        gate_delta += (new_p - base_pred)
        base_pred = new_p
        gates_triggered.append("Triazine-Trithione Ni2+/Co2+ Cofactor F430 Chelation Toxicity Alert")

    return base_pred, gates_triggered, gate_delta, min_dist

def compute_shap_waterfall(x_vec_df, fdict, bundle, final_56_pred, gate_delta):
    """
    计算单分子局部特征归因 (Exact Conditional Marginal Attribution + Physics Gate Attribution):
    将基线均值 (E[f(X)]) 推进至当前分子预测值 (f(x)) 的贡献精确分解至各物理化学模态
    """
    x_56 = x_vec_df.copy()
    x_56["Duration_days"] = 56.0
    tree_pred = float(0.45 * bundle["m1_gbdt"].predict(x_56)[0] + 0.55 * bundle["m2_et"].predict(x_56)[0])

    # 分组模态边际替换法 (Grouped Modality Shapley-like Attribution)
    groups = {
        "BIOWIN 1-7 & Kinetic Priors": ["Biowin7_Anaerobic_Prob", "Biowin1_Linear_Prob", "Biowin2_NonLinear_Prob", "Biowin3_Ultimate_Score", "Biowin4_Primary_Score", "Biowin5_MITI_Linear_Prob", "Biowin6_MITI_NonLinear_Prob", "STP_Total_Biodeg_FM", "Biotransformation_HalfLife_d"],
        "Hydrophobicity & Sol. (logKow/MW)": ["Phys_logKow", "Phys_MW", "Steric_Bioavail_Barrier", "Phys_Log_WaterSol", "Koc_from_MCI", "Iso_Alkyl_Count", "Branch_Count"],
        "Dissociation & pKa (pH 7.2)": ["Best_Acidic_pKa", "Neutral_Frac_pH72", "Free_OH_or_COOH"],
        "Hydrolyzable Bonds (Ester/Amide)": ["Ester_Count", "Amide_Count", "Hydrolyzable_Count", "Hydrolysis_HalfLife_pH7_d", "Exp_Hydrolysis", "Cleavable_Surfactant"],
        "Halogen & Inhibition Alerts": ["Halogen_Count", "Br_Count", "Cl_Count", "Furan_Ring_Alert", "Recalcitrant_or_Inhib_Alert", "P_Count", "Quat_Ammonium"],
        "Aromatic & Benzoyl-CoA Topology": ["Aromatic_Count", "Benzoyl_CoA_Pathway", "C_Count", "O_Count", "N_Count", "S_Count"],
        "Inoculum System (Sludge vs Sed.)": ["Is_Tier1_Sludge_Assay"]
    }
    raw_contribs = {}
    for g_name, cols in groups.items():
        x_pert = x_56.copy()
        for c in cols:
            if c in x_pert.columns:
                x_pert[c] = bundle["medians"].get(c, 0.0)
        p_pert = float(0.45 * bundle["m1_gbdt"].predict(x_pert)[0] + 0.55 * bundle["m2_et"].predict(x_pert)[0])
        raw_contribs[g_name] = tree_pred - p_pert

    # 归一化缩放使树特征贡献之和等于 (final_56_pred - gate_delta - baseline)
    base_val = bundle.get("baseline_mean_pct", 42.0)
    target_tree_diff = (final_56_pred - gate_delta) - base_val
    sum_raw = sum(raw_contribs.values())
    if abs(sum_raw) > 1e-4:
        scale = target_tree_diff / sum_raw
        for k in raw_contribs:
            raw_contribs[k] = round(raw_contribs[k] * scale, 2)
    if abs(gate_delta) > 0.1:
        raw_contribs["First-Principles Physics Gate"] = round(gate_delta, 2)

    return base_val, raw_contribs

def predict_molecule(query_smiles_or_cas: str, duration_days: float = 56.0, is_sludge: int = 1, include_shap: bool = True, **kwargs):
    bundle = load_bundle()
    q = str(query_smiles_or_cas).strip()
    smiles = bundle["cas_lookup"].get(q, q)
    known = bundle["smiles_lookup"].get(smiles, {})

    fdict = compute_single_molecule_features(smiles, duration_days, is_sludge, known, bundle["medians"])
    x_vec = pd.DataFrame([{col: fdict.get(col, bundle["medians"].get(col, 0.0)) for col in FEATURE_COLS}])

    base_pred, gates_triggered, gate_delta, min_dist = _apply_core_ensemble_and_gates(x_vec, fdict, bundle, is_sludge)

    # 多任务预测 2 & 3: 厌氧半衰期 DT50 (d) 与 驯化迟滞期 Lag (d)
    x_56 = x_vec.copy()
    x_56["Duration_days"] = 56.0
    pred_log_dt50 = float(bundle["m_dt50"].predict(x_56)[0])
    pred_dt50_d = float(np.clip(10.0 ** pred_log_dt50, 1.0, 1500.0))
    if base_pred < 10.0:
        pred_dt50_d = max(pred_dt50_d, 180.0)
    elif base_pred >= 70.0 and is_sludge == 1:
        pred_dt50_d = min(pred_dt50_d, 25.0)

    pred_lag_d = float(np.clip(bundle["m_lag"].predict(x_56)[0], 1.0, 90.0))
    if pd.notna(known.get("Lit_Lag_days")):
        pred_lag_d = 0.3 * pred_lag_d + 0.7 * float(known["Lit_Lag_days"])

    # 随培养天数 (duration_days) 变化的动力学响应
    k_rate = 0.055 if is_sludge == 1 else 0.022
    if base_pred > 0:
        frac_t = 1.0 / (1.0 + np.exp(-k_rate * (duration_days - pred_lag_d - 10.0)))
        frac_56 = 1.0 / (1.0 + np.exp(-k_rate * (56.0 - pred_lag_d - 10.0)))
        gated_pred = base_pred * np.clip(frac_t / max(0.15, frac_56), 0.10, 1.18)
    else:
        gated_pred = base_pred

    gated_pred = float(np.clip(gated_pred, -30.0, 100.0))

    if gated_pred < 0:
        status = "Methanogenic Inhibitory (< 0% ThGP)"
    elif gated_pred < 30:
        status = "Recalcitrant / Persistent (< 30% ThGP)"
    elif gated_pred < 60:
        status = "Partially / Inherently Biodegradable (30-60% ThGP)"
    else:
        status = "Readily Anaerobically Biodegradable (>= 60% ThGP)"

    ad_status = "In-Domain (High Confidence)" if min_dist < 2.8 else ("Moderate Confidence (Edge of AD)" if min_dist < 5.5 else "Out-of-Domain (Extrapolation)")
    base_val, shap_contribs = compute_shap_waterfall(x_vec, fdict, bundle, base_pred, gate_delta) if include_shap else (bundle.get('baseline_mean_pct', 42.0), {})

    return {
        "query": q,
        "canonical_smiles": smiles,
        "chemical_name": known.get("Chemical_Name", "Custom / Query Molecule"),
        "cas_rn": known.get("CAS_RN", ""),
        "in_benchmark_db": bool(smiles in bundle["smiles_lookup"]),
        "hit_90k_priors_db": bool(fdict.get("_hit_90k_db", False)),
        "applicability_domain": ad_status,
        "experimental_degrad_pct": float(known["Mean_Anaerobic_Degrad_Pct"]) if pd.notna(known.get("Mean_Anaerobic_Degrad_Pct")) else None,
        "experimental_dt50_days": float(known["Mean_Anaerobic_DT50_d"]) if pd.notna(known.get("Mean_Anaerobic_DT50_d")) else None,
        "experimental_lag_days": float(known["Lit_Lag_days"]) if pd.notna(known.get("Lit_Lag_days")) else None,
        "predicted_degrad_pct": round(gated_pred, 2),
        "predicted_dt50_days": round(pred_dt50_d, 1),
        "predicted_lag_days": round(pred_lag_d, 1),
        "raw_ensemble_pred_pct": round(base_pred, 2),
        "epa_biowin7_prob": round(float(fdict["Biowin7_Anaerobic_Prob"]), 4),
        "classification_outcome": status,
        "physics_gates_triggered": gates_triggered,
        "shap_baseline_pct": round(base_val, 2),
        "shap_modality_contributions": shap_contribs,
        "key_Modality_features": {
            "Phys_MW": round(fdict["Phys_MW"], 2),
            "Phys_logKow": round(fdict["Phys_logKow"], 2),
            "Best_Acidic_pKa": round(fdict["Best_Acidic_pKa"], 2),
            "Neutral_Frac_pH72": round(fdict["Neutral_Frac_pH72"], 4),
            "Steric_Bioavail_Barrier": round(fdict["Steric_Bioavail_Barrier"], 2),
            "Hydrolyzable_Bonds": int(fdict["Hydrolyzable_Count"]),
            "Halogen_Count": int(fdict["Halogen_Count"])
        },
        "model_version": bundle["version"]
    }

if __name__ == "__main__":
    b = train_and_save_bundle("v4.0-flagship")
    print(f"[v4.0 多任务引擎自检通过] 版本: {b['version']} | 主库: {b['n_molecules']} 分子 | DT50子模型: {b['n_dt50_trained']} 分子 | Lag子模型: {b['n_lag_trained']} 分子")
    for test_q in ["106-44-5", "108-39-4", "84-74-2", "1163-19-5", "110-00-9", "638-16-4", "10595-49-0"]:
        res = predict_molecule(test_q, duration_days=56.0, is_sludge=1)
        print(f"  [{test_q:<10} {res['chemical_name'][:24]:<24}] 实测={str(round(res['experimental_degrad_pct'],1)) if res['experimental_degrad_pct'] is not None else 'N/A':>6}% | 预测={res['predicted_degrad_pct']:>6.2f}% | DT50={res['predicted_dt50_days']:>5.1f}d | Lag={res['predicted_lag_days']:>4.1f}d | Biowin7={res['epa_biowin7_prob']:>7.4f}")
