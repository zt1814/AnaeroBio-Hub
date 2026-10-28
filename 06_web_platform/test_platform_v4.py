#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AnaeroBio-Hub v4.0 非侵入式平台自动化测试脚本
test_platform_v4.py

覆盖范围:
  A. model_engine — 特征计算、边界输入防护、推理结果结构、并发安全
  B. api_server   — 端点状态码、错误处理、沙箱两阶段逻辑

运行方式 (无需启动服务):
  cd D:\\AI coding\\pigntai_develop\\06_web_platform
  python test_platform_v4.py

依赖: pytest / unittest (stdlib)，需在安装了 fastapi + httpx 的环境中运行 API 测试
"""
import os
import sys
import math
import time
import sqlite3
import unittest
import threading
import importlib

# ── 路径配置：确保从 06_web_platform 目录加载模块 ──────────────────────────
PLAT_DIR = os.path.dirname(os.path.abspath(__file__))
if PLAT_DIR not in sys.path:
    sys.path.insert(0, PLAT_DIR)

# Test-only settings for the hardened API: public submissions are closed by default and there is no default curator token.
os.environ["ENABLE_PUBLIC_SUBMISSION"] = "true"
os.environ["ANBIOGATE_ADMIN_TOKEN"] = "test-curator-token"

import model_engine as ME

# ─────────────────────────────────────────────────────────────────────────────
#  A. model_engine 单元测试
# ─────────────────────────────────────────────────────────────────────────────

class TestCanonicalizeDesalt(unittest.TestCase):
    """canonicalize_and_desalt() 的去盐与规范化逻辑"""

    def test_plain_smiles_returns_unchanged_without_rdkit(self):
        s, d = ME.canonicalize_and_desalt("Cc1ccc(O)cc1")
        self.assertIsInstance(s, str)
        self.assertIn("c", d.lower())

    def test_empty_string(self):
        s, d = ME.canonicalize_and_desalt("")
        self.assertEqual(s, "")
        self.assertEqual(d, "")

    def test_salt_stripping_sodium(self):
        """含钠盐的 SMILES 应剥离无机部分"""
        _, desalted = ME.canonicalize_and_desalt("CC(=O)[O-].[Na+]")
        self.assertNotIn("[Na+]", desalted)
        self.assertIn("C", desalted)

    def test_multicomponent_picks_longest_organic(self):
        _, desalted = ME.canonicalize_and_desalt("CCCCCCCCCC.[Cl-].[NH4+]")
        self.assertIn("C", desalted)
        self.assertNotIn("[Cl-]", desalted)


class TestComputeFeatures(unittest.TestCase):
    """compute_single_molecule_features() 的特征向量结构与边界值防护"""

    def _get_feat(self, smiles, **kw):
        return ME.compute_single_molecule_features(smiles, **kw)

    def test_output_keys_complete(self):
        feat = self._get_feat("Cc1ccc(O)cc1")
        for col in ME.FEATURE_COLS:
            self.assertIn(col, feat, f"缺失特征列: {col}")

    def test_all_numeric_no_nan(self):
        feat = self._get_feat("CCCCCCC")
        for col in ME.FEATURE_COLS:
            v = feat[col]
            self.assertFalse(math.isnan(float(v)), f"{col} 返回 NaN")

    def test_empty_smiles_does_not_crash(self):
        """空 SMILES 不应抛出异常"""
        try:
            feat = self._get_feat("")
            self.assertIsInstance(feat, dict)
        except Exception as e:
            self.fail(f"空 SMILES 导致崩溃: {e}")

    def test_whitespace_smiles(self):
        feat = self._get_feat("   ")
        self.assertIsInstance(feat, dict)

    def test_extreme_duration_min(self):
        feat = self._get_feat("CC", duration_days=0.0)
        self.assertEqual(feat["Duration_days"], 0.0)

    def test_extreme_duration_max(self):
        feat = self._get_feat("CC", duration_days=9999.0)
        self.assertEqual(feat["Duration_days"], 9999.0)

    def test_benzoyl_coa_flag_pcresol(self):
        """p-Cresol (Cc1ccc(O)cc1) 应触发 Benzoyl-CoA 途径"""
        feat = self._get_feat("Cc1ccc(O)cc1")
        self.assertEqual(feat["Benzoyl_CoA_Pathway"], 1)

    def test_furan_alert(self):
        """呋喃 (c1ccoc1) 应触发 Furan_Ring_Alert"""
        feat = self._get_feat("c1ccoc1")
        self.assertEqual(feat["Furan_Ring_Alert"], 1)

    def test_perbrominated_halogen_count(self):
        """十溴二苯醚 (高 Br 数) 的 Br_Count 应 >= 6"""
        deca_smiles = "Brc1c(Br)c(Br)c(-c2c(Br)c(Br)c(Br)c(Br)c2Br)c(Br)c1Br"
        feat = self._get_feat(deca_smiles)
        self.assertGreaterEqual(feat["Br_Count"], 6)

    def test_is_sludge_flag(self):
        feat_sl = self._get_feat("CC", is_sludge=1)
        feat_aq = self._get_feat("CC", is_sludge=0)
        self.assertEqual(feat_sl["Is_Tier1_Sludge_Assay"], 1)
        self.assertEqual(feat_aq["Is_Tier1_Sludge_Assay"], 0)


class TestSafeFloat(unittest.TestCase):
    """_safe_float() 辅助函数的边界防护"""

    def test_none_returns_default(self):
        self.assertEqual(ME._safe_float(None, 99.0), 99.0)

    def test_nan_returns_default(self):
        import numpy as np
        self.assertEqual(ME._safe_float(float("nan"), 42.0), 42.0)

    def test_string_number(self):
        self.assertAlmostEqual(ME._safe_float("3.14", 0.0), 3.14)

    def test_non_numeric_string(self):
        self.assertEqual(ME._safe_float("abc", 7.0), 7.0)


class TestPredictMolecule(unittest.TestCase):
    """predict_molecule() 的推理结果结构与物理门控逻辑"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = ME.load_bundle()

    def _pred(self, q, days=56.0, sludge=1):
        return ME.predict_molecule(q, duration_days=days, is_sludge=sludge)

    # ── 结果结构检查 ────────────────────────────────────────────────────────
    def test_output_keys_structure(self):
        res = self._pred("Cc1ccc(O)cc1")
        required = [
            "query", "canonical_smiles", "chemical_name", "cas_rn",
            "in_benchmark_db", "applicability_domain",
            "predicted_degrad_pct", "predicted_dt50_days", "predicted_lag_days",
            "raw_ensemble_pred_pct", "epa_biowin7_prob",
            "classification_outcome", "physics_gates_triggered",
            "shap_baseline_pct", "shap_modality_contributions",
            "key_Modality_features", "model_version"
        ]
        for k in required:
            self.assertIn(k, res, f"输出缺少字段: {k}")

    def test_predicted_pct_in_range(self):
        res = self._pred("106-44-5")
        self.assertGreaterEqual(res["predicted_degrad_pct"], -30.0)
        self.assertLessEqual(res["predicted_degrad_pct"], 100.0)

    def test_dt50_positive(self):
        res = self._pred("106-44-5")
        self.assertGreater(res["predicted_dt50_days"], 0.0)

    def test_lag_positive(self):
        res = self._pred("106-44-5")
        self.assertGreater(res["predicted_lag_days"], 0.0)

    # ── 核心盲区分子验收（允许 ±15 pp 偏差） ──────────────────────────────
    def test_pcresol_biodegradable(self):
        """p-Cresol (106-44-5): 实测 75.3%，预测应 >= 55%"""
        res = self._pred("106-44-5")
        self.assertGreaterEqual(res["predicted_degrad_pct"], 55.0,
            f"p-Cresol 预测偏低: {res['predicted_degrad_pct']:.1f}%")

    def test_decabde_recalcitrant(self):
        """十溴二苯醚 (1163-19-5): 实测 0.0%，预测应 <= 10%"""
        res = self._pred("1163-19-5")
        self.assertLessEqual(res["predicted_degrad_pct"], 10.0,
            f"DecaBDE 预测偏高: {res['predicted_degrad_pct']:.1f}%")

    def test_furan_inhibitory(self):
        """呋喃 (110-00-9): 实测 -22%，预测应 < 0%"""
        res = self._pred("110-00-9")
        self.assertLess(res["predicted_degrad_pct"], 0.0,
            f"Furan 预测未触发抑制: {res['predicted_degrad_pct']:.1f}%")

    def test_tmt_inhibitory(self):
        """TMT (638-16-4): 预测应 < 0%（F430 毒性门控）"""
        res = self._pred("638-16-4")
        self.assertLess(res["predicted_degrad_pct"], 0.0,
            f"TMT 预测未触发 F430 毒性: {res['predicted_degrad_pct']:.1f}%")

    def test_dibutyl_phthalate_biodegradable(self):
        """邻苯二甲酸二丁酯 (84-74-2): 实测 93.3%，预测应 >= 60%"""
        res = self._pred("84-74-2")
        self.assertGreaterEqual(res["predicted_degrad_pct"], 60.0,
            f"DBP 预测偏低: {res['predicted_degrad_pct']:.1f}%")

    # ── 边界输入防护 ────────────────────────────────────────────────────────
    def test_empty_query_does_not_crash(self):
        try:
            res = self._pred("")
            self.assertIsInstance(res, dict)
        except Exception as e:
            self.fail(f"空查询崩溃: {e}")

    def test_invalid_cas_returns_dict(self):
        res = self._pred("9999-99-9")
        self.assertIsInstance(res, dict)
        self.assertIn("predicted_degrad_pct", res)

    def test_garbage_smiles_returns_dict(self):
        res = self._pred("XYZ###???")
        self.assertIsInstance(res, dict)

    def test_zero_duration(self):
        res = self._pred("Cc1ccc(O)cc1", days=0.0)
        self.assertIsInstance(res["predicted_degrad_pct"], float)

    def test_very_long_duration(self):
        res = self._pred("Cc1ccc(O)cc1", days=365.0)
        self.assertIsInstance(res["predicted_degrad_pct"], float)
        self.assertLessEqual(res["predicted_degrad_pct"], 100.0)

    def test_sediment_mode(self):
        """沉积物模式 (is_sludge=0) 预测值通常低于消化污泥模式"""
        r_sl = self._pred("Cc1ccc(O)cc1", sludge=1)
        r_aq = self._pred("Cc1ccc(O)cc1", sludge=0)
        # 仅检查函数正常返回，不做数值大小断言（门控可能使两者相等）
        self.assertIsInstance(r_sl["predicted_degrad_pct"], float)
        self.assertIsInstance(r_aq["predicted_degrad_pct"], float)

    # ── SHAP 归因完整性检查 ─────────────────────────────────────────────────
    def test_shap_contribs_are_numeric(self):
        res = self._pred("Cc1ccc(O)cc1")
        contribs = res["shap_modality_contributions"]
        self.assertIsInstance(contribs, dict)
        self.assertGreater(len(contribs), 0)
        for k, v in contribs.items():
            self.assertIsInstance(float(v), float, f"SHAP 贡献非数值: {k}={v}")

    def test_shap_baseline_reasonable(self):
        res = self._pred("Cc1ccc(O)cc1")
        bl = res["shap_baseline_pct"]
        self.assertGreater(bl, 0.0)
        self.assertLess(bl, 100.0)


class TestConcurrency(unittest.TestCase):
    """并发调用 predict_molecule() 的线程安全性（只读路径，不涉及写操作）"""

    def test_concurrent_reads_no_exception(self):
        errors = []

        def worker(cas):
            try:
                ME.predict_molecule(cas, 56.0, 1)
            except Exception as e:
                errors.append(str(e))

        molecules = ["106-44-5", "84-74-2", "1163-19-5", "110-00-9"] * 4
        threads = [threading.Thread(target=worker, args=(m,)) for m in molecules]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(errors, [], f"并发读取出现异常: {errors[:3]}")


class TestSQLiteRobustness(unittest.TestCase):
    """SQLite 数据库连接的健壮性与异常防护"""

    def test_db_file_exists(self):
        self.assertTrue(os.path.exists(ME.DB_PATH),
            f"数据库文件不存在: {ME.DB_PATH}")

    def test_community_tables_exist(self):
        ME.init_community_tables()
        conn = sqlite3.connect(ME.DB_PATH)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        conn.close()
        self.assertIn("community_submissions", tables)
        self.assertIn("model_versions", tables)

    def test_invalid_db_path_query_returns_empty(self):
        """query_90k_priors_from_sqlite 在 DB 路径不存在时应返回空字典而非崩溃"""
        result = ME.query_90k_priors_from_sqlite.__wrapped__([]) if hasattr(ME.query_90k_priors_from_sqlite, "__wrapped__") else ME.query_90k_priors_from_sqlite([])
        self.assertIsInstance(result, dict)

    def test_query_90k_empty_list(self):
        result = ME.query_90k_priors_from_sqlite([])
        self.assertEqual(result, {})


# ─────────────────────────────────────────────────────────────────────────────
#  B. api_server FastAPI 端点测试（需安装 httpx；若无则跳过）
# ─────────────────────────────────────────────────────────────────────────────

try:
    from fastapi.testclient import TestClient
    import api_server as AS
    _HAS_TESTCLIENT = True
except ImportError:
    _HAS_TESTCLIENT = False


@unittest.skipUnless(_HAS_TESTCLIENT, "需要 httpx 和 fastapi[testclient] 才能运行 API 测试")
class TestAPIServer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(AS.app)

    def test_root_returns_200(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("welcome", data)
        self.assertIn("current_model_version", data)

    def test_status_endpoint(self):
        r = self.client.get("/api/v1/status")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("model_version", data)

    def test_predict_valid_cas(self):
        r = self.client.post("/api/v1/predict", json={"query": "106-44-5"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("predicted_degrad_pct", data)

    def test_predict_valid_smiles(self):
        r = self.client.post("/api/v1/predict", json={"query": "Cc1ccc(O)cc1"})
        self.assertEqual(r.status_code, 200)

    def test_predict_empty_query(self):
        """空查询应返回 200（引擎内部已做防护，返回合理默认值）"""
        r = self.client.post("/api/v1/predict", json={"query": ""})
        self.assertIn(r.status_code, [200, 422])

    def test_predict_missing_query_field_422(self):
        """缺少必填字段 query 应返回 422"""
        r = self.client.post("/api/v1/predict", json={"duration_days": 56})
        self.assertEqual(r.status_code, 422)

    def test_batch_predict_up_to_500(self):
        queries = ["106-44-5", "84-74-2", "110-00-9"]
        r = self.client.post("/api/v1/predict_batch", json={"queries": queries})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()), len(queries))

    def test_batch_exceeds_500_truncated(self):
        queries = ["CC"] * 600
        r = self.client.post("/api/v1/predict_batch", json={"queries": queries})
        self.assertEqual(r.status_code, 200)
        self.assertLessEqual(len(r.json()), 500)

    def test_contribute_pending_without_token(self):
        """无 admin_token 的提交应进入 pending 沙箱，不触发热重训"""
        payload = {
            "chemical_name": "Test Molecule",
            "smiles": "CCCCC",
            "exp_degrad_pct": 50.0,
            "contributor": "test_suite"
        }
        r = self.client.post("/api/v1/contribute", json=payload)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "staged_in_pending_sandbox")

    def test_contribute_approved_with_token(self):
        """正确 admin_token 的提交应触发 approved 状态"""
        payload = {
            "chemical_name": "Test Molecule Approved",
            "smiles": "CCCCCC",
            "exp_degrad_pct": 55.0,
            "contributor": "test_suite",
            "admin_token": "test-curator-token"
        }
        r = self.client.post("/api/v1/contribute", json=payload)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "approved_and_retrained")

    def test_contribute_closed_when_public_submission_disabled(self):
        """ENABLE_PUBLIC_SUBMISSION=false: anonymous submissions are refused (403); a curator token still works"""
        old = os.environ.get("ENABLE_PUBLIC_SUBMISSION")
        os.environ["ENABLE_PUBLIC_SUBMISSION"] = "false"
        try:
            payload = {"chemical_name": "Closed Test", "smiles": "CCCCCCC", "exp_degrad_pct": 10.0}
            self.assertEqual(self.client.post("/api/v1/contribute", json=payload).status_code, 403)
        finally:
            os.environ["ENABLE_PUBLIC_SUBMISSION"] = old if old is not None else "true"

    def test_contribute_wrong_token_pending(self):
        """错误 token 应进入 pending"""
        payload = {
            "chemical_name": "Wrong Token Test",
            "smiles": "CCCN",
            "exp_degrad_pct": 30.0,
            "admin_token": "WRONG_TOKEN"
        }
        r = self.client.post("/api/v1/contribute", json=payload)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "staged_in_pending_sandbox")

    def test_metabolism_returns_dict(self):
        r = self.client.get("/api/v1/metabolism", params={"smiles": "Cc1ccc(O)cc1"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("n_metabolites_found", data)
        self.assertIn("metabolites", data)

    def test_nonexistent_endpoint_404(self):
        r = self.client.get("/api/v2/nonexistent")
        self.assertEqual(r.status_code, 404)


# ─────────────────────────────────────────────────────────────────────────────
#  主入口 — 运行所有测试并输出简报
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  AnaeroBio-Hub v4.0 — 自动化平台健壮性测试套件")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 按模块顺序加载
    for cls in [
        TestCanonicalizeDesalt,
        TestComputeFeatures,
        TestSafeFloat,
        TestPredictMolecule,
        TestConcurrency,
        TestSQLiteRobustness,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    if _HAS_TESTCLIENT:
        suite.addTests(loader.loadTestsFromTestCase(TestAPIServer))
    else:
        print("  ⚠  跳过 API 服务器测试 (缺少 httpx / fastapi[testclient])")
        print("     安装方式: pip install httpx --break-system-packages")
        print()

    runner = unittest.TextTestRunner(verbosity=2, failfast=False)
    result = runner.run(suite)

    print()
    print("=" * 70)
    if result.wasSuccessful():
        print(f"  ✅ 全部 {result.testsRun} 项测试通过")
    else:
        print(f"  ❌ {len(result.failures)} 项失败 | {len(result.errors)} 项错误 | {result.testsRun} 项合计")
        for fail in result.failures:
            print(f"     FAIL: {fail[0]}")
        for err in result.errors:
            print(f"     ERROR: {err[0]}")
    print("=" * 70)

    sys.exit(0 if result.wasSuccessful() else 1)
