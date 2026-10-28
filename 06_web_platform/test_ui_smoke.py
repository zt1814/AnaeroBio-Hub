# -*- coding: utf-8 -*-
"""Headless smoke tests for the multi-page portal (Streamlit AppTest). Run from 06_web_platform/:
    python -m unittest test_ui_smoke -v
Uses the real model bundle + DB read-only; it does NOT submit community records (no writes to the production DB)."""
import os
import sys
import unittest

PLAT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PLAT)
os.chdir(PLAT)

try:
    from streamlit.testing.v1 import AppTest
except Exception:  # streamlit missing or too old
    AppTest = None

PAGES = [
    "web_portal.py",
    "ui/pages/predictor.py",
    "ui/pages/batch.py",
    "ui/pages/micropath.py",
    "ui/pages/database.py",
    "ui/pages/community.py",
    "ui/pages/api_hub.py",
]


def _html(at):
    return "\n".join(m.value for m in at.markdown)


@unittest.skipIf(AppTest is None, "streamlit.testing not available")
class TestPortalPages(unittest.TestCase):
    def _run(self, rel, **kw):
        at = AppTest.from_file(os.path.join(PLAT, rel), default_timeout=240).run()
        self.assertFalse(at.exception, f"{rel}: {[e.value for e in at.exception]}")
        return at

    def test_every_page_renders_without_exception(self):
        for rel in PAGES:
            with self.subTest(page=rel):
                self._run(rel)

    def test_predictor_default_shows_kpis_and_gate(self):
        at = self._run("ui/pages/predictor.py")
        self.assertIn("ab-kpi", _html(at))

    def test_predictor_inhibitor_is_flagged_negative(self):
        at = self._run("ui/pages/predictor.py")
        at.text_input(key="pred_q").set_value("110-00-9").run()  # furan
        self.assertFalse(at.exception)
        self.assertIn("ab-kpi bad", _html(at))

    def test_predictor_invalid_query_is_friendly(self):
        at = self._run("ui/pages/predictor.py")
        at.text_input(key="pred_q").set_value("not a smiles!!").run()
        self.assertFalse(at.exception)
        self.assertTrue(len(at.info) >= 1)

    def test_batch_runs_default_list(self):
        at = self._run("ui/pages/batch.py")
        at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertTrue(len(at.dataframe) >= 1)


# ───────────── portal hosting (FastAPI serves ../site at /portal/) ─────────────
try:
    from fastapi.testclient import TestClient
except Exception:
    TestClient = None


@unittest.skipIf(TestClient is None, "fastapi TestClient (httpx) not available")
class TestPortalHosting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import api_server
        cls.client = TestClient(api_server.app, follow_redirects=False)
        cls.has_site = os.path.isdir(os.path.join(os.path.dirname(PLAT), "site"))

    def test_root_still_200(self):
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_portal_redirects_to_trailing_slash(self):
        if not self.has_site:
            self.skipTest("site/ not present")
        r = self.client.get("/portal")
        self.assertIn(r.status_code, (301, 302, 307, 308))
        self.assertTrue(r.headers["location"].endswith("/portal/"))

    def test_portal_index_and_assets(self):
        if not self.has_site:
            self.skipTest("site/ not present")
        r = self.client.get("/portal/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("AnaeroBio-Hub", r.text)
        css = self.client.get("/portal/assets/style.css")
        self.assertEqual(css.status_code, 200)
        self.assertIn("css", css.headers["content-type"])
        self.assertEqual(self.client.get("/portal/assets/app.js").status_code, 200)


if __name__ == "__main__":
    unittest.main()
