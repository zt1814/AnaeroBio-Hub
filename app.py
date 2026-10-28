#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time
import socket
import subprocess
import runpy
import streamlit as st

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAT_DIR = os.path.join(ROOT_DIR, "06_web_platform")
PORTAL_FILE = os.path.join(PLAT_DIR, "web_portal.py")

for p in (PLAT_DIR, ROOT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["PYTHONPATH"] = f"{PLAT_DIR}{os.pathsep}{ROOT_DIR}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
os.environ.setdefault("ENABLE_PUBLIC_SUBMISSION", "false")
os.environ.setdefault("ANAEROBIO_API_URL", "http://127.0.0.1:8000")

# 核心修复 1：在根目录建立 ui -> 06_web_platform/ui 的符号链接，让 st.Page("ui/pages/...") 直接命中
root_ui_link = os.path.join(ROOT_DIR, "ui")
plat_ui_dir = os.path.join(PLAT_DIR, "ui")
if not os.path.exists(root_ui_link) and os.path.exists(plat_ui_dir):
    try:
        os.symlink(plat_ui_dir, root_ui_link)
    except Exception:
        pass

# 核心修复 2：包装 st.Page，若传入相对路径则自动拼接到 06_web_platform 下（双重保险）
_orig_st_page = st.Page
def _patched_st_page(page, *args, **kwargs):
    if isinstance(page, str) and not os.path.isabs(page):
        candidate = os.path.join(PLAT_DIR, page)
        if os.path.exists(candidate):
            page = candidate
    return _orig_st_page(page, *args, **kwargs)
st.Page = _patched_st_page

def _is_api_up() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", 8000)) == 0

@st.cache_resource(show_spinner="Starting AnBioGate v4.0-clean Mechanistic & ML Engine...")
def ensure_fastapi_backend():
    if not _is_api_up():
        subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api_server:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=PLAT_DIR,
            env=os.environ.copy(),
        )
        for _ in range(40):
            if _is_api_up():
                break
            time.sleep(0.5)
    return True

ensure_fastapi_backend()

os.chdir(PLAT_DIR)
runpy.run_path(PORTAL_FILE, run_name="__main__")
