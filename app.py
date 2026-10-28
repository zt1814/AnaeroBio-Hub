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

# 切换工作目录至 06_web_platform 并直接执行主界面 web_portal.py
os.chdir(PLAT_DIR)
runpy.run_path(PORTAL_FILE, run_name="__main__")
