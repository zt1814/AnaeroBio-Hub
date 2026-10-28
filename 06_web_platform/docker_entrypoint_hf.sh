#!/bin/bash
# AnaeroBio-Hub v4.0 — Hugging Face Spaces entrypoint: uvicorn (8000) + Streamlit (8501) behind nginx (7860).
set -u
cd /app/06_web_platform

uvicorn api_server:app --host 127.0.0.1 --port 8000 --workers 1 \
    --proxy-headers --forwarded-allow-ips="*" &
API_PID=$!

# HF serves the Space inside an iframe on huggingface.co, so XSRF/CORS checks for the websocket are relaxed here.
streamlit run web_portal.py --server.port 8501 --server.address 127.0.0.1 \
    --server.headless true --server.baseUrlPath app \
    --server.enableXsrfProtection false --server.enableCORS false &
UI_PID=$!

nginx -e /dev/stderr -c /app/06_web_platform/nginx.hf.conf -g 'daemon off;' &
NGX_PID=$!

trap 'kill $API_PID $UI_PID $NGX_PID 2>/dev/null; wait; exit 0' TERM INT
wait -n $API_PID $UI_PID $NGX_PID
CODE=$?
echo "[entrypoint] a service exited (code $CODE) — stopping container so the platform restarts it"
kill $API_PID $UI_PID $NGX_PID 2>/dev/null
exit $CODE
