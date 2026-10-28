#!/bin/bash
# AnaeroBio-Hub v4.0 — runs FastAPI (uvicorn) + Streamlit; logs go to stdout (docker compose logs -f hub)
set -u
cd /app/06_web_platform

uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 2 \
    --proxy-headers --forwarded-allow-ips="*" &
API_PID=$!

# Streamlit is served under /app (nginx: /app/ → :8501) so the public site has one origin.
streamlit run web_portal.py --server.port 8501 --server.address 0.0.0.0 \
    --server.headless true --server.baseUrlPath app &
UI_PID=$!

trap 'kill $API_PID $UI_PID 2>/dev/null; wait; exit 0' TERM INT
wait -n $API_PID $UI_PID
CODE=$?
echo "[entrypoint] a service exited (code $CODE) — stopping container so Docker restarts it"
kill $API_PID $UI_PID 2>/dev/null
exit $CODE
