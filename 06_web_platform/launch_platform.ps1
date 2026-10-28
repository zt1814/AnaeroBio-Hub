# AnaeroBio-Hub - one-click (re)start for Windows local development.
# Stops stale uvicorn/streamlit processes, starts API (8000) + web portal (8501), opens the browser.
# Usage:  powershell -ExecutionPolicy Bypass -File .\launch_platform.ps1
$ErrorActionPreference = "Continue"
$platDir = $PSScriptRoot
if (-not $platDir) { $platDir = Split-Path -Parent $MyInvocation.MyCommand.Path }
Set-Location $platDir

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  AnaeroBio-Hub: (re)starting API + Web Portal            " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 0) kill stale processes so new code is always the code that runs
Write-Host "[0/3] Stopping old uvicorn / streamlit processes ..." -ForegroundColor Yellow
foreach ($port in 8000, 8501) {
    Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | ForEach-Object {
        try { Stop-Process -Id $_.OwningProcess -Force -ErrorAction Stop; Write-Host "      stopped PID $($_.OwningProcess) (port $port)" } catch {}
    }
}
Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'uvicorn|streamlit' -and $_.ProcessId -ne $PID } |
    ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop; Write-Host "      stopped python PID $($_.ProcessId)" } catch {} }
Start-Sleep -Seconds 2

# Settings inherited by both child windows
$env:PORTAL_HOME_URL = "http://127.0.0.1:8000/portal/"   # "Home" link inside the Streamlit portal
$env:API_DOCS_URL    = "http://127.0.0.1:8000/docs"      # "API docs" link inside the Streamlit portal
$env:PUBLIC_BASE_URL = "http://127.0.0.1:8000"           # base URL shown in API examples
$env:ANAERO_DEBUG    = "1"                               # show real tracebacks in the page (local only)
$env:ENABLE_PUBLIC_SUBMISSION = "true"                   # local dev only: open the community sandbox (production default is closed)
# $env:ANBIOGATE_ADMIN_TOKEN = "choose-a-long-random-string"   # optional: enable curator approval locally

# 1) API
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$platDir'; python -m uvicorn api_server:app --host 127.0.0.1 --port 8000"
Write-Host "[1/3] API starting      -> http://127.0.0.1:8000/docs" -ForegroundColor Green

# 2) Web portal
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$platDir'; python -m streamlit run web_portal.py --server.port 8501 --server.headless true"
Write-Host "[2/3] Web portal starting -> http://localhost:8501" -ForegroundColor Green

# 3) wait for the API, then open both pages
Write-Host "[3/3] Waiting for the API to be ready ..." -ForegroundColor Yellow
$ready = $false
for ($i = 0; $i -lt 90; $i++) {
    try {
        $r = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:8000/api/v1/status" -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch {}
    Start-Sleep -Seconds 1
}
if ($ready) { Write-Host "      API is up." -ForegroundColor Green } else { Write-Host "      API not ready after 90 s - check the uvicorn window for errors." -ForegroundColor Red }
Start-Process "http://127.0.0.1:8000/portal/"
Start-Process "http://localhost:8501"
Write-Host ""
Write-Host "Landing page : http://127.0.0.1:8000/portal/"
Write-Host "Web portal   : http://localhost:8501"
Write-Host "API docs     : http://127.0.0.1:8000/docs"
