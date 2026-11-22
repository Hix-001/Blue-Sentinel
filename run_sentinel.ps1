# ==============================================================================
# BLUE SENTINEL - Unified System Launcher
# Spins up FastAPI Python ML Backend & Vite/CesiumJS 3D Tactical Frontend
# ==============================================================================

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "⚓ INITIALIZING BLUE SENTINEL C4ISR PLATFORM" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

# 1. Start Python FastAPI Backend (Port 8000)
Write-Host "[1/2] Starting Python LSTM Telemetry Service on http://127.0.0.1:8000..." -ForegroundColor Green
$backendProcess = Start-Process powershell -ArgumentList "-NoExit", "-Command", "$env:KERAS_BACKEND='torch'; uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload" -PassThru

# 2. Start Vite CesiumJS Frontend (Port 5173)
Write-Host "[2/2] Starting 3D Tactical Console on http://localhost:5173..." -ForegroundColor Green
$frontendProcess = Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; npm run dev" -PassThru

Write-Host "Both processes launched in parallel windows." -ForegroundColor Yellow
Write-Host "Access Tactical Console: http://localhost:5173" -ForegroundColor Cyan
