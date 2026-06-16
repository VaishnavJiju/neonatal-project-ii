@echo off
title MAL-ED Clinical Nexus
echo.
echo  =============================================
echo   MAL-ED Clinical Nexus - Starting Services
echo  =============================================
echo.

:: Kill any existing processes on ports 8000 and 5173
echo  [1/4] Cleaning up old processes...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5173" ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>&1
timeout /t 1 /nobreak >nul

:: Start Backend
echo  [2/4] Starting Backend (FastAPI on port 8000)...
start "Nexus Backend" cmd /k "cd /d %~dp0backend && python main.py"
timeout /t 3 /nobreak >nul

:: Start Frontend
echo  [3/4] Starting Frontend (Vite on port 5173)...
start "Nexus Frontend" cmd /k "cd /d %~dp0ui && npm run dev"
timeout /t 3 /nobreak >nul

:: Open browser
echo  [4/4] Opening browser...
timeout /t 2 /nobreak >nul
start http://localhost:5173

echo.
echo  =============================================
echo   All services running!
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:5173
echo  =============================================
echo.
echo  Close this window or press any key to exit.
echo  (Backend and Frontend will keep running in their own windows)
pause >nul
