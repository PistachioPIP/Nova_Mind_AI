@echo off
cd /d "%~dp0"

echo [NovaMind AI] Checking virtual environment...
if not exist ".venv" (
    echo [NovaMind AI] Creating virtual environment (.venv)...
    python -m venv .venv
    call .venv\Scripts\activate
    echo [NovaMind AI] Installing dependencies...
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate
)

echo [NovaMind AI] Starting application...
python -u main.py
pause
