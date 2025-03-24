@echo off
REM Remove existing environment if present
rmdir /s /q venv

REM Create fresh environment
python -m venv venv

REM Activate environment
call venv\Scripts\activate.bat

REM Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
