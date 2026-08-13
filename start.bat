@echo off
title Mumble
cd /d "%~dp0app"
"%~dp0venv\Scripts\python.exe" main.py
echo.
echo (App stopped. Close this window, or press any key.)
pause >nul
