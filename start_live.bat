@echo off
title QuantDix Live
cd /d "%~dp0"
echo Zaroori libraries check ho rahi hain...
python -m pip install -q requests pandas numpy yfinance matplotlib
python live_engine.py %*
echo.
echo Engine band ho gaya. Window band karne ke liye koi key dabayein.
pause
