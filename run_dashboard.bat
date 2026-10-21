@echo off
title Healthcare IoT Pregel Clustering Dashboard
cd /d "%~dp0"
echo ==============================================================
echo Launching Healthcare IoT Pregel GraphX Dashboard...
echo Guide: Mr. R. SriRamu
echo ==============================================================
python -m streamlit run app.py
pause
