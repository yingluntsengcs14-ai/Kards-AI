@echo off
cd /d "%~dp0"
python tools\demo\server.py --model runs\champion.pt
pause
