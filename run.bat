@echo off
chcp 65001 >nul
set "PY=D:\MuJoCo_Projects\mujoco_env\python.exe"
if not exist "%PY%" set "PY=python"
set "SCRIPT=%~dp0achievement\ctrl_3G_random.py"
echo [run] %SCRIPT%
"%PY%" "%SCRIPT%"
echo.
echo [exit] code=%errorlevel%
pause
