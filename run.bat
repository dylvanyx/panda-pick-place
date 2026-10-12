@echo off
chcp 65001 >nul
set "PY=D:\MuJoCo_Projects\mujoco_env\python.exe"
set "SCRIPT=D:\MuJoCo_Projects\test\1.py"
set "SCENE=%~1"
if "%SCENE%"=="" set "SCENE=D:\MuJoCo_Projects\test\franka_emika_panda\scene_pick.xml"
echo [run] %SCENE%
"%PY%" "%SCRIPT%" "%SCENE%"
echo.
echo [exit] code=%errorlevel%
pause
