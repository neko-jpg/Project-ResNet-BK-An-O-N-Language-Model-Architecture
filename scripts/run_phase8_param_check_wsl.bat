@echo off
REM Phase 8 Parameter Check - Windows to WSL Bridge
REM このバッチファイルはWindows側からWSL環境でパラメータチェックを実行します

echo ==========================================
echo Phase 8 Parameter Check (Windows to WSL)
echo ==========================================
echo.

REM Get the current directory in Windows format
set CURRENT_DIR=%CD%

REM Convert Windows path to WSL path
REM C:\dev\Project-ResNet-BK... -> /mnt/c/dev/Project-ResNet-BK...
set WSL_PATH=%CURRENT_DIR:C:\=/mnt/c/%
set WSL_PATH=%WSL_PATH:D:\=/mnt/d/%
set WSL_PATH=%WSL_PATH:E:\=/mnt/e/%
set WSL_PATH=%WSL_PATH:\=/%

echo Current directory: %CURRENT_DIR%
echo WSL path: %WSL_PATH%
echo.

REM Check if WSL is available
wsl --list >nul 2>&1
if errorlevel 1 (
    echo Error: WSL not found or not installed
    echo Please install WSL: https://docs.microsoft.com/en-us/windows/wsl/install
    pause
    exit /b 1
)

echo WSL is available
echo.

REM Make script executable and run in WSL
echo Running parameter check in WSL Ubuntu...
echo.

wsl -d Ubuntu bash -c "cd '%WSL_PATH%' && chmod +x scripts/run_phase8_param_check.sh && bash scripts/run_phase8_param_check.sh configs/phase8_max_push.yaml"

echo.
echo ==========================================
echo Parameter Check Complete!
echo ==========================================
echo.

pause
