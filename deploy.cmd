@echo off
REM ============================================================================
REM CENOP Deployment Script for laguna.ku.lt
REM ============================================================================
REM 
REM This script deploys CENOP to the Shiny Server on laguna.ku.lt
REM User: razinka
REM 
REM Prerequisites:
REM   - SSH access to laguna.ku.lt as razinka
REM   - Python 3.10+ installed on server
REM   - Shiny Server for Python configured
REM
REM Usage:
REM   deploy.cmd                    - Full deployment
REM   deploy.cmd --sync-only        - Only sync files (no install)
REM   deploy.cmd --restart-only     - Only restart Shiny Server
REM
REM ============================================================================

setlocal enabledelayedexpansion

REM Configuration
set SERVER=laguna.ku.lt
set USER=razinka
set REMOTE_PATH=/srv/shiny-server/cenop
set LOCAL_PATH=%~dp0
set APP_NAME=cenop

REM Parse arguments
set SYNC_ONLY=0
set RESTART_ONLY=0

if "%1"=="--sync-only" set SYNC_ONLY=1
if "%1"=="--restart-only" set RESTART_ONLY=1

echo.
echo ============================================================
echo   CENOP Deployment to %SERVER%
echo ============================================================
echo   User: %USER%
echo   Remote Path: %REMOTE_PATH%
echo   Local Path: %LOCAL_PATH%
echo ============================================================
echo.

REM Check if restart only
if %RESTART_ONLY%==1 (
    echo [STEP] Restarting Shiny Server...
    ssh %USER%@%SERVER% "sudo systemctl restart shiny-server"
    if errorlevel 1 (
        echo [ERROR] Failed to restart Shiny Server
        exit /b 1
    )
    echo [OK] Shiny Server restarted successfully
    goto :end
)

REM Step 1: Create remote directory if needed
echo [STEP 1/6] Ensuring remote directory exists...
ssh %USER%@%SERVER% "mkdir -p %REMOTE_PATH%"
if errorlevel 1 (
    echo [ERROR] Failed to create remote directory
    exit /b 1
)
echo [OK] Remote directory ready

REM Step 2: Sync application files using rsync (via SSH)
echo.
echo [STEP 2/6] Syncing application files...
echo   Excluding: __pycache__, .git, .pytest_cache, output/, *.pyc

REM Using scp for Windows (or rsync if available via WSL/Git Bash)
REM Option A: Using rsync via Git Bash or WSL
where rsync >nul 2>&1
if %errorlevel%==0 (
    rsync -avz --delete ^
        --exclude "__pycache__" ^
        --exclude "*.pyc" ^
        --exclude ".git" ^
        --exclude ".pytest_cache" ^
        --exclude "output" ^
        --exclude ".venv" ^
        --exclude "*.egg-info" ^
        "%LOCAL_PATH%" %USER%@%SERVER%:%REMOTE_PATH%/
) else (
    echo   [INFO] rsync not found, using scp instead
    echo   [INFO] For better sync, install Git Bash or use WSL
    
    REM Create a temporary exclude list and use scp
    scp -r "%LOCAL_PATH%app.py" %USER%@%SERVER%:%REMOTE_PATH%/
    scp -r "%LOCAL_PATH%requirements.txt" %USER%@%SERVER%:%REMOTE_PATH%/
    scp -r "%LOCAL_PATH%pyproject.toml" %USER%@%SERVER%:%REMOTE_PATH%/
    scp -r "%LOCAL_PATH%src" %USER%@%SERVER%:%REMOTE_PATH%/
    scp -r "%LOCAL_PATH%server" %USER%@%SERVER%:%REMOTE_PATH%/
    scp -r "%LOCAL_PATH%ui" %USER%@%SERVER%:%REMOTE_PATH%/
    scp -r "%LOCAL_PATH%static" %USER%@%SERVER%:%REMOTE_PATH%/
    scp -r "%LOCAL_PATH%data" %USER%@%SERVER%:%REMOTE_PATH%/
)

if errorlevel 1 (
    echo [ERROR] Failed to sync files
    exit /b 1
)
echo [OK] Files synced

if %SYNC_ONLY%==1 (
    echo.
    echo [DONE] Sync-only mode complete
    goto :end
)

REM Step 3: Create/update virtual environment
echo.
echo [STEP 3/6] Setting up Python virtual environment...
ssh %USER%@%SERVER% "cd %REMOTE_PATH% && python3 -m venv .venv"
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    exit /b 1
)
echo [OK] Virtual environment ready

REM Step 4: Install dependencies
echo.
echo [STEP 4/6] Installing Python dependencies...
ssh %USER%@%SERVER% "cd %REMOTE_PATH% && source .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt"
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    exit /b 1
)
echo [OK] Dependencies installed

REM Step 5: Install CENOP package in development mode
echo.
echo [STEP 5/6] Installing CENOP package...
ssh %USER%@%SERVER% "cd %REMOTE_PATH% && source .venv/bin/activate && pip install -e ."
if errorlevel 1 (
    echo [ERROR] Failed to install CENOP package
    exit /b 1
)
echo [OK] CENOP package installed

REM Step 6: Restart Shiny Server
echo.
echo [STEP 6/6] Restarting Shiny Server...
ssh %USER%@%SERVER% "sudo systemctl restart shiny-server"
if errorlevel 1 (
    echo [WARNING] Could not restart Shiny Server (may need sudo privileges)
    echo [INFO] Please restart manually: sudo systemctl restart shiny-server
) else (
    echo [OK] Shiny Server restarted
)

:end
echo.
echo ============================================================
echo   Deployment Complete!
echo ============================================================
echo   Application URL: https://%SERVER%/%APP_NAME%/
echo   Server logs: /var/log/shiny-server/
echo ============================================================
echo.

endlocal
