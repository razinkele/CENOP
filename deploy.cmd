@echo off
REM ============================================================================
REM CENOP-JASMINE Deployment Script for laguna.ku.lt
REM ============================================================================
REM
REM This script deploys CENOP-JASMINE to the Shiny Server on laguna.ku.lt
REM User: razinka
REM App directory: /home/razinka/cenjas (symlinked from /srv/shiny-server/cenjas)
REM
REM Prerequisites:
REM   - SSH access to laguna.ku.lt as razinka
REM   - Git repository cloned at /home/razinka/cenjas
REM   - Python 3.10+ with venv at /home/razinka/cenjas/venv
REM   - Shiny Server configured with cenjas app
REM
REM ============================================================================

setlocal enabledelayedexpansion

REM Configuration
set SERVER=laguna.ku.lt
set USER=razinka
set APP_DIR=/home/razinka/cenjas
set SHINY_LINK=/srv/shiny-server/cenjas
set APP_NAME=cenjas
set BRANCH=main

:menu
cls
echo.
echo ============================================================
echo   CENOP-JASMINE Deployment to %SERVER%
echo ============================================================
echo   User: %USER%
echo   App Directory: %APP_DIR%
echo   Branch: %BRANCH%
echo ============================================================
echo.
echo   Select an option:
echo.
echo   [1] Full deployment (pull, install, permissions, restart)
echo   [2] Pull latest changes only
echo   [3] Update dependencies only
echo   [4] Fix permissions for shiny user only
echo   [5] Restart Shiny Server only
echo   [6] View server logs
echo   [7] Check application status
echo   [0] Exit
echo.
echo ============================================================
echo.

set /p choice="Enter your choice (0-7): "

if "%choice%"=="1" goto :full_deploy
if "%choice%"=="2" goto :pull_only
if "%choice%"=="3" goto :update_deps
if "%choice%"=="4" goto :fix_permissions
if "%choice%"=="5" goto :restart_server
if "%choice%"=="6" goto :view_logs
if "%choice%"=="7" goto :check_status
if "%choice%"=="0" goto :exit

echo.
echo [ERROR] Invalid choice. Please enter a number between 0 and 7.
pause
goto :menu

REM ============================================================================
REM Option 1: Full Deployment
REM ============================================================================
:full_deploy
cls
echo.
echo ============================================================
echo   Full Deployment
echo ============================================================
echo.

call :check_ssh
if %ERRORLEVEL% neq 0 goto :menu

echo [STEP 1/5] Pulling latest changes from %BRANCH%...
ssh %USER%@%SERVER% "cd %APP_DIR% && git fetch origin && git reset --hard origin/%BRANCH%"
if errorlevel 1 (
    echo [ERROR] Failed to pull changes
    pause
    goto :menu
)
echo [OK] Repository updated
echo.

echo [STEP 2/5] Updating Python dependencies...
ssh %USER%@%SERVER% "cd %APP_DIR% && source venv/bin/activate && pip install -r requirements.txt --quiet"
if errorlevel 1 (
    echo [WARNING] Failed to update dependencies, continuing...
)
echo [OK] Dependencies updated
echo.

echo [STEP 3/5] Installing CENOP-JASMINE package...
ssh %USER%@%SERVER% "cd %APP_DIR% && source venv/bin/activate && pip install -e . --quiet"
if errorlevel 1 (
    echo [WARNING] Failed to install package, continuing...
)
echo [OK] Package installed
echo.

echo [STEP 4/5] Setting permissions for shiny user...
call :do_permissions
echo.

echo [STEP 5/5] Restarting Shiny Server...
ssh %USER%@%SERVER% "sudo systemctl restart shiny-server"
if errorlevel 1 (
    echo [WARNING] Could not restart Shiny Server
) else (
    echo [OK] Shiny Server restarted
)

echo.
echo ============================================================
echo   Deployment Complete!
echo   Application URL: https://%SERVER%/%APP_NAME%/
echo ============================================================
echo.
pause
goto :menu

REM ============================================================================
REM Option 2: Pull Only
REM ============================================================================
:pull_only
cls
echo.
echo ============================================================
echo   Pull Latest Changes
echo ============================================================
echo.

call :check_ssh
if %ERRORLEVEL% neq 0 goto :menu

echo Pulling latest changes from %BRANCH%...
ssh %USER%@%SERVER% "cd %APP_DIR% && git fetch origin && git reset --hard origin/%BRANCH%"
if errorlevel 1 (
    echo [ERROR] Failed to pull changes
) else (
    echo [OK] Repository updated successfully
)
echo.
pause
goto :menu

REM ============================================================================
REM Option 3: Update Dependencies
REM ============================================================================
:update_deps
cls
echo.
echo ============================================================
echo   Update Dependencies
echo ============================================================
echo.

call :check_ssh
if %ERRORLEVEL% neq 0 goto :menu

echo Updating Python dependencies...
ssh %USER%@%SERVER% "cd %APP_DIR% && source venv/bin/activate && pip install -r requirements.txt"
if errorlevel 1 (
    echo [ERROR] Failed to update dependencies
) else (
    echo [OK] Dependencies updated
)
echo.

echo Installing CENOP-JASMINE package...
ssh %USER%@%SERVER% "cd %APP_DIR% && source venv/bin/activate && pip install -e ."
if errorlevel 1 (
    echo [ERROR] Failed to install package
) else (
    echo [OK] Package installed
)
echo.
pause
goto :menu

REM ============================================================================
REM Option 4: Fix Permissions
REM ============================================================================
:fix_permissions
cls
echo.
echo ============================================================
echo   Fix Permissions for Shiny User
echo ============================================================
echo.

call :check_ssh
if %ERRORLEVEL% neq 0 goto :menu

call :do_permissions
echo.
pause
goto :menu

REM ============================================================================
REM Option 5: Restart Server
REM ============================================================================
:restart_server
cls
echo.
echo ============================================================
echo   Restart Shiny Server
echo ============================================================
echo.

call :check_ssh
if %ERRORLEVEL% neq 0 goto :menu

echo Restarting Shiny Server...
ssh %USER%@%SERVER% "sudo systemctl restart shiny-server"
if errorlevel 1 (
    echo [ERROR] Failed to restart Shiny Server
) else (
    echo [OK] Shiny Server restarted successfully
)
echo.
pause
goto :menu

REM ============================================================================
REM Option 6: View Logs
REM ============================================================================
:view_logs
cls
echo.
echo ============================================================
echo   Server Logs (last 50 lines)
echo ============================================================
echo.

call :check_ssh
if %ERRORLEVEL% neq 0 goto :menu

ssh %USER%@%SERVER% "sudo tail -50 /var/log/shiny-server.log 2>/dev/null || echo 'Could not read log file'"
echo.
echo ============================================================
pause
goto :menu

REM ============================================================================
REM Option 7: Check Status
REM ============================================================================
:check_status
cls
echo.
echo ============================================================
echo   Application Status
echo ============================================================
echo.

call :check_ssh
if %ERRORLEVEL% neq 0 goto :menu

echo [Git Status]
ssh %USER%@%SERVER% "cd %APP_DIR% && git log -1 --oneline && echo. && git status -s"
echo.

echo [Shiny Server Status]
ssh %USER%@%SERVER% "sudo systemctl status shiny-server --no-pager -l | head -15"
echo.

echo [Symlink Check]
ssh %USER%@%SERVER% "ls -la %SHINY_LINK%"
echo.

echo ============================================================
pause
goto :menu

REM ============================================================================
REM Exit
REM ============================================================================
:exit
echo.
echo Goodbye!
endlocal
exit /b 0

REM ============================================================================
REM Helper: Check SSH availability
REM ============================================================================
:check_ssh
where ssh >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] SSH not found. Please install OpenSSH or use Git Bash.
    pause
    exit /b 1
)
exit /b 0

REM ============================================================================
REM Helper: Set permissions for shiny user
REM ============================================================================
:do_permissions
echo Setting directory permissions (755)...
ssh %USER%@%SERVER% "find %APP_DIR% -type d -exec chmod 755 {} \;"

echo Setting file permissions (644)...
ssh %USER%@%SERVER% "find %APP_DIR% -type f -exec chmod 644 {} \;"

echo Setting executable permissions for venv scripts...
ssh %USER%@%SERVER% "chmod 755 %APP_DIR%/venv/bin/*"

echo Ensuring shiny user can access home directory...
ssh %USER%@%SERVER% "chmod 755 /home/razinka"

echo Verifying symlink...
ssh %USER%@%SERVER% "ls -la %SHINY_LINK% 2>/dev/null || echo '[WARNING] Symlink not found'"

echo [OK] Permissions configured for shiny user access
exit /b 0
