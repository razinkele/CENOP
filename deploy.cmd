@echo off
REM ============================================================================
REM CENOP-JASMINE Deployment Script for laguna.ku.lt
REM ============================================================================
REM
REM Updates the CENOP app on the Shiny Server at laguna.ku.lt from Windows.
REM User: razinka
REM App directory: /home/razinka/cenop/CENOP (this repository's working tree)
REM Served at:     /srv/shiny-server/cenop and /srv/shiny-server/cenjas (both
REM                symlinks to the app directory)
REM Python:        /opt/micromamba/envs/shiny/bin/python3 (shared env owned by
REM                the shiny user; apps run as shiny)
REM
REM The app directory is also the development checkout, so this script never
REM resets it: pulls are fast-forward only and refuse to run unless the
REM checkout is on %BRANCH%.
REM
REM Prerequisites:
REM   - SSH access to laguna.ku.lt as razinka (OpenSSH or Git Bash)
REM   - sudo rights on the server (prompted for over ssh -t)
REM
REM ============================================================================

setlocal enabledelayedexpansion

REM Configuration
set SERVER=laguna.ku.lt
set USER=razinka
set APP_DIR=/home/razinka/cenop/CENOP
set SHINY_LINKS=/srv/shiny-server/cenop /srv/shiny-server/cenjas
set APP_NAMES=cenop cenjas
set BRANCH=CENOP-JASMINE
set PYTHON=/opt/micromamba/envs/shiny/bin/python3
set REQ_TMP=/tmp/cenop-runtime-requirements.txt

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
echo   [1] Full deployment (pull, dependencies, permissions, restart)
echo   [2] Pull latest changes only
echo   [3] Install missing runtime dependencies only
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

echo [STEP 1/4] Pulling latest changes from %BRANCH%...
call :do_pull
if errorlevel 1 (
    pause
    goto :menu
)
echo.

echo [STEP 2/4] Installing missing runtime dependencies...
call :do_deps
echo.

echo [STEP 3/4] Setting permissions for shiny user...
call :do_permissions
echo.

echo [STEP 4/4] Restarting Shiny Server (restarts every app on the server)...
ssh -t %USER%@%SERVER% "sudo systemctl restart shiny-server"
if errorlevel 1 (
    echo [WARNING] Could not restart Shiny Server
) else (
    echo [OK] Shiny Server restarted
)

echo.
echo ============================================================
echo   Deployment Complete!
echo   Application URLs:
for %%A in (%APP_NAMES%) do echo     https://%SERVER%/%%A/
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
call :do_pull
echo.
pause
goto :menu

REM ============================================================================
REM Option 3: Install Missing Runtime Dependencies
REM ============================================================================
:update_deps
cls
echo.
echo ============================================================
echo   Install Missing Runtime Dependencies
echo ============================================================
echo.

call :check_ssh
if %ERRORLEVEL% neq 0 goto :menu

call :do_deps
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

echo Restarting Shiny Server (restarts every app on the server)...
ssh -t %USER%@%SERVER% "sudo systemctl restart shiny-server"
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
echo   Server Logs
echo ============================================================
echo.

call :check_ssh
if %ERRORLEVEL% neq 0 goto :menu

echo [Shiny Server log, last 30 lines]
ssh %USER%@%SERVER% "tail -30 /var/log/shiny-server.log 2>/dev/null || echo 'Could not read log file'"
echo.
echo [Latest CENOP app log, last 50 lines]
ssh %USER%@%SERVER% "ls -t /var/log/shiny-server/cenop-* /var/log/shiny-server/cenjas-* 2>/dev/null | head -1 | xargs -r tail -50"
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
ssh %USER%@%SERVER% "git -C %APP_DIR% status -sb | head -1 && git -C %APP_DIR% log -1 --oneline && git -C %APP_DIR% status -s --untracked-files=no"
echo.

echo [Shiny Server Status]
ssh %USER%@%SERVER% "systemctl status shiny-server --no-pager -l | head -15"
echo.

echo [Symlink Check]
ssh %USER%@%SERVER% "ls -la %SHINY_LINKS%"
echo.

echo [HTTP Check]
for %%A in (%APP_NAMES%) do (
    echo   %%A:
    ssh %USER%@%SERVER% "curl -s -o /dev/null -w %%{http_code} http://localhost:3838/%%A/; echo"
)
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
REM Helper: Fast-forward the server checkout (never resets)
REM ============================================================================
:do_pull
ssh %USER%@%SERVER% "git -C %APP_DIR% symbolic-ref --short HEAD | grep -qx %BRANCH%"
if errorlevel 1 (
    echo [ERROR] The server checkout is not on %BRANCH%. Refusing to deploy.
    echo         Check out %BRANCH% in %APP_DIR% on the server first.
    exit /b 1
)
ssh %USER%@%SERVER% "git -C %APP_DIR% pull --ff-only origin %BRANCH%"
if errorlevel 1 (
    echo [ERROR] Fast-forward pull failed: the checkout has diverged from origin
    echo         or has local changes that conflict. Resolve it on the server.
    exit /b 1
)
echo [OK] Repository updated
exit /b 0

REM ============================================================================
REM Helper: Install missing runtime dependencies into the shared env
REM ============================================================================
REM The env is owned by the shiny user, so pip runs as shiny; run as razinka it
REM would fall back to a user install in ~/.local that the server never sees.
REM Only the runtime part of requirements.txt (everything above the
REM "# Development" marker) is installed, and without -U, so packages that are
REM already satisfied are left alone.
:do_deps
ssh %USER%@%SERVER% "sed '/^# Development/,$d' %APP_DIR%/requirements.txt > %REQ_TMP% && chmod 644 %REQ_TMP%"
if errorlevel 1 (
    echo [ERROR] Could not prepare the runtime requirements list
    exit /b 1
)
ssh -t %USER%@%SERVER% "sudo -u shiny -H %PYTHON% -s -m pip install --no-cache-dir -r %REQ_TMP%"
if errorlevel 1 (
    echo [WARNING] Failed to install dependencies, continuing...
) else (
    echo [OK] Runtime dependencies satisfied
)
ssh %USER%@%SERVER% "rm -f %REQ_TMP%"
exit /b 0

REM ============================================================================
REM Helper: Give the shiny user read access (adds permissions, never removes)
REM ============================================================================
:do_permissions
echo Making the app directory readable by others (o+rX)...
ssh %USER%@%SERVER% "chmod -R o+rX %APP_DIR%"

echo Ensuring shiny user can traverse the home directories...
ssh %USER%@%SERVER% "chmod o+rx /home/razinka /home/razinka/cenop"

echo Verifying symlinks...
ssh %USER%@%SERVER% "ls -la %SHINY_LINKS% 2>/dev/null || echo '[WARNING] Symlink not found'"

echo [OK] Permissions configured for shiny user access
exit /b 0
