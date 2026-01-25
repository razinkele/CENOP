@echo on
REM ============================================================================
REM CENOP Deployment DEBUG Script
REM ============================================================================
REM
REM Debug wrapper for deploy.cmd - traces execution, logs output, pauses at steps
REM
REM Usage:
REM   deploy_debug.cmd              - Run with debug output
REM   deploy_debug.cmd /log         - Run with logging to file
REM   deploy_debug.cmd /step        - Step through each command (pause after each)
REM   deploy_debug.cmd /log /step   - Both logging and stepping
REM
REM ============================================================================

setlocal enabledelayedexpansion

REM ===== DEBUG CONFIGURATION =====
set DEBUG=1
set LOG_TO_FILE=0
set STEP_MODE=0
set SSH_VERBOSE=0
set LOG_FILE=%~dp0deploy_debug_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log
set LOG_FILE=%LOG_FILE: =0%

REM Parse arguments
:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="/log" set LOG_TO_FILE=1
if /i "%~1"=="/step" set STEP_MODE=1
if /i "%~1"=="/verbose" set SSH_VERBOSE=1
if /i "%~1"=="/help" goto :show_help
shift
goto :parse_args
:args_done

REM ===== SERVER CONFIGURATION =====
set SERVER=laguna.ku.lt
set USER=razinka
set REMOTE_PATH=/srv/shiny-server/cenop
set LOCAL_PATH=%~dp0
set APP_NAME=cenop
set MAMBA_ENV=shiny
set MAMBA_ROOT=/opt/micromamba

REM ===== SSH OPTIONS =====
REM -o BatchMode=yes              : Disable password prompts (fail if key auth fails)
REM -o ConnectTimeout=10          : 10 second connection timeout
REM -o ServerAliveInterval=5      : Send keepalive every 5 seconds
REM -o ServerAliveCountMax=3      : Disconnect after 3 missed keepalives
REM -o StrictHostKeyChecking=accept-new : Auto-accept new host keys
REM -v                            : Verbose mode (use /verbose flag)
set SSH_BASE_OPTS=-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=accept-new
if %SSH_VERBOSE%==1 (
    set SSH_OPTS=-v %SSH_BASE_OPTS%
) else (
    set SSH_OPTS=%SSH_BASE_OPTS%
)

REM Initialize log file
if %LOG_TO_FILE%==1 (
    echo ============================================================ > "%LOG_FILE%"
    echo CENOP Deployment Debug Log >> "%LOG_FILE%"
    echo Started: %date% %time% >> "%LOG_FILE%"
    echo ============================================================ >> "%LOG_FILE%"
    echo. >> "%LOG_FILE%"
)

call :log "============================================================"
call :log "CENOP DEPLOYMENT - DEBUG MODE"
call :log "============================================================"
call :log "Debug Options:"
call :log "  DEBUG=%DEBUG%"
call :log "  LOG_TO_FILE=%LOG_TO_FILE%"
call :log "  STEP_MODE=%STEP_MODE%"
call :log "  SSH_VERBOSE=%SSH_VERBOSE%"
if %LOG_TO_FILE%==1 call :log "  LOG_FILE=%LOG_FILE%"
call :log ""
call :log "Configuration:"
call :log "  SERVER=%SERVER%"
call :log "  USER=%USER%"
call :log "  REMOTE_PATH=%REMOTE_PATH%"
call :log "  LOCAL_PATH=%LOCAL_PATH%"
call :log "  MAMBA_ENV=%MAMBA_ENV%"
call :log "  SSH_OPTS=%SSH_OPTS%"
call :log ""

call :step_pause "Configuration loaded. Continue?"

REM ===== MAIN MENU =====
:main_menu
cls
call :log ""
call :log "[MENU] Displaying main menu at %time%"

echo.
echo ************************************************************
echo *   CENOP Deployment Script - DEBUG MODE                   *
echo ************************************************************
echo.
echo   1. Full Deployment (sync + install + restart)
echo   2. Sync files only
echo   3. Sync files and restart server
echo   4. Install dependencies only
echo   5. Restart server only
echo   6. Check server status
echo   7. View server logs
echo   8. Test SSH connection
echo   9. Exit
echo.
if %LOG_TO_FILE%==1 echo   [LOGGING TO: %LOG_FILE%]
if %STEP_MODE%==1 echo   [STEP MODE: ON]
if %SSH_VERBOSE%==1 echo   [SSH VERBOSE: ON]
echo.
set /p CHOICE="Enter your choice (1-9): "

call :log "[MENU] User selected option: %CHOICE%"

if "%CHOICE%"=="1" goto :full_deploy
if "%CHOICE%"=="2" goto :sync_only
if "%CHOICE%"=="3" goto :sync_restart
if "%CHOICE%"=="4" goto :install_only
if "%CHOICE%"=="5" goto :restart_only
if "%CHOICE%"=="6" goto :status
if "%CHOICE%"=="7" goto :logs
if "%CHOICE%"=="8" goto :test_ssh
if "%CHOICE%"=="9" goto :exit_script

call :log "[ERROR] Invalid choice: %CHOICE%"
echo [ERROR] Invalid choice
pause
goto :main_menu

REM ===== FULL DEPLOYMENT =====
:full_deploy
call :log ""
call :log "============================================================"
call :log "[FULL_DEPLOY] Starting full deployment at %time%"
call :log "============================================================"

echo.
echo ============================================================
echo   FULL DEPLOYMENT - DEBUG MODE
echo ============================================================
echo.
set /p CONFIRM="Proceed with full deployment? (Y/N): "
call :log "[FULL_DEPLOY] User confirmation: %CONFIRM%"

if /i not "%CONFIRM%"=="Y" (
    call :log "[FULL_DEPLOY] User cancelled"
    goto :main_menu
)

REM --- PRE-FLIGHT SSH CHECK ---
call :log ""
call :log "[FULL_DEPLOY] === PRE-FLIGHT: SSH CONNECTIVITY ==="
echo.
echo [PRE-FLIGHT] Testing SSH connectivity...
echo             Target: %USER%@%SERVER%
echo             Options: %SSH_OPTS%
echo.

ssh %SSH_OPTS% %USER%@%SERVER% "echo 'SSH_OK'"
set SSH_TEST=%ERRORLEVEL%
call :log "[PRE-FLIGHT] SSH test result: ERRORLEVEL=%SSH_TEST%"

if %SSH_TEST% NEQ 0 (
    call :log "[PRE-FLIGHT] ERROR: SSH connection failed!"
    echo.
    echo ************************************************************
    echo   [ERROR] SSH CONNECTION FAILED!
    echo ************************************************************
    echo.
    echo   Possible causes:
    echo     1. No SSH key configured for %USER%@%SERVER%
    echo     2. SSH key not loaded in ssh-agent
    echo     3. Server is unreachable
    echo     4. Firewall blocking port 22
    echo.
    echo   Try manually:
    echo     ssh %USER%@%SERVER%
    echo.
    echo   If prompted for password, set up SSH key:
    echo     ssh-keygen -t ed25519
    echo     ssh-copy-id %USER%@%SERVER%
    echo.
    goto :error_pause
)
echo             [OK] SSH connection successful
echo.
call :step_pause "SSH connectivity verified. Continue with deployment?"

REM --- SYNC PHASE ---
call :log ""
call :log "[FULL_DEPLOY] === PHASE 1: SYNC ==="
call :step_pause "About to start SYNC phase"

set SYNC_RESULT=0
call :do_sync_debug
set SYNC_RESULT=%ERRORLEVEL%
call :log "[FULL_DEPLOY] SYNC phase completed with ERRORLEVEL=%SYNC_RESULT%"

if %SYNC_RESULT% NEQ 0 (
    call :log "[FULL_DEPLOY] SYNC failed, aborting"
    echo.
    echo [DEBUG] SYNC failed with error code %SYNC_RESULT%
    goto :error_pause
)
call :step_pause "SYNC completed successfully. Continue to INSTALL?"

REM --- INSTALL PHASE ---
call :log ""
call :log "[FULL_DEPLOY] === PHASE 2: INSTALL ==="
call :step_pause "About to start INSTALL phase"

set INSTALL_RESULT=0
call :do_install_debug
set INSTALL_RESULT=%ERRORLEVEL%
call :log "[FULL_DEPLOY] INSTALL phase completed with ERRORLEVEL=%INSTALL_RESULT%"

if %INSTALL_RESULT% NEQ 0 (
    call :log "[FULL_DEPLOY] INSTALL failed, aborting"
    echo.
    echo [DEBUG] INSTALL failed with error code %INSTALL_RESULT%
    goto :error_pause
)
call :step_pause "INSTALL completed successfully. Continue to RESTART?"

REM --- RESTART PHASE ---
call :log ""
call :log "[FULL_DEPLOY] === PHASE 3: RESTART ==="
call :step_pause "About to start RESTART phase"

set RESTART_RESULT=0
call :do_restart_debug
set RESTART_RESULT=%ERRORLEVEL%
call :log "[FULL_DEPLOY] RESTART phase completed with ERRORLEVEL=%RESTART_RESULT%"

call :log "[FULL_DEPLOY] Full deployment completed!"
goto :success_pause

REM ===== SYNC ONLY =====
:sync_only
call :log "[SYNC_ONLY] Starting at %time%"
call :do_sync_debug
goto :success_pause

REM ===== SYNC AND RESTART =====
:sync_restart
call :log "[SYNC_RESTART] Starting at %time%"
call :do_sync_debug
if %ERRORLEVEL% NEQ 0 goto :error_pause
call :do_restart_debug
goto :success_pause

REM ===== INSTALL ONLY =====
:install_only
call :log "[INSTALL_ONLY] Starting at %time%"
call :do_install_debug
goto :success_pause

REM ===== RESTART ONLY =====
:restart_only
call :log "[RESTART_ONLY] Starting at %time%"
call :do_restart_debug
goto :success_pause

REM ===== STATUS =====
:status
call :log "[STATUS] Checking server status at %time%"
echo.
echo ============================================================
echo   SERVER STATUS - DEBUG MODE
echo ============================================================
echo.

call :log "[STATUS] Testing SSH connection..."
call :ssh_cmd "echo 'Connected'" "SSH connection test"
if %ERRORLEVEL% NEQ 0 goto :error_pause

call :log "[STATUS] Checking shiny-server status..."
call :ssh_cmd "sudo systemctl status shiny-server --no-pager -l 2>&1 || echo '[SSH returned error]'" "Shiny Server status"

call :log "[STATUS] Checking .python-version..."
call :ssh_cmd "cat %REMOTE_PATH%/.python-version 2>/dev/null || echo 'File not found'" "Python version file"

call :log "[STATUS] Checking micromamba Python..."
call :ssh_cmd "micromamba run -n %MAMBA_ENV% python --version 2>/dev/null || echo 'Environment not found'" "Micromamba Python"

call :log "[STATUS] Checking CENOP package..."
call :ssh_cmd "micromamba run -n %MAMBA_ENV% pip show cenop 2>/dev/null | head -6 || echo 'CENOP not installed'" "CENOP package info"

goto :success_pause

REM ===== LOGS =====
:logs
call :log "[LOGS] Viewing logs at %time%"
echo.
echo   1. Last 50 lines of Shiny Server log
echo   2. Last 100 lines of Shiny Server log
echo   3. List app-specific log files
echo   4. Back to main menu
echo.
set /p LOG_CHOICE="Choice: "
call :log "[LOGS] User selected: %LOG_CHOICE%"

if "%LOG_CHOICE%"=="1" call :ssh_cmd "sudo tail -50 /var/log/shiny-server.log 2>/dev/null || echo 'Log not found'" "Last 50 log lines"
if "%LOG_CHOICE%"=="2" call :ssh_cmd "sudo tail -100 /var/log/shiny-server.log 2>/dev/null || echo 'Log not found'" "Last 100 log lines"
if "%LOG_CHOICE%"=="3" call :ssh_cmd "ls -la /var/log/shiny-server/ 2>/dev/null || echo 'No log directory'" "Log file listing"
if "%LOG_CHOICE%"=="4" goto :main_menu
goto :success_pause

REM ===== TEST SSH =====
:test_ssh
call :log "[TEST_SSH] Testing SSH connection at %time%"
echo.
echo ============================================================
echo   SSH CONNECTION TEST - DEBUG MODE
echo ============================================================
echo.

call :ssh_cmd "echo 'SSH OK'; hostname; uname -a; micromamba --version 2>/dev/null || echo 'micromamba not found'" "SSH connection test"

goto :success_pause

REM ============================================================================
REM DEBUG SYNC FUNCTION
REM ============================================================================
:do_sync_debug
call :log ""
call :log "[SYNC] ================================================"
call :log "[SYNC] Starting file synchronization"
call :log "[SYNC] ================================================"

echo.
echo [SYNC] Starting file synchronization...
echo ============================================================
echo.

REM Step 1: Create directory (mkdir -p is idempotent - safe to run even if exists)
call :log "[SYNC] Step 1/2: Ensuring remote directory exists"
call :step_pause "About to ensure remote directory: %REMOTE_PATH%"

echo [SYNC] Step 1/2: Ensuring remote directory exists...
echo        Path: %REMOTE_PATH%
echo.

REM mkdir -p is safe - creates if not exists, does nothing if exists
echo        Running mkdir -p (idempotent - safe if dir exists)...
call :ssh_cmd "mkdir -p %REMOTE_PATH% && echo MKDIR_OK" "Create directory"
set DIR_RESULT=!ERRORLEVEL!
call :log "[SYNC] mkdir result: ERRORLEVEL=!DIR_RESULT!"

if !DIR_RESULT! NEQ 0 (
    call :log "[SYNC] ERROR: Failed to create remote directory"
    echo [ERROR] Failed to create remote directory
    exit /b 1
)
echo        [OK] Directory ready
echo.

REM Step 2: Transfer changed files only
call :log "[SYNC] Step 2/2: Transferring changed files"
call :step_pause "About to transfer changed files"

echo [SYNC] Step 2/2: Syncing changed files only...
echo        Source: %LOCAL_PATH%
echo        Dest:   %USER%@%SERVER%:%REMOTE_PATH%/
echo.

REM Try to find rsync in various locations
set "RSYNC_CMD="
set RSYNC_FOUND=0

REM Check 1: rsync in PATH
where rsync >nul 2>&1
if !ERRORLEVEL!==0 (
    set "RSYNC_CMD=rsync"
    set RSYNC_FOUND=1
    call :log "[SYNC] Found rsync in PATH"
    echo        Found: rsync in PATH
)

REM Check 2: Git Bash rsync
if !RSYNC_FOUND!==0 (
    if exist "C:\Program Files\Git\usr\bin\rsync.exe" (
        set "RSYNC_CMD=C:\Program Files\Git\usr\bin\rsync.exe"
        set RSYNC_FOUND=1
        call :log "[SYNC] Found rsync in Git Bash"
        echo        Found: rsync in Git Bash
    )
)

REM Check 3: MSYS2 rsync
if !RSYNC_FOUND!==0 (
    if exist "C:\msys64\usr\bin\rsync.exe" (
        set "RSYNC_CMD=C:\msys64\usr\bin\rsync.exe"
        set RSYNC_FOUND=1
        call :log "[SYNC] Found rsync in MSYS2"
        echo        Found: rsync in MSYS2
    )
)

if !RSYNC_FOUND!==1 (
    echo.
    echo        Using rsync (transfers only changed files)...
    call :log "[SYNC] rsync command starting..."

    "!RSYNC_CMD!" -avz --progress ^
        --exclude "__pycache__" ^
        --exclude "*.pyc" ^
        --exclude ".git" ^
        --exclude ".pytest_cache" ^
        --exclude "output" ^
        --exclude ".venv" ^
        --exclude "*.egg-info" ^
        --exclude ".claude" ^
        --exclude "*.log" ^
        --exclude "deploy_debug_*.log" ^
        -e "ssh !SSH_OPTS!" ^
        "%LOCAL_PATH%." %USER%@%SERVER%:%REMOTE_PATH%/
    set RSYNC_ERR=!ERRORLEVEL!
    call :log "[SYNC] rsync completed with ERRORLEVEL=!RSYNC_ERR!"

    if !RSYNC_ERR! NEQ 0 (
        call :log "[SYNC] ERROR: rsync failed"
        echo [ERROR] rsync failed with error !RSYNC_ERR!
        exit /b 1
    )
) else (
    call :log "[SYNC] No rsync found - using tar delta sync"
    echo        rsync not found, using tar-based delta sync...
    echo        [TIP] Install Git for Windows for native rsync support
    echo.

    REM Create local tarball of source files (excluding unnecessary files)
    echo        Creating archive of local files...
    set "TAR_FILE=%TEMP%\cenop_sync.tar.gz"

    REM Use tar to create archive (Windows 10+ has tar built-in)
    pushd "%LOCAL_PATH%"
    tar -czf "!TAR_FILE!" ^
        --exclude="__pycache__" ^
        --exclude="*.pyc" ^
        --exclude=".git" ^
        --exclude=".pytest_cache" ^
        --exclude="output" ^
        --exclude=".venv" ^
        --exclude="*.egg-info" ^
        --exclude=".claude" ^
        --exclude="*.log" ^
        app.py requirements.txt pyproject.toml src data 2>nul
    set TAR_ERR=!ERRORLEVEL!
    popd

    if !TAR_ERR! NEQ 0 (
        call :log "[SYNC] tar not available, falling back to scp"
        echo        tar failed, using scp fallback...
        goto :scp_fallback
    )

    echo        Uploading archive to server...
    scp !SSH_OPTS! "!TAR_FILE!" %USER%@%SERVER%:/tmp/cenop_sync.tar.gz
    if !ERRORLEVEL! NEQ 0 (
        call :log "[SYNC] ERROR: Failed to upload archive"
        echo [ERROR] Failed to upload archive
        del "!TAR_FILE!" 2>nul
        exit /b 1
    )

    echo        Extracting on server (overwrites only changed files)...
    call :ssh_cmd "cd %REMOTE_PATH% && tar -xzf /tmp/cenop_sync.tar.gz && rm /tmp/cenop_sync.tar.gz && echo EXTRACT_OK" "Extract archive"
    if !ERRORLEVEL! NEQ 0 (
        call :log "[SYNC] ERROR: Failed to extract archive"
        echo [ERROR] Failed to extract archive on server
        del "!TAR_FILE!" 2>nul
        exit /b 1
    )

    del "!TAR_FILE!" 2>nul
    echo        [OK] Delta sync complete
    goto :sync_done
)

:scp_fallback
call :log "[SYNC] Using scp fallback (copies all files)"
echo        Using scp (copies all files - slower)...
echo.

scp !SSH_OPTS! "%LOCAL_PATH%app.py" %USER%@%SERVER%:%REMOTE_PATH%/
scp !SSH_OPTS! "%LOCAL_PATH%requirements.txt" %USER%@%SERVER%:%REMOTE_PATH%/
scp !SSH_OPTS! "%LOCAL_PATH%pyproject.toml" %USER%@%SERVER%:%REMOTE_PATH%/
scp !SSH_OPTS! -r "%LOCAL_PATH%src" %USER%@%SERVER%:%REMOTE_PATH%/
scp !SSH_OPTS! -r "%LOCAL_PATH%data" %USER%@%SERVER%:%REMOTE_PATH%/
if exist "%LOCAL_PATH%static" scp !SSH_OPTS! -r "%LOCAL_PATH%static" %USER%@%SERVER%:%REMOTE_PATH%/

:sync_done

echo.
echo [SYNC] File synchronization complete!
call :log "[SYNC] Synchronization completed successfully"
echo ============================================================
exit /b 0

REM ============================================================================
REM DEBUG INSTALL FUNCTION
REM ============================================================================
:do_install_debug
call :log ""
call :log "[INSTALL] ================================================"
call :log "[INSTALL] Starting dependency installation"
call :log "[INSTALL] ================================================"

echo.
echo [INSTALL] Starting dependency installation...
echo ============================================================
echo.

REM Step 1: Verify environment
call :log "[INSTALL] Step 1/5: Verifying micromamba environment"
call :step_pause "About to verify micromamba environment: %MAMBA_ENV%"

echo [INSTALL] Step 1/5: Verifying micromamba environment...
call :ssh_cmd "micromamba env list | grep -q %MAMBA_ENV% && echo 'Found' || exit 1" "Check micromamba env"
if %ERRORLEVEL% NEQ 0 (
    call :log "[INSTALL] ERROR: Environment '%MAMBA_ENV%' not found"
    echo [ERROR] Environment '%MAMBA_ENV%' not found!
    exit /b 1
)
echo          [OK] Environment exists
echo.

REM Step 2: Upgrade pip
call :log "[INSTALL] Step 2/5: Upgrading pip"
call :step_pause "About to upgrade pip"

echo [INSTALL] Step 2/5: Upgrading pip...
call :ssh_cmd "micromamba run -n %MAMBA_ENV% pip install --upgrade pip 2>&1" "Upgrade pip"
call :log "[INSTALL] pip upgrade result: ERRORLEVEL=%ERRORLEVEL%"
echo          [OK] pip upgraded
echo.

REM Step 3: Install requirements
call :log "[INSTALL] Step 3/5: Installing requirements.txt"
call :step_pause "About to install requirements.txt (this may take a while)"

echo [INSTALL] Step 3/5: Installing requirements.txt...
echo          (This may take several minutes)
echo.
call :ssh_cmd "cd %REMOTE_PATH% && micromamba run -n %MAMBA_ENV% pip install -r requirements.txt 2>&1" "Install requirements"
set REQ_ERR=%ERRORLEVEL%
call :log "[INSTALL] requirements.txt install result: ERRORLEVEL=%REQ_ERR%"

if %REQ_ERR% NEQ 0 (
    call :log "[INSTALL] ERROR: Failed to install requirements"
    echo [ERROR] Failed to install requirements
    exit /b 1
)
echo          [OK] Dependencies installed
echo.

REM Step 4: Install CENOP package
call :log "[INSTALL] Step 4/5: Installing CENOP package"
call :step_pause "About to install CENOP package"

echo [INSTALL] Step 4/5: Installing CENOP package...
call :ssh_cmd "cd %REMOTE_PATH% && micromamba run -n %MAMBA_ENV% pip install -e . 2>&1" "Install CENOP"
set CENOP_ERR=%ERRORLEVEL%
call :log "[INSTALL] CENOP install result: ERRORLEVEL=%CENOP_ERR%"

if %CENOP_ERR% NEQ 0 (
    call :log "[INSTALL] ERROR: Failed to install CENOP"
    echo [ERROR] Failed to install CENOP package
    exit /b 1
)
echo          [OK] CENOP installed
echo.

REM Step 5: Create .python-version
call :log "[INSTALL] Step 5/5: Creating .python-version file"
call :step_pause "About to create .python-version file"

echo [INSTALL] Step 5/5: Creating .python-version file...
call :ssh_cmd "echo '%MAMBA_ROOT%/envs/%MAMBA_ENV%/bin/python' > %REMOTE_PATH%/.python-version && cat %REMOTE_PATH%/.python-version" "Create .python-version"
call :log "[INSTALL] .python-version creation result: ERRORLEVEL=%ERRORLEVEL%"
echo          [OK] File created
echo.

echo [INSTALL] Dependency installation complete!
call :log "[INSTALL] Installation completed successfully"
echo ============================================================
exit /b 0

REM ============================================================================
REM DEBUG RESTART FUNCTION
REM ============================================================================
:do_restart_debug
call :log ""
call :log "[RESTART] ================================================"
call :log "[RESTART] Restarting Shiny Server"
call :log "[RESTART] ================================================"

echo.
echo [RESTART] Restarting Shiny Server...
echo ============================================================
echo.

call :step_pause "About to restart shiny-server"

call :ssh_cmd "sudo systemctl restart shiny-server 2>&1" "Restart shiny-server"
set RESTART_ERR=%ERRORLEVEL%
call :log "[RESTART] Restart command result: ERRORLEVEL=%RESTART_ERR%"

if %RESTART_ERR% NEQ 0 (
    call :log "[RESTART] WARNING: Restart may have failed"
    echo [WARNING] Restart command returned error
    echo          May need to restart manually
) else (
    echo          [OK] Restart command sent
)
echo.

echo          Waiting 3 seconds...
timeout /t 3 /nobreak >nul

call :log "[RESTART] Checking server status..."
echo          Checking status...
call :ssh_cmd "sudo systemctl is-active shiny-server && echo 'Server is RUNNING' || echo 'Server is NOT running'" "Check server status"
call :log "[RESTART] Status check result: ERRORLEVEL=%ERRORLEVEL%"

echo.
echo [RESTART] Complete!
call :log "[RESTART] Restart phase completed"
echo ============================================================
exit /b 0

REM ============================================================================
REM HELPER: SSH COMMAND WITH LOGGING
REM ============================================================================
:ssh_cmd
set "SSH_COMMAND=%~1"
set "SSH_DESC=%~2"

echo.
echo [%time%] [SSH] %SSH_DESC%
echo          Command: %SSH_COMMAND%
echo ------------------------------------------------------------

REM Small delay to avoid SSH connection rate limiting
timeout /t 1 /nobreak >nul 2>&1

REM Execute SSH and capture result
echo          Executing SSH...
ssh %SSH_OPTS% %USER%@%SERVER% "%SSH_COMMAND%"
set SSH_RESULT=%ERRORLEVEL%

echo ------------------------------------------------------------
echo [%time%] [SSH] Completed with ERRORLEVEL=%SSH_RESULT%

if %SSH_RESULT% NEQ 0 (
    echo [SSH] Warning: Command returned error code %SSH_RESULT%
)
echo.
exit /b %SSH_RESULT%

REM ============================================================================
REM HELPER: LOGGING
REM ============================================================================
:log
set LOG_MSG=%~1
echo [%time%] %LOG_MSG%
if %LOG_TO_FILE%==1 echo [%time%] %LOG_MSG% >> "%LOG_FILE%"
exit /b 0

REM ============================================================================
REM HELPER: STEP PAUSE (only in step mode)
REM ============================================================================
:step_pause
if %STEP_MODE%==0 exit /b 0
set STEP_MSG=%~1
echo.
echo [STEP] %STEP_MSG%
pause
exit /b 0

REM ============================================================================
REM SUCCESS/ERROR HANDLERS
REM ============================================================================
:success_pause
call :log ""
call :log "[SUCCESS] Operation completed successfully at %time%"
echo.
echo ************************************************************
echo   Operation completed successfully!
echo   Application URL: https://%SERVER%/%APP_NAME%/
echo ************************************************************
if %LOG_TO_FILE%==1 (
    echo.
    echo   Debug log saved to: %LOG_FILE%
)
echo.
pause
goto :main_menu

:error_pause
call :log ""
call :log "[ERROR] Operation failed at %time%"
echo.
echo ************************************************************
echo   [ERROR] An error occurred!
echo ************************************************************
if %LOG_TO_FILE%==1 (
    echo.
    echo   Debug log saved to: %LOG_FILE%
    echo   Review the log for details.
)
echo.
pause
goto :main_menu

REM ============================================================================
REM HELP
REM ============================================================================
:show_help
echo.
echo CENOP Deployment Debug Script
echo ==============================
echo.
echo Usage: deploy_debug.cmd [options]
echo.
echo Options:
echo   /log      Log all output to a timestamped file
echo   /step     Step mode - pause after each operation
echo   /verbose  Enable verbose SSH output (shows connection details)
echo   /help     Show this help message
echo.
echo Examples:
echo   deploy_debug.cmd                  Run with debug output
echo   deploy_debug.cmd /log             Run with file logging
echo   deploy_debug.cmd /step            Step through operations
echo   deploy_debug.cmd /verbose         Verbose SSH for connection issues
echo   deploy_debug.cmd /log /step       Both logging and stepping
echo   deploy_debug.cmd /verbose /log    SSH debug with logging
echo.
echo SSH Options Applied:
echo   -o BatchMode=yes              Fail immediately if password needed
echo   -o ConnectTimeout=10          10 second timeout
echo   -o StrictHostKeyChecking=accept-new   Auto-accept new hosts
echo.
exit /b 0

REM ============================================================================
REM EXIT
REM ============================================================================
:exit_script
call :log "[EXIT] Script terminated by user at %time%"
if %LOG_TO_FILE%==1 (
    echo.
    echo Debug log saved to: %LOG_FILE%
)
echo.
echo Goodbye!
echo.
endlocal
exit /b 0
