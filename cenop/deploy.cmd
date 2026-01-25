@echo off
REM ============================================================================
REM CENOP Deployment Script for laguna.ku.lt
REM ============================================================================
REM
REM Interactive deployment script for CENOP Shiny application
REM
REM Prerequisites:
REM   - SSH access to laguna.ku.lt as razinka
REM   - Micromamba installed with 'shiny' environment on server
REM   - Shiny Server for Python configured (runs as 'shiny' user)
REM
REM Server Notes:
REM   - /srv/shiny-server/cenop is a symlink to /home/razinka/cenop
REM   - Shiny Server runs as 'shiny' user, so files need o+rX permissions
REM   - Package is installed in editable mode to user site-packages
REM
REM ============================================================================

setlocal enabledelayedexpansion

REM ===== CONFIGURATION =====
set SERVER=laguna.ku.lt
set USER=razinka
set REMOTE_PATH=/home/razinka/cenop
set REMOTE_LINK=/srv/shiny-server/cenop
set LOCAL_PATH=%~dp0
set APP_NAME=cenop
set MAMBA_ENV=shiny
set MAMBA_ROOT=/opt/micromamba

:main_menu
cls
echo.
echo ************************************************************
echo *                                                          *
echo *   CENOP Deployment Script                                *
echo *   Target: %USER%@%SERVER%                       *
echo *                                                          *
echo ************************************************************
echo.
echo   Configuration:
echo   --------------
echo   Remote Path:     %REMOTE_PATH%
echo   Symlink:         %REMOTE_LINK%
echo   Local Path:      %LOCAL_PATH%
echo   Micromamba Env:  %MAMBA_ENV%
echo   Shiny User:      shiny (requires o+rX permissions)
echo.
echo ============================================================
echo.
echo   Select an option:
echo.
echo     1. Full Deployment (sync + permissions + install + restart)
echo     2. Sync files only (includes permission fix)
echo     3. Sync files and restart server
echo     4. Install dependencies only
echo     5. Restart server only
echo     6. Check server status
echo     7. View server logs
echo     8. Test SSH connection
echo     9. Fix permissions only
echo     0. Exit
echo.
echo ============================================================
echo.
set /p CHOICE="Enter your choice (0-9): "

if "%CHOICE%"=="1" goto :full_deploy
if "%CHOICE%"=="2" goto :sync_only
if "%CHOICE%"=="3" goto :sync_restart
if "%CHOICE%"=="4" goto :install_only
if "%CHOICE%"=="5" goto :restart_only
if "%CHOICE%"=="6" goto :status
if "%CHOICE%"=="7" goto :logs
if "%CHOICE%"=="8" goto :test_ssh
if "%CHOICE%"=="9" goto :fix_permissions_only
if "%CHOICE%"=="0" goto :exit_script

echo.
echo [ERROR] Invalid choice. Please enter a number between 0 and 9.
pause
goto :main_menu

REM ===== FULL DEPLOYMENT =====
:full_deploy
cls
echo.
echo ============================================================
echo   FULL DEPLOYMENT
echo ============================================================
echo.
echo This will:
echo   1. Sync all files to the server
echo   2. Fix permissions for Shiny Server (shiny user)
echo   3. Install/update Python dependencies
echo   4. Install CENOP package (editable mode)
echo   5. Verify import works
echo   6. Restart Shiny Server
echo.
set /p CONFIRM="Proceed with full deployment? (Y/N): "
if /i not "%CONFIRM%"=="Y" goto :main_menu

call :do_sync
if errorlevel 1 goto :error_pause
call :do_fix_permissions
if errorlevel 1 goto :error_pause
call :do_install
if errorlevel 1 goto :error_pause
call :do_verify_import
if errorlevel 1 goto :error_pause
call :do_restart
goto :success_pause

REM ===== SYNC ONLY =====
:sync_only
cls
echo.
echo ============================================================
echo   SYNC FILES ONLY
echo ============================================================
echo.
call :do_sync
if errorlevel 1 goto :error_pause
call :do_fix_permissions
goto :success_pause

REM ===== SYNC AND RESTART =====
:sync_restart
cls
echo.
echo ============================================================
echo   SYNC FILES AND RESTART
echo ============================================================
echo.
call :do_sync
if errorlevel 1 goto :error_pause
call :do_fix_permissions
if errorlevel 1 goto :error_pause
call :do_restart
goto :success_pause

REM ===== INSTALL ONLY =====
:install_only
cls
echo.
echo ============================================================
echo   INSTALL DEPENDENCIES ONLY
echo ============================================================
echo.
call :do_install
if errorlevel 1 goto :error_pause
call :do_verify_import
goto :success_pause

REM ===== RESTART ONLY =====
:restart_only
cls
echo.
echo ============================================================
echo   RESTART SERVER ONLY
echo ============================================================
echo.
call :do_restart
goto :success_pause

REM ===== FIX PERMISSIONS ONLY =====
:fix_permissions_only
cls
echo.
echo ============================================================
echo   FIX PERMISSIONS ONLY
echo ============================================================
echo.
call :do_fix_permissions
goto :success_pause

REM ===== STATUS =====
:status
cls
echo.
echo ============================================================
echo   SERVER STATUS
echo ============================================================
echo.

echo [1/7] Checking SSH connection...
echo       Command: ssh %USER%@%SERVER% "echo 'Connected'"
ssh %USER%@%SERVER% "echo 'Connected successfully'" 2>nul
if errorlevel 1 (
    echo [ERROR] Cannot connect to server
    goto :error_pause
)
echo.

echo [2/7] Shiny Server status:
echo ------------------------------------------------------------
ssh %USER%@%SERVER% "sudo systemctl status shiny-server --no-pager -l 2>/dev/null || echo 'Cannot get status (sudo required)'"
echo ------------------------------------------------------------
echo.

echo [3/7] Symlink verification:
ssh %USER%@%SERVER% "ls -la %REMOTE_LINK% 2>/dev/null || echo 'Symlink not found'"
echo.

echo [4/7] Python interpreter path (.python-version):
ssh %USER%@%SERVER% "cat %REMOTE_PATH%/.python-version 2>/dev/null || echo 'File not found'"
echo.

echo [5/7] Python version in micromamba '%MAMBA_ENV%' environment:
ssh %USER%@%SERVER% "micromamba run -n %MAMBA_ENV% python --version 2>/dev/null || echo 'Environment not found'"
echo.

echo [6/7] CENOP package info:
ssh %USER%@%SERVER% "micromamba run -n %MAMBA_ENV% pip show cenop 2>/dev/null | head -6 || echo 'CENOP not installed'"
echo.

echo [7/7] Directory permissions (src/cenop):
ssh %USER%@%SERVER% "ls -la %REMOTE_PATH%/src/cenop/ 2>/dev/null | head -5 || echo 'Directory not found'"
echo.

goto :success_pause

REM ===== LOGS =====
:logs
cls
echo.
echo ============================================================
echo   SERVER LOGS
echo ============================================================
echo.
echo   Select log type:
echo.
echo     1. Shiny Server main log (last 50 lines)
echo     2. Shiny Server main log (last 100 lines)
echo     3. List app-specific log files
echo     4. Back to main menu
echo.
set /p LOG_CHOICE="Enter your choice (1-4): "

if "%LOG_CHOICE%"=="1" (
    echo.
    echo [LOGS] Last 50 lines of Shiny Server log:
    echo ------------------------------------------------------------
    ssh %USER%@%SERVER% "sudo tail -50 /var/log/shiny-server.log 2>/dev/null || echo 'Log not found at /var/log/shiny-server.log'"
    echo ------------------------------------------------------------
    goto :success_pause
)
if "%LOG_CHOICE%"=="2" (
    echo.
    echo [LOGS] Last 100 lines of Shiny Server log:
    echo ------------------------------------------------------------
    ssh %USER%@%SERVER% "sudo tail -100 /var/log/shiny-server.log 2>/dev/null || echo 'Log not found at /var/log/shiny-server.log'"
    echo ------------------------------------------------------------
    goto :success_pause
)
if "%LOG_CHOICE%"=="3" (
    echo.
    echo [LOGS] App-specific log files:
    echo ------------------------------------------------------------
    ssh %USER%@%SERVER% "ls -la /var/log/shiny-server/ 2>/dev/null || echo 'No log directory found'"
    echo ------------------------------------------------------------
    goto :success_pause
)
if "%LOG_CHOICE%"=="4" goto :main_menu

echo [ERROR] Invalid choice
goto :logs

REM ===== TEST SSH =====
:test_ssh
cls
echo.
echo ============================================================
echo   TEST SSH CONNECTION
echo ============================================================
echo.
echo [TEST] Connecting to %USER%@%SERVER%...
echo        Command: ssh %USER%@%SERVER% "echo 'SSH OK'; hostname; uname -a"
echo.
echo ------------------------------------------------------------
ssh %USER%@%SERVER% "echo 'SSH connection successful!'; echo; echo 'Hostname:'; hostname; echo; echo 'System:'; uname -a; echo; echo 'Micromamba:'; micromamba --version 2>/dev/null || echo 'micromamba not found'; echo; echo 'Shiny Server user:'; grep -E 'run_as' /etc/shiny-server/shiny-server.conf 2>/dev/null || echo 'Cannot read config'"
echo ------------------------------------------------------------
if errorlevel 1 (
    echo.
    echo [ERROR] SSH connection failed!
    echo.
    echo Troubleshooting:
    echo   1. Check if SSH key is configured
    echo   2. Try: ssh %USER%@%SERVER%
    echo   3. Ensure the server is reachable
)
goto :success_pause

REM ===== DO SYNC FUNCTION =====
:do_sync
echo.
echo [SYNC] Starting file synchronization...
echo ============================================================
echo.

echo [SYNC] Step 1/2: Ensuring remote directory exists...
echo        Path: %REMOTE_PATH%

REM mkdir -p is idempotent - creates if not exists, does nothing if exists
ssh -o BatchMode=yes -o ConnectTimeout=10 %USER%@%SERVER% "mkdir -p %REMOTE_PATH%/src %REMOTE_PATH%/data %REMOTE_PATH%/static"
if errorlevel 1 (
    echo [ERROR] Failed to create remote directory
    exit /b 1
)
echo        [OK] Directory ready
echo.

echo [SYNC] Step 2/2: Syncing files...
echo        Source:      %LOCAL_PATH%
echo        Destination: %USER%@%SERVER%:%REMOTE_PATH%/
echo.

REM Try to find rsync in various locations
set "RSYNC_CMD="
set RSYNC_FOUND=0

where rsync >nul 2>&1
if !errorlevel!==0 (
    set "RSYNC_CMD=rsync"
    set RSYNC_FOUND=1
    echo        Found: rsync in PATH
)

if !RSYNC_FOUND!==0 (
    if exist "C:\Program Files\Git\usr\bin\rsync.exe" (
        set "RSYNC_CMD=C:\Program Files\Git\usr\bin\rsync.exe"
        set RSYNC_FOUND=1
        echo        Found: rsync in Git Bash
    )
)

if !RSYNC_FOUND!==0 (
    if exist "C:\msys64\usr\bin\rsync.exe" (
        set "RSYNC_CMD=C:\msys64\usr\bin\rsync.exe"
        set RSYNC_FOUND=1
        echo        Found: rsync in MSYS2
    )
)

if !RSYNC_FOUND!==1 (
    echo        Using rsync (syncs only changed files)...
    echo.
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
        --exclude "_ul*" ^
        --exclude "tools" ^
        --exclude "optimizations" ^
        -e "ssh -o BatchMode=yes -o ConnectTimeout=10" ^
        "%LOCAL_PATH%." %USER%@%SERVER%:%REMOTE_PATH%/
) else (
    echo        rsync not found, using tar-based sync...
    echo.

    REM Use tar to create and transfer archive (Windows 10+ has tar)
    set "TAR_FILE=%TEMP%\cenop_sync.tar.gz"

    echo        Creating archive...
    pushd "%LOCAL_PATH%"

    REM Create tar with all necessary files
    tar -czf "!TAR_FILE!" ^
        --exclude="__pycache__" ^
        --exclude="*.pyc" ^
        --exclude=".git" ^
        --exclude=".pytest_cache" ^
        --exclude="output" ^
        --exclude=".venv" ^
        --exclude="*.egg-info" ^
        --exclude=".claude" ^
        --exclude="_ul*" ^
        --exclude="tools" ^
        --exclude="optimizations" ^
        app.py requirements.txt pyproject.toml src data static 2>nul

    set TAR_ERROR=!errorlevel!
    popd

    if !TAR_ERROR!==0 (
        echo        Uploading archive...
        scp -o BatchMode=yes -o ConnectTimeout=10 "!TAR_FILE!" %USER%@%SERVER%:/tmp/cenop_sync.tar.gz
        if errorlevel 1 (
            echo [ERROR] Failed to upload archive
            del "!TAR_FILE!" 2>nul
            exit /b 1
        )

        echo        Extracting on server...
        ssh -o BatchMode=yes -o ConnectTimeout=10 %USER%@%SERVER% "cd %REMOTE_PATH% && tar -xzf /tmp/cenop_sync.tar.gz && rm /tmp/cenop_sync.tar.gz"
        if errorlevel 1 (
            echo [ERROR] Failed to extract archive
            del "!TAR_FILE!" 2>nul
            exit /b 1
        )

        del "!TAR_FILE!" 2>nul
    ) else (
        echo        tar failed, using scp fallback...
        echo.

        echo        Syncing app.py...
        scp -o BatchMode=yes "%LOCAL_PATH%app.py" %USER%@%SERVER%:%REMOTE_PATH%/

        echo        Syncing requirements.txt...
        scp -o BatchMode=yes "%LOCAL_PATH%requirements.txt" %USER%@%SERVER%:%REMOTE_PATH%/

        echo        Syncing pyproject.toml...
        scp -o BatchMode=yes "%LOCAL_PATH%pyproject.toml" %USER%@%SERVER%:%REMOTE_PATH%/

        echo        Syncing src directory...
        scp -o BatchMode=yes -r "%LOCAL_PATH%src" %USER%@%SERVER%:%REMOTE_PATH%/

        echo        Syncing data directory...
        scp -o BatchMode=yes -r "%LOCAL_PATH%data" %USER%@%SERVER%:%REMOTE_PATH%/

        echo        Syncing static directory...
        if exist "%LOCAL_PATH%static" (
            scp -o BatchMode=yes -r "%LOCAL_PATH%static" %USER%@%SERVER%:%REMOTE_PATH%/
        )
    )
)

if errorlevel 1 (
    echo.
    echo [ERROR] File sync failed
    exit /b 1
)

echo.
echo [SYNC] File synchronization complete!
echo ============================================================
exit /b 0

REM ===== DO FIX PERMISSIONS FUNCTION =====
:do_fix_permissions
echo.
echo [PERMISSIONS] Fixing file permissions for Shiny Server...
echo ============================================================
echo.
echo        Shiny Server runs as 'shiny' user
echo        Setting o+rX on src directory for read access...
echo.

ssh -o BatchMode=yes %USER%@%SERVER% "chmod -R o+rX %REMOTE_PATH%/src/ && chmod -R o+rX %REMOTE_PATH%/data/ 2>/dev/null; chmod o+rX %REMOTE_PATH%/ %REMOTE_PATH%/app.py %REMOTE_PATH%/pyproject.toml"
if errorlevel 1 (
    echo [WARNING] Some permission changes may have failed
) else (
    echo        [OK] Permissions updated
)

echo.
echo        Verifying permissions...
ssh %USER%@%SERVER% "ls -la %REMOTE_PATH%/src/cenop/ | head -3"
echo.

echo [PERMISSIONS] Complete!
echo ============================================================
exit /b 0

REM ===== DO INSTALL FUNCTION =====
:do_install
echo.
echo [INSTALL] Starting dependency installation...
echo ============================================================
echo.

echo [INSTALL] Step 1/5: Verifying micromamba environment '%MAMBA_ENV%'...
echo          Command: micromamba env list ^| grep %MAMBA_ENV%
ssh %USER%@%SERVER% "micromamba env list | grep -q %MAMBA_ENV%"
if errorlevel 1 (
    echo.
    echo [ERROR] Micromamba environment '%MAMBA_ENV%' not found!
    echo.
    echo Create it with:
    echo   ssh %USER%@%SERVER%
    echo   micromamba create -n %MAMBA_ENV% python=3.13
    echo   micromamba activate %MAMBA_ENV%
    echo   pip install shiny
    exit /b 1
)
echo          [OK] Environment exists
echo.

echo [INSTALL] Step 2/5: Upgrading pip...
echo          Command: micromamba run -n %MAMBA_ENV% pip install --upgrade pip
ssh %USER%@%SERVER% "micromamba run -n %MAMBA_ENV% pip install --upgrade pip -q"
if errorlevel 1 (
    echo          [WARNING] pip upgrade failed, continuing...
) else (
    echo          [OK] pip upgraded
)
echo.

echo [INSTALL] Step 3/5: Installing requirements.txt...
echo          Command: pip install -r requirements.txt
echo          (This may take several minutes)
echo.
ssh %USER%@%SERVER% "cd %REMOTE_PATH% && micromamba run -n %MAMBA_ENV% pip install -r requirements.txt -q"
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install dependencies
    exit /b 1
)
echo          [OK] Dependencies installed
echo.

echo [INSTALL] Step 4/5: Installing CENOP package (editable mode)...
echo          Command: pip install -e .
echo          Note: Installs to ~/.local (user site-packages)
ssh %USER%@%SERVER% "cd %REMOTE_PATH% && micromamba run -n %MAMBA_ENV% pip install -e . -q"
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install CENOP package
    exit /b 1
)
echo          [OK] CENOP installed
echo.

echo [INSTALL] Step 5/5: Creating .python-version file...
echo          Path: %MAMBA_ROOT%/envs/%MAMBA_ENV%/bin/python
ssh %USER%@%SERVER% "echo '%MAMBA_ROOT%/envs/%MAMBA_ENV%/bin/python' > %REMOTE_PATH%/.python-version"
echo          [OK] File created
echo.

echo [INSTALL] Dependency installation complete!
echo ============================================================
exit /b 0

REM ===== DO VERIFY IMPORT FUNCTION =====
:do_verify_import
echo.
echo [VERIFY] Testing Python imports...
echo ============================================================
echo.

echo          Testing: from cenop.ui.layout import app_ui
ssh %USER%@%SERVER% "micromamba run -n %MAMBA_ENV% python -c 'from cenop.ui.layout import app_ui; print(\"[OK] cenop.ui.layout imported successfully\")'"
if errorlevel 1 (
    echo.
    echo [ERROR] Import failed! Check:
    echo   1. Permissions: chmod -R o+rX ~/cenop/src/
    echo   2. Package installed: pip install -e .
    echo   3. Files exist: ls ~/cenop/src/cenop/ui/
    exit /b 1
)
echo.

echo          Testing: from cenop.server.main import server
ssh %USER%@%SERVER% "micromamba run -n %MAMBA_ENV% python -c 'from cenop.server.main import server; print(\"[OK] cenop.server.main imported successfully\")'"
if errorlevel 1 (
    echo.
    echo [WARNING] Server import failed, but continuing...
)
echo.

echo [VERIFY] Import verification complete!
echo ============================================================
exit /b 0

REM ===== DO RESTART FUNCTION =====
:do_restart
echo.
echo [RESTART] Restarting Shiny Server...
echo ============================================================
echo.
echo          Command: sudo systemctl restart shiny-server
echo.
ssh %USER%@%SERVER% "sudo systemctl restart shiny-server"
if errorlevel 1 (
    echo.
    echo [WARNING] Could not restart automatically (sudo may need password)
    echo.
    echo Please restart manually:
    echo   ssh %USER%@%SERVER%
    echo   sudo systemctl restart shiny-server
    echo.
) else (
    echo          [OK] Server restarted
    echo.
    echo          Waiting 3 seconds for server to start...
    timeout /t 3 /nobreak >nul
    echo.
    echo          Checking status...
    ssh %USER%@%SERVER% "sudo systemctl is-active shiny-server && echo 'Server is running!' || echo 'Server may not be running'"
)
echo.
echo [RESTART] Complete!
echo ============================================================
exit /b 0

REM ===== SUCCESS PAUSE =====
:success_pause
echo.
echo ************************************************************
echo   Application URL: http://%SERVER%:3838/%APP_NAME%/
echo ************************************************************
echo.
pause
goto :main_menu

REM ===== ERROR PAUSE =====
:error_pause
echo.
echo [!] An error occurred. Check the messages above.
echo.
pause
goto :main_menu

REM ===== EXIT =====
:exit_script
echo.
echo Goodbye!
echo.
endlocal
exit /b 0
