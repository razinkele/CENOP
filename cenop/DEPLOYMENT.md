# CENOP Deployment Guide

## Deployment to laguna.ku.lt Shiny Server

This guide explains how to deploy CENOP to the Shiny Server on `laguna.ku.lt`.

---

## Prerequisites

### On Your Local Machine (Windows)
- SSH client (Windows 10+ has OpenSSH built-in)
- SSH key configured for passwordless login to laguna.ku.lt
- Git Bash or WSL (optional, for rsync support)

### On the Server (laguna.ku.lt)
- Micromamba installed with `shiny` environment
- Shiny Server for Python installed and configured
- User `razinka` with:
  - Write access to `/srv/shiny-server/cenop`
  - Sudo access to restart shiny-server (optional)

### Micromamba Environment Setup (One-time)
If the `shiny` environment doesn't exist, create it on the server:
```bash
micromamba create -n shiny python=3.10
micromamba activate shiny
pip install shiny shinyswatch shinywidgets
```

---

## Quick Deployment

### Option 1: Using the Deployment Script

```cmd
cd c:\Users\DELL\OneDrive - ku.lt\HORIZON_EUROPE\AI4WIND\CENOP\cenop
deploy.cmd
```

This will perform a full deployment (sync + install + restart).

### Deployment Script Options

```cmd
deploy.cmd [options]

Options:
  /sync      Sync files to server (rsync or scp)
  /install   Install Python dependencies and CENOP package
  /restart   Restart Shiny Server
  /status    Show server status
  /logs      Show recent server logs
  /full      Full deployment (sync + install + restart)
  /help      Show help message

Examples:
  deploy.cmd              Full deployment (default)
  deploy.cmd /sync        Only sync files
  deploy.cmd /restart     Only restart server
  deploy.cmd /sync /restart   Sync files and restart
  deploy.cmd /status      Check server status
  deploy.cmd /logs        View recent logs
```

### Full Deployment Steps

When running `deploy.cmd` or `deploy.cmd /full`:
1. Create the remote directory
2. Sync all application files
3. Verify micromamba `shiny` environment exists
4. Install Python dependencies in micromamba environment
5. Install CENOP package
6. Create `.python-version` file for Shiny Server
7. Restart Shiny Server

### Option 2: Manual Deployment

#### Step 1: Connect to Server
```cmd
ssh razinka@laguna.ku.lt
```

#### Step 2: Create Application Directory
```bash
sudo mkdir -p /srv/shiny-server/cenop
sudo chown razinka:razinka /srv/shiny-server/cenop
```

#### Step 3: Copy Files from Windows
Open a new Command Prompt window:
```cmd
scp -r "c:\Users\DELL\OneDrive - ku.lt\HORIZON_EUROPE\AI4WIND\CENOP\cenop\*" razinka@laguna.ku.lt:/srv/shiny-server/cenop/
```

Or using rsync (via Git Bash):
```bash
rsync -avz --delete \
    --exclude "__pycache__" \
    --exclude "*.pyc" \
    --exclude ".git" \
    --exclude ".pytest_cache" \
    --exclude "output" \
    --exclude ".venv" \
    "/c/Users/DELL/OneDrive - ku.lt/HORIZON_EUROPE/AI4WIND/CENOP/cenop/" \
    razinka@laguna.ku.lt:/srv/shiny-server/cenop/
```

#### Step 4: Install Dependencies in Micromamba Environment (on server)
```bash
cd /srv/shiny-server/cenop
micromamba run -n shiny pip install --upgrade pip
micromamba run -n shiny pip install -r requirements.txt
micromamba run -n shiny pip install -e .
```

#### Step 5: Create .python-version File
```bash
echo '/opt/micromamba/envs/shiny/bin/python' > /srv/shiny-server/cenop/.python-version
```

#### Step 6: Restart Shiny Server
```bash
sudo systemctl restart shiny-server
```

---

## Shiny Server Configuration

### Application Configuration

Create or update `/etc/shiny-server/shiny-server.conf`:

```nginx
# Run as the shiny user
run_as shiny;

# Define the root directory for applications
server {
    listen 3838;

    # CENOP Application
    location /cenop {
        site_dir /srv/shiny-server/cenop;
        python /opt/micromamba/envs/shiny/bin/python;
        log_dir /var/log/shiny-server/cenop;
        directory_index on;
    }

    # Other applications can be added here
}
```

### Per-Application Python Configuration (Recommended)

Create `.python-version` in the app directory to specify the micromamba Python interpreter:

```bash
echo "/opt/micromamba/envs/shiny/bin/python" > /srv/shiny-server/cenop/.python-version
```

This file is automatically created by `deploy.cmd` during deployment.

---

## Application Entry Point

Shiny Server for Python expects an `app.py` file in the application root. 
The current `app.py` should work directly with Shiny Server.

### Verify Entry Point
```bash
cat /srv/shiny-server/cenop/app.py
```

The file should contain:
```python
from shiny import App
from ui.layout import create_layout
from server.main import create_server

# Create the Shiny app
app = App(create_layout(), create_server())
```

---

## Environment Variables (Optional)

Create `/srv/shiny-server/cenop/.env` for any custom configuration:

```bash
# CENOP Configuration
CENOP_LOG_LEVEL=INFO
CENOP_OUTPUT_DIR=/srv/shiny-server/cenop/output
CENOP_DATA_DIR=/srv/shiny-server/cenop/data
```

---

## Debugging

### Enable Verbose Logging

#### Set Debug Level in Environment
```bash
# Create or edit .env file
echo 'CENOP_LOG_LEVEL=DEBUG' >> /srv/shiny-server/cenop/.env

# Or set temporarily for testing
export CENOP_LOG_LEVEL=DEBUG
micromamba run -n shiny shiny run app.py --host 0.0.0.0 --port 8000
```

#### Python Debugging with Verbose Output
```bash
cd /srv/shiny-server/cenop
micromamba run -n shiny python -v -c "import cenop; print('CENOP loaded')" 2>&1 | head -100
```

#### Enable Shiny Debug Mode
```bash
cd /srv/shiny-server/cenop
SHINY_LOG_LEVEL=debug micromamba run -n shiny shiny run app.py --reload
```

### Interactive Debugging Session

#### Start Python REPL for Testing
```bash
cd /srv/shiny-server/cenop
micromamba run -n shiny python
```
```python
# Test imports step by step
import sys
sys.path.insert(0, '/srv/shiny-server/cenop/src')

# Test core modules
from cenop.core.simulation import CenopSimulation
from cenop.parameters.simulation_params import SimulationParameters

# Test UI modules
from cenop.ui.layout import create_layout
from cenop.server.main import create_server

print("All imports successful!")
```

#### Debug with pdb
```bash
cd /srv/shiny-server/cenop
micromamba run -n shiny python -m pdb app.py
```

### Log Analysis

#### View All Shiny Server Logs
```bash
# Recent logs
sudo tail -100 /var/log/shiny-server/*.log

# Follow logs in real-time
sudo tail -f /var/log/shiny-server/*.log

# Search for errors
sudo grep -i "error\|exception\|traceback" /var/log/shiny-server/*.log
```

#### View Application-Specific Logs
```bash
# CENOP application logs
sudo tail -f /var/log/shiny-server/cenop/*.log

# Filter by date
sudo grep "$(date +%Y-%m-%d)" /var/log/shiny-server/cenop/*.log
```

#### Journalctl for System Logs
```bash
# Shiny server service logs
sudo journalctl -u shiny-server -f

# Last 50 lines
sudo journalctl -u shiny-server -n 50

# Since last hour
sudo journalctl -u shiny-server --since "1 hour ago"

# Filter errors only
sudo journalctl -u shiny-server -p err
```

### Check Running Processes

```bash
# List all shiny-related processes
ps aux | grep -E "shiny|cenop|python"

# Check open ports
sudo netstat -tlnp | grep -E "3838|8000"

# Or with ss
sudo ss -tlnp | grep -E "3838|8000"

# Check memory usage
ps aux --sort=-%mem | head -10
```

### Network Debugging

```bash
# Test if port is accessible
curl -I http://localhost:3838/cenop/

# Check firewall rules
sudo iptables -L -n | grep 3838

# Test WebSocket connection
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" \
     http://localhost:3838/cenop/
```

---

## Stopping and Terminating

### Graceful Shutdown

#### Stop Shiny Server Service
```bash
# Graceful stop (waits for active connections to close)
sudo systemctl stop shiny-server

# Check it stopped
sudo systemctl status shiny-server
```

#### Stop with Timeout
```bash
# Stop with 30-second timeout for graceful shutdown
sudo systemctl stop shiny-server --timeout=30
```

### Force Termination

#### Kill Shiny Server Immediately
```bash
# Force stop the service
sudo systemctl kill shiny-server

# Or kill all shiny processes
sudo pkill -9 -f shiny-server
```

#### Kill Specific Application Processes
```bash
# Find CENOP-related processes
ps aux | grep cenop

# Kill by PID
sudo kill -9 <PID>

# Kill all Python processes running CENOP
sudo pkill -9 -f "python.*cenop"
```

#### Kill Orphaned Processes
```bash
# Find zombie/orphaned Python processes
ps aux | grep python | grep -v grep

# Kill all Python processes (use with caution!)
sudo pkill -9 python
```

### Restart Procedures

#### Standard Restart
```bash
sudo systemctl restart shiny-server
```

#### Reload Configuration (Without Full Restart)
```bash
sudo systemctl reload shiny-server
```

#### Restart with Clean State
```bash
# Stop completely
sudo systemctl stop shiny-server

# Clear any temporary files
sudo rm -rf /tmp/shiny-*

# Start fresh
sudo systemctl start shiny-server
```

### Check Process Status After Actions
```bash
# Verify no orphaned processes
ps aux | grep -E "shiny|cenop" | grep -v grep

# Verify ports are released
sudo ss -tlnp | grep 3838

# Verify service is running (after restart)
sudo systemctl is-active shiny-server
```

---

## Session Management

### View Active Sessions

```bash
# Check Shiny Server connections
sudo netstat -an | grep 3838 | grep ESTABLISHED

# Count active connections
sudo netstat -an | grep 3838 | grep ESTABLISHED | wc -l
```

### Close User Sessions

#### Force-Close All Sessions (Restart)
```bash
sudo systemctl restart shiny-server
```

#### Disconnect Specific IP
```bash
# Find connections from specific IP
sudo netstat -an | grep 3838 | grep <IP_ADDRESS>

# Block IP temporarily (disconnects session)
sudo iptables -A INPUT -s <IP_ADDRESS> -p tcp --dport 3838 -j DROP

# Unblock later
sudo iptables -D INPUT -s <IP_ADDRESS> -p tcp --dport 3838 -j DROP
```

### Session Cleanup

```bash
# Clear session data
sudo rm -rf /var/lib/shiny-server/sessions/*

# Clear application cache
sudo rm -rf /srv/shiny-server/cenop/__pycache__
sudo rm -rf /srv/shiny-server/cenop/src/cenop/__pycache__

# Restart to apply
sudo systemctl restart shiny-server
```

---

## Troubleshooting

### Check Shiny Server Status
```bash
sudo systemctl status shiny-server
```

### View Application Logs
```bash
sudo tail -f /var/log/shiny-server/cenop/*.log
```

### Check Python Dependencies
```bash
cd /srv/shiny-server/cenop
micromamba run -n shiny pip list
micromamba run -n shiny python -c "import cenop; print('CENOP OK')"
```

### Test Application Locally on Server
```bash
cd /srv/shiny-server/cenop
micromamba run -n shiny shiny run app.py --host 0.0.0.0 --port 8000
```

### Common Issues

#### 1. Import Errors
```
ModuleNotFoundError: No module named 'cenop'
```
**Solution:** Ensure package is installed in micromamba environment:
```bash
cd /srv/shiny-server/cenop
micromamba run -n shiny pip install -e .
```

#### 2. Permission Denied
```
PermissionError: [Errno 13] Permission denied
```
**Solution:** Fix directory ownership:
```bash
sudo chown -R shiny:shiny /srv/shiny-server/cenop
```

#### 3. Missing Dependencies
```
ModuleNotFoundError: No module named 'numpy'
```
**Solution:** Install requirements in micromamba environment:
```bash
micromamba run -n shiny pip install -r /srv/shiny-server/cenop/requirements.txt
```

#### 4. GDAL/Rasterio Issues
If rasterio fails to install, install system dependencies first:
```bash
sudo apt-get install gdal-bin libgdal-dev python3-gdal
micromamba run -n shiny pip install rasterio --no-binary rasterio
```

#### 5. Micromamba Environment Not Found
```
Environment 'shiny' not found
```
**Solution:** Create the micromamba environment:
```bash
micromamba create -n shiny python=3.10
micromamba run -n shiny pip install shiny shinyswatch shinywidgets
```

#### 6. Wrong Python Interpreter
If Shiny Server uses the wrong Python, verify `.python-version` file:
```bash
cat /srv/shiny-server/cenop/.python-version
# Should output: /opt/micromamba/envs/shiny/bin/python
```

---

## Updating the Application

### Quick Update (Files Only)
```cmd
deploy.cmd /sync
```

### Sync and Restart (Most Common)
```cmd
deploy.cmd /sync /restart
```

### Full Update (With Dependency Installation)
```cmd
deploy.cmd /full
```

### Check Server Status
```cmd
deploy.cmd /status
```

### View Server Logs
```cmd
deploy.cmd /logs
```

---

## Security Considerations

1. **SSH Keys:** Use key-based authentication, not passwords
2. **Firewall:** Ensure only necessary ports are open (3838, 22)
3. **HTTPS:** Use a reverse proxy (nginx/Apache) with SSL
4. **File Permissions:** Application files should be owned by shiny user

### Example Nginx Reverse Proxy
```nginx
server {
    listen 443 ssl;
    server_name laguna.ku.lt;
    
    ssl_certificate /etc/ssl/certs/laguna.ku.lt.crt;
    ssl_certificate_key /etc/ssl/private/laguna.ku.lt.key;
    
    location /cenop/ {
        proxy_pass http://localhost:3838/cenop/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Advanced Troubleshooting

### Diagnose Startup Failures

#### Check if App Loads
```bash
cd /srv/shiny-server/cenop
micromamba run -n shiny python -c "
from cenop.ui.layout import create_layout
from cenop.server.main import create_server
from shiny import App
app = App(create_layout(), create_server())
print('App created successfully!')
"
```

#### Check for Syntax Errors
```bash
micromamba run -n shiny python -m py_compile /srv/shiny-server/cenop/app.py
micromamba run -n shiny python -m compileall /srv/shiny-server/cenop/src/cenop/
```

#### Trace Import Issues
```bash
micromamba run -n shiny python -c "
import sys
import traceback
try:
    import cenop
except Exception as e:
    traceback.print_exc()
"
```

### Debug Memory Issues

```bash
# Check current memory usage
free -h

# Find memory-hungry processes
ps aux --sort=-%mem | head -10

# Monitor in real-time
watch -n 1 'ps aux --sort=-%mem | head -5'

# Check for memory leaks (requires htop)
htop -p $(pgrep -f shiny-server)
```

### Debug CPU Issues

```bash
# High CPU usage check
top -p $(pgrep -f shiny-server)

# Process tree
pstree -p $(pgrep -f shiny-server)

# Profile Python execution
micromamba run -n shiny python -m cProfile -s cumtime app.py
```

### Debug File/Permission Issues

```bash
# Check file ownership
ls -la /srv/shiny-server/cenop/

# Check data directory permissions
ls -la /srv/shiny-server/cenop/data/

# Find files not owned by shiny
find /srv/shiny-server/cenop -not -user shiny

# Fix all permissions
sudo chown -R shiny:shiny /srv/shiny-server/cenop
sudo chmod -R 755 /srv/shiny-server/cenop
```

### Debug Data Loading Issues

```bash
micromamba run -n shiny python -c "
from cenop.landscape.loader import LandscapeLoader
import os

data_dir = '/srv/shiny-server/cenop/data'
print(f'Data directory exists: {os.path.exists(data_dir)}')
print(f'Contents: {os.listdir(data_dir)}')

# Try loading a landscape
for landscape in os.listdir(data_dir):
    path = os.path.join(data_dir, landscape)
    if os.path.isdir(path):
        print(f'\\nChecking {landscape}:')
        print(f'  Files: {os.listdir(path)}')
"
```

### Emergency Recovery

#### Reset to Clean State
```bash
# Stop everything
sudo systemctl stop shiny-server
sudo pkill -9 -f shiny

# Clear all caches and temp files
sudo rm -rf /tmp/shiny-*
sudo rm -rf /var/lib/shiny-server/*
sudo rm -rf /srv/shiny-server/cenop/__pycache__
sudo rm -rf /srv/shiny-server/cenop/src/cenop/**/__pycache__

# Reinstall dependencies
cd /srv/shiny-server/cenop
micromamba run -n shiny pip install -e . --force-reinstall

# Start fresh
sudo systemctl start shiny-server
```

#### Rollback Deployment
```bash
# If you have a backup
sudo cp -r /srv/shiny-server/cenop.bak /srv/shiny-server/cenop
sudo systemctl restart shiny-server
```

#### Create Backup Before Changes
```bash
sudo cp -r /srv/shiny-server/cenop /srv/shiny-server/cenop.bak.$(date +%Y%m%d_%H%M%S)
```

---

## Debug Deployment Script

If the main `deploy.cmd` script terminates unexpectedly, use `deploy_debug.cmd` to diagnose the issue.

### Usage

```cmd
deploy_debug.cmd              # Run with debug output (echo on)
deploy_debug.cmd /log         # Log all output to timestamped file
deploy_debug.cmd /step        # Step mode - pause after each operation
deploy_debug.cmd /log /step   # Both logging and stepping
deploy_debug.cmd /help        # Show help
```

### Features

| Feature | Description |
|---------|-------------|
| **Echo On** | Shows every command being executed |
| **Timestamped Logging** | Logs all operations with timestamps |
| **Step Mode** | Pauses after each step for manual inspection |
| **Error Tracking** | Captures and displays ERRORLEVEL after each command |
| **SSH Debugging** | Wraps all SSH commands with detailed logging |
| **Phase Tracking** | Clearly marks SYNC, INSTALL, RESTART phases |

### Diagnosing Early Termination

1. **Run with step mode** to find exactly where it stops:
   ```cmd
   deploy_debug.cmd /step
   ```

2. **Check the log file** for the last successful operation:
   ```cmd
   deploy_debug.cmd /log
   REM After failure, review: deploy_debug_YYYYMMDD_HHMMSS.log
   ```

3. **Common causes of early termination**:
   - SSH connection timeout or failure
   - SSH prompting for password (no key configured)
   - Remote command returning non-zero exit code
   - Network interruption during file transfer
   - `exit /b` not properly returning to caller

### Log File Format

Log files are created in the script directory with format:
```
deploy_debug_20260124_143052.log
```

Log entries include timestamps:
```
[14:30:52.45] [SYNC] Starting file synchronization
[14:30:53.12] [SSH] Executing: Create remote directory
[14:30:53.12] [SSH] Command: ssh razinka@laguna.ku.lt mkdir -p /srv/shiny-server/cenop
[14:30:54.89] [SSH] Result: ERRORLEVEL=0
```

---

## Quick Reference Commands

| Action | Command |
|--------|---------|
| Check status | `sudo systemctl status shiny-server` |
| Start server | `sudo systemctl start shiny-server` |
| Stop server | `sudo systemctl stop shiny-server` |
| Restart server | `sudo systemctl restart shiny-server` |
| Force kill | `sudo pkill -9 -f shiny-server` |
| View logs | `sudo tail -f /var/log/shiny-server/*.log` |
| Check port | `sudo ss -tlnp \| grep 3838` |
| List processes | `ps aux \| grep shiny` |
| Clear cache | `sudo rm -rf /tmp/shiny-*` |
| Test app | `micromamba run -n shiny shiny run app.py` |

---

## Contact

For deployment issues, contact the system administrator of laguna.ku.lt.
