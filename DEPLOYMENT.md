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
- Python 3.10 or higher
- Shiny Server for Python installed
- User `razinka` with:
  - Write access to `/srv/shiny-server/cenop`
  - Sudo access to restart shiny-server (optional)

---

## Quick Deployment

### Option 1: Using the Deployment Script

```cmd
cd c:\Users\DELL\OneDrive - ku.lt\HORIZON_EUROPE\AI4WIND\CENOP\cenop
deploy.cmd
```

This will:
1. Create the remote directory
2. Sync all application files
3. Create/update virtual environment
4. Install Python dependencies
5. Install CENOP package
6. Restart Shiny Server

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

#### Step 4: Set Up Python Environment (on server)
```bash
cd /srv/shiny-server/cenop
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

#### Step 5: Restart Shiny Server
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
        python /srv/shiny-server/cenop/.venv/bin/python;
        log_dir /var/log/shiny-server/cenop;
        directory_index on;
    }
    
    # Other applications can be added here
}
```

### Alternative: Per-Application Python Configuration

If using Shiny Server Pro or a custom setup, create `.python-version` in the app directory:

```bash
echo "/srv/shiny-server/cenop/.venv/bin/python" > /srv/shiny-server/cenop/.python-version
```

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
source .venv/bin/activate
pip list
python -c "import cenop; print('CENOP OK')"
```

### Test Application Locally on Server
```bash
cd /srv/shiny-server/cenop
source .venv/bin/activate
shiny run app.py --host 0.0.0.0 --port 8000
```

### Common Issues

#### 1. Import Errors
```
ModuleNotFoundError: No module named 'cenop'
```
**Solution:** Ensure package is installed:
```bash
cd /srv/shiny-server/cenop
source .venv/bin/activate
pip install -e .
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
**Solution:** Install requirements:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

#### 4. GDAL/Rasterio Issues
If rasterio fails to install, install system dependencies first:
```bash
sudo apt-get install gdal-bin libgdal-dev python3-gdal
pip install rasterio --no-binary rasterio
```

---

## Updating the Application

### Quick Update (Files Only)
```cmd
deploy.cmd --sync-only
```
Then SSH to server to restart:
```bash
ssh razinka@laguna.ku.lt "sudo systemctl restart shiny-server"
```

### Full Update
```cmd
deploy.cmd
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

## Contact

For deployment issues, contact the system administrator of laguna.ku.lt.
