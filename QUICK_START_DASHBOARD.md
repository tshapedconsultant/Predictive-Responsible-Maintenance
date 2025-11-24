# Quick Start - Dashboard

## If Dashboard Won't Start Automatically

### Option 1: Run the Batch File
Double-click `start_dashboard.bat` in the project folder.

### Option 2: Manual Command
Open PowerShell or Command Prompt in this folder and run:
```bash
python -m streamlit run dashboard.py --server.port 8502
```

### Option 3: Try Different Ports
If port 8502 doesn't work, try:
```bash
python -m streamlit run dashboard.py --server.port 8503
```
or
```bash
python -m streamlit run dashboard.py --server.port 8504
```

### Troubleshooting

1. **Check if port is in use:**
   ```powershell
   netstat -ano | findstr :8502
   ```

2. **Kill any stuck Python processes:**
   ```powershell
   taskkill /F /IM python.exe
   ```
   Then try starting again.

3. **Check for errors:**
   Run the dashboard command directly (not in background) to see error messages.

4. **Verify dependencies:**
   ```bash
   pip install streamlit plotly pandas
   ```

### Access Dashboard
Once started, open your browser to:
- http://localhost:8502 (or whatever port you used)

