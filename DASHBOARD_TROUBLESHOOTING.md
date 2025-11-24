# Dashboard Troubleshooting Guide

## If Dashboard Won't Start or Connect

### Step 1: Run the Batch File
1. **Double-click `start_dashboard.bat`**
2. **Keep the window open** - don't close it!
3. Wait for the message: "You can now view your Streamlit app in your browser"
4. Open your browser to: http://localhost:8502

### Step 2: Check for Errors
If the batch file window shows errors:
- **"Python not found"** → Install Python and add it to PATH
- **"Streamlit not installed"** → Run: `pip install streamlit`
- **"dashboard.py not found"** → Make sure you're in the project folder

### Step 3: Manual Start (Alternative)
If the batch file doesn't work:

1. **Open PowerShell or Command Prompt**
2. **Navigate to the project folder:**
   ```powershell
   cd "C:\Users\andres\Desktop\predictive manteiance"
   ```

3. **Run the dashboard:**
   ```bash
   python -m streamlit run dashboard.py --server.port 8502
   ```

4. **Keep the terminal window open** - the dashboard runs in this window!

5. **Open browser to:** http://localhost:8502

### Step 4: Try Different Ports
If port 8502 is busy, try:
```bash
python -m streamlit run dashboard.py --server.port 8503
```
Then access: http://localhost:8503

### Step 5: Check Firewall
Windows Firewall might be blocking the connection:
1. Open Windows Defender Firewall
2. Allow Python through firewall
3. Or temporarily disable firewall to test

### Step 6: Verify Dependencies
Make sure all packages are installed:
```bash
pip install -r requirements.txt
```

### Common Issues:

**"Connection Refused"**
- The dashboard process isn't running
- Make sure you **kept the terminal/batch window open**
- Check if port is already in use: `netstat -ano | findstr :8502`

**"Port already in use"**
- Kill the process: `taskkill /F /IM python.exe`
- Or use a different port (8503, 8504, etc.)

**"Module not found"**
- Run: `pip install streamlit plotly pandas`

**Dashboard loads but shows errors**
- Make sure you've run `python ml_pipeline.py` first to generate artifacts
- Check that `artifacts/` folder contains the required files

### Still Not Working?
1. Check the terminal window for error messages
2. Make sure Python and all packages are installed correctly
3. Try running from a different terminal (PowerShell vs CMD)
4. Restart your computer and try again

