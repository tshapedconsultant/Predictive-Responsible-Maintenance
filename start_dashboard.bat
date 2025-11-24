@echo off
REM Change to the directory where this batch file is located
cd /d "%~dp0"
echo ========================================
echo Starting Predictive Maintenance Dashboard
echo ========================================
echo.
echo Current directory: %CD%
echo.

REM Check if dashboard.py exists
if not exist "dashboard.py" (
    echo ERROR: dashboard.py not found!
    echo Current directory: %CD%
    echo Please make sure you're running this from the project folder.
    pause
    exit /b 1
)

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python or add it to PATH.
    pause
    exit /b 1
)

REM Check if Streamlit is installed
python -m streamlit --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Streamlit not installed!
    echo Installing Streamlit...
    pip install streamlit
    if errorlevel 1 (
        echo Failed to install Streamlit. Please run: pip install streamlit
        pause
        exit /b 1
    )
)

echo Found dashboard.py - starting Streamlit...
echo.
echo ========================================
echo IMPORTANT: Keep this window open!
echo ========================================
echo.
echo Dashboard will be available at: http://localhost:8502
echo.
echo If you see errors below, please read them carefully.
echo.
echo Press Ctrl+C to stop the dashboard
echo.
echo ========================================
echo.

REM Start Streamlit - this will keep running until you press Ctrl+C
python -m streamlit run "dashboard.py" --server.port 8502 --server.headless false

echo.
echo Dashboard stopped.
pause
