@echo off
echo ========================================
echo CSIRO Biomass Prediction Web App
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Virtual environment not found!
    echo Please run setup.ps1 first.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if activation was successful
if errorlevel 1 (
    echo Failed to activate virtual environment!
    pause
    exit /b 1
)

echo.
echo Starting Flask application...
echo Web interface will be available at: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.

REM Run the application
python app.py

pause
