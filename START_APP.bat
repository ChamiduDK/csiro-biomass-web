@echo off
echo.
echo ========================================
echo   CSIRO Biomass Prediction Web App
echo        ULTIMATE PIPELINE
echo ========================================
echo.

REM 1. Environment Setup
if exist "venv310\Scripts\activate.bat" goto use_venv310
if exist "venv\Scripts\activate.bat" goto use_venv
echo [ERROR] No virtual environment found!
pause
exit /b 1

:use_venv310
echo [INFO] Activating Python 3.10 environment...
call venv310\Scripts\activate.bat
goto check_deps

:use_venv
echo [WARN] venv310 not found. Using standard 'venv'...
call venv\Scripts\activate.bat
goto check_deps

:check_deps
echo.
python --version
echo [INFO] Checking dependencies...
pip install -r requirements.txt
if errorlevel 1 goto install_error

REM 3. Model Verification
echo.
echo [INFO] Verifying models...
if not exist "models\ensemble_models.pkl" goto train_models
echo [INFO] Real models found. fast-forwarding...
goto start_app

:train_models
echo [WARN] Trained models not found.
echo [INFO] Starting training pipeline...
python train_pipeline.py
if errorlevel 1 goto train_error
echo [SUCCESS] Models trained and saved!
goto start_app

:start_app
echo.
echo [INFO] Starting Flask application...
echo ========================================
echo Access: http://localhost:5000
echo ========================================
python app.py
pause
exit /b 0

:install_error
echo [ERROR] Failed to install dependencies.
pause
exit /b 1

:train_error
echo [ERROR] Training failed!
pause
exit /b 1
