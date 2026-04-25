@echo off
echo ==========================================
echo STARTING DATA PIPELINE
echo ==========================================

echo [1/2] Merging Datasets...
python utils/merge_datasets.py
if %ERRORLEVEL% NEQ 0 (
    echo Error merging datasets!
    exit /b %ERRORLEVEL%
)

echo.
echo [2/2] Starting Sophisticated Training...
echo This may take a while. Please do not close this window.
python train_sophisticated.py
if %ERRORLEVEL% NEQ 0 (
    echo Error during training!
    exit /b %ERRORLEVEL%
)

echo.
echo ==========================================
echo PIPELINE COMPLETE!
echo You can now restart the app.
echo ==========================================
pause
