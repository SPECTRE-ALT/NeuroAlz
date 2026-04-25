@echo off
echo Starting Alzheimer's Detection API...
start http://localhost:5001
echo Access at http://localhost:5001
python api/app.py
pause
