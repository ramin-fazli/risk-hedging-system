@echo off
echo Installing FBX Hedging Strategy Backtesting System...
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Create directories
echo Creating directories...
mkdir data_files 2>nul
mkdir reports 2>nul
mkdir logs 2>nul
mkdir charts 2>nul

echo.
echo Installation complete!
echo.
echo To run the system:
echo 1. Activate virtual environment: venv\Scripts\activate
echo 2. Run the system: python main.py
echo.
pause
