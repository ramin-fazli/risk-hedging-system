@echo off
echo Running FBX Hedging Strategy Backtesting System...
echo.

REM Activate virtual environment
call venv\Scripts\activate

REM Run the system
python main.py

echo.
echo System execution completed!
echo Check the reports directory for the generated Excel report.
echo.
pause
