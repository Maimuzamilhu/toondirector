@echo off
cd /d "%~dp0"
if not exist .venv (
  echo Creating virtual environment with Python 3.12...
  py -3.12 -m venv .venv || goto :err
  .venv\Scripts\python.exe -m pip install -r requirements.txt || goto :err
)
start "" .venv\Scripts\pythonw.exe -m studio
goto :eof
:err
echo Setup failed. See the message above.
pause
