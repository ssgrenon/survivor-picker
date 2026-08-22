@echo off
REM Double-click launcher for the Survivor Picker app (Windows).
REM See README.md > "Running the app without a terminal" for details.

cd /d "%~dp0"

python --version >nul 2>nul
if errorlevel 1 (
    echo Python was not found on this PC.
    echo Install it from https://www.python.org/downloads/ and try again.
    echo Be sure to check "Add python.exe to PATH" during installation.
    pause
    exit /b 1
)

echo Survivor Picker: installing/updating dependencies...
python -m pip install --quiet --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo.
    echo Failed to install dependencies. See the error above.
    pause
    exit /b 1
)

echo Survivor Picker: starting the app...
echo A browser tab will open automatically once it's ready.
echo Close this window to stop the app.
python -m streamlit run ui\app.py

echo.
echo Survivor Picker has stopped.
pause
