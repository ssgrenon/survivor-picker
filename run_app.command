#!/bin/bash
# Double-click launcher for the Survivor Picker app (macOS).
# See README.md > "Running the app without a terminal" for details.

cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 is required but wasn't found on this Mac."
    echo "Install it from https://www.python.org/downloads/ and try again."
    read -n 1 -s -r -p "Press any key to close this window..."
    exit 1
fi

echo "Survivor Picker: installing/updating dependencies..."
python3 -m pip install --quiet --disable-pip-version-check -r requirements.txt
if [ $? -ne 0 ]; then
    echo
    echo "Failed to install dependencies. See the error above."
    read -n 1 -s -r -p "Press any key to close this window..."
    exit 1
fi

echo "Survivor Picker: starting the app..."
echo "A browser tab will open automatically once it's ready."
echo "Close this window to stop the app."
python3 -m streamlit run ui/app.py

echo
echo "Survivor Picker has stopped."
read -n 1 -s -r -p "Press any key to close this window..."
