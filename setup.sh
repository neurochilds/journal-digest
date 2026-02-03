#!/bin/bash
#
# Setup script for Neuroscience Paper Tracker
# Run this once to install dependencies and configure the daily scheduler
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLIST_NAME="com.neuro.papertracker"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_NAME}.plist"

echo "=== Neuroscience Paper Tracker Setup ==="
echo ""

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found."
    echo "Install it from https://www.python.org/ or via Homebrew: brew install python"
    exit 1
fi

PYTHON_PATH=$(which python3)
echo "Using Python: $PYTHON_PATH"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv "$SCRIPT_DIR/venv"

# Install dependencies
echo "Installing dependencies..."
"$SCRIPT_DIR/venv/bin/pip" install --upgrade pip
"$SCRIPT_DIR/venv/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "Dependencies installed successfully!"

# Create launchd plist
echo ""
echo "Creating launchd scheduler..."

cat > "$PLIST_PATH" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_NAME}</string>

    <key>ProgramArguments</key>
    <array>
        <string>${SCRIPT_DIR}/venv/bin/python</string>
        <string>${SCRIPT_DIR}/paper_tracker.py</string>
    </array>

    <key>WorkingDirectory</key>
    <string>${SCRIPT_DIR}</string>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>8</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>${SCRIPT_DIR}/logs/stdout.log</string>

    <key>StandardErrorPath</key>
    <string>${SCRIPT_DIR}/logs/stderr.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
EOF

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

echo "Launchd plist created at: $PLIST_PATH"
echo "Scheduled to run daily at 8:00 AM"

# Load the scheduler
echo ""
echo "Loading scheduler..."
launchctl unload "$PLIST_PATH" 2>/dev/null || true
launchctl load "$PLIST_PATH"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "IMPORTANT: Before the tracker will work, you need to:"
echo ""
echo "1. Edit config.py and add your:"
echo "   - Gmail address (GMAIL_ADDRESS)"
echo "   - Gmail App Password (GMAIL_APP_PASSWORD)"
echo "     Get one at: Google Account > Security > 2-Step Verification > App Passwords"
echo "   - Anthropic API key (ANTHROPIC_API_KEY)"
echo "     Get one at: https://console.anthropic.com/"
echo ""
echo "2. Test the tracker manually:"
echo "   cd $SCRIPT_DIR"
echo "   ./venv/bin/python paper_tracker.py"
echo ""
echo "3. The tracker will now run automatically every day at 8:00 AM"
echo "   (your Mac needs to be on and logged in)"
echo ""
echo "To change the schedule time, edit: $PLIST_PATH"
echo "Then run: launchctl unload \"$PLIST_PATH\" && launchctl load \"$PLIST_PATH\""
echo ""
echo "To run manually anytime:"
echo "   cd $SCRIPT_DIR && ./venv/bin/python paper_tracker.py"
echo ""
echo "To stop the daily scheduler:"
echo "   launchctl unload \"$PLIST_PATH\""
