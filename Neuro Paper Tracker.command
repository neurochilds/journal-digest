#!/bin/zsh
# Double-clickable launcher for Neuro Paper Tracker
# Prompts for lookback window + whether to include previously seen papers.

set -euo pipefail

SCRIPT_DIR="/Users/ninja/Desktop/ClaudeTest/neuro_paper_tracker"
PY="$SCRIPT_DIR/venv/bin/python"
APP="$SCRIPT_DIR/paper_tracker.py"

cd "$SCRIPT_DIR"

# Prompt for days (default from config is fine; we default to 3 here)
DAYS=$(osascript -e 'text returned of (display dialog "How many days back to check for papers? (1-90)" default answer "3" buttons {"Cancel","Run"} default button "Run" with title "Neuro Paper Tracker")')

# Validate days
if ! [[ "$DAYS" =~ '^[0-9]+$' ]]; then
  osascript -e 'display alert "Please enter a valid number"'
  exit 1
fi
if (( DAYS < 1 || DAYS > 90 )); then
  osascript -e 'display alert "Please enter a number between 1 and 90"'
  exit 1
fi

# Prompt include seen
INCLUDE_SEEN=$(osascript -e 'button returned of (display dialog "Include previously found papers?" buttons {"Skip Previous","Include All"} default button "Skip Previous" with title "Paper History")')

ARGS=("--days" "$DAYS")
if [[ "$INCLUDE_SEEN" == "Include All" ]]; then
  ARGS+=("--include-seen")
fi

# Run in this Terminal window
"$PY" "$APP" "${ARGS[@]}"

echo ""
echo "Done. Press Enter to close."
read -r
