#!/bin/zsh
# Double-clickable launcher for Neuro Paper Tracker - Historical Mode (OpenAlex API)
# Use this for date ranges > 2 weeks where RSS feeds don't have data.
# Prompts for start/end dates + whether to include previously seen papers.

set -euo pipefail

SCRIPT_DIR="/Users/ninja/Desktop/ClaudeTest/neuro_paper_tracker"
PY="$SCRIPT_DIR/venv/bin/python"
APP="$SCRIPT_DIR/paper_tracker.py"

cd "$SCRIPT_DIR"

# Get today's date and a default start date (30 days ago for historical)
TODAY=$(date +%Y-%m-%d)
DEFAULT_START=$(date -v-30d +%Y-%m-%d)

# Explain what this mode does
osascript -e 'display dialog "HISTORICAL MODE (OpenAlex API)\n\nThis mode searches the OpenAlex database for papers.\nUse this for date ranges longer than ~2 weeks,\nwhere RSS feeds no longer have data.\n\nAbstracts are included in results." buttons {"Cancel","Continue"} default button "Continue" with title "Neuro Paper Tracker - Historical"'

# Prompt for start date
START_DATE=$(osascript -e "text returned of (display dialog \"Enter START date (earliest papers to include):\" default answer \"$DEFAULT_START\" buttons {\"Cancel\",\"Next\"} default button \"Next\" with title \"Historical Search - Date Range\")")

# Validate start date format
if ! [[ "$START_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  osascript -e 'display alert "Invalid date format" message "Please use YYYY-MM-DD format (e.g., 2025-01-15)"'
  exit 1
fi

# Prompt for end date
END_DATE=$(osascript -e "text returned of (display dialog \"Enter END date (latest papers to include):\" default answer \"$TODAY\" buttons {\"Cancel\",\"Run\"} default button \"Run\" with title \"Historical Search - Date Range\")")

# Validate end date format
if ! [[ "$END_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  osascript -e 'display alert "Invalid date format" message "Please use YYYY-MM-DD format (e.g., 2025-01-20)"'
  exit 1
fi

# Prompt include seen
INCLUDE_SEEN=$(osascript -e 'button returned of (display dialog "Include previously found papers?" buttons {"Skip Previous","Include All"} default button "Skip Previous" with title "Paper History")')

ARGS=("--historical" "--start-date" "$START_DATE" "--end-date" "$END_DATE")
if [[ "$INCLUDE_SEEN" == "Include All" ]]; then
  ARGS+=("--include-seen")
fi

# Run in this Terminal window
"$PY" "$APP" "${ARGS[@]}"

echo ""
echo "Done. Press Enter to close."
read -r
