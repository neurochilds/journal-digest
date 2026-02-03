-- Neuroscience Paper Tracker Launcher
-- Double-click to run with custom days lookback

set defaultDays to "3"
set dialogResult to display dialog "How many days back to check for papers?" default answer defaultDays buttons {"Cancel", "Run"} default button "Run" with title "Neuro Paper Tracker"

if button returned of dialogResult is "Run" then
    set daysBack to text returned of dialogResult

    -- Validate input is a number
    try
        set daysNum to daysBack as integer
        if daysNum < 1 or daysNum > 90 then
            display alert "Please enter a number between 1 and 90"
            return
        end if
    on error
        display alert "Please enter a valid number"
        return
    end try

    -- Ask about including previously seen papers
    set includeSeenResult to display dialog "Include previously found papers?" buttons {"Skip Previous", "Include All"} default button "Skip Previous" with title "Paper History"

    set includeSeen to ""
    if button returned of includeSeenResult is "Include All" then
        set includeSeen to " --include-seen"
    end if

    -- Get the script directory
    set scriptPath to "/Users/ninja/Desktop/ClaudeTest/neuro_paper_tracker"
    set pythonPath to scriptPath & "/venv/bin/python"
    set trackerPath to scriptPath & "/paper_tracker.py"

    -- Run the tracker in Terminal so user can see progress
    tell application "Terminal"
        activate
        do script "cd " & quoted form of scriptPath & " && " & quoted form of pythonPath & " " & quoted form of trackerPath & " " & daysBack & includeSeen & "; echo ''; echo 'Press any key to close...'; read -n 1"
    end tell
end if
