# Desktop launcher (macOS)

You now have a double-clickable launcher:

- `Neuro Paper Tracker.command`

## How it works
- Prompts for **days back** (1–90).
- Prompts whether to **Include All** (ignore `seen_papers.json`) or **Skip Previous**.
- Runs your venv Python: `venv/bin/python paper_tracker.py --days <N> [--include-seen]`.

## Put it on the Desktop
In Finder:
1. Open `neuro_paper_tracker/`
2. Drag `Neuro Paper Tracker.command` to your Desktop (Option-drag to copy).

## If it won’t run
- Right click → **Open** the first time (Gatekeeper).
- Ensure the venv exists:
  - Run `setup.sh` once, or create `venv/` manually.

## Notes
- The launcher assumes this project is located at:
  `/Users/ninja/Desktop/ClaudeTest/neuro_paper_tracker`
  If you move the folder, update `SCRIPT_DIR` inside the `.command` file.
