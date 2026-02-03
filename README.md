# Journal Digest (Neuroscience Paper Tracker)

This repo runs a daily/weekly digest that scans neuroscience journal feeds, scores relevance with keywords + GPT, summarizes top papers, and emails a digest. It keeps track of previously seen papers in `seen_papers.json` so you don’t get duplicates.

## How It Runs (GitHub Actions)
The workflow lives at:
- `.github/workflows/paper-digest.yml`

### Schedule
Currently scheduled for **Mondays and Thursdays at 09:45 UTC**.
If you want **09:45 in your local timezone**, update the cron time accordingly (GitHub schedules use UTC).

### Manual Runs
You can [run the workflow manually](https://github.com/neurochilds/journal-digest/actions/workflows/paper-digest.yml) with custom inputs:
1. Go to **Actions** → **Neuro Paper Digest**.
2. Click **Run workflow**.
3. Fill any of the optional inputs:

- `days`: Number of days to look back (1–90). Leave blank to use the default from `config.py`.
- `include_seen`: Set to `true` to include previously seen papers (ignores `seen_papers.json`).
- `historical`: Set to `true` to use the OpenAlex historical search (better for longer date ranges).
- `start_date`: Start date in `YYYY-MM-DD` (overrides `days`).
- `end_date`: End date in `YYYY-MM-DD` (optional).

**Examples**
- Look back 7 days:
  - `days: 7`
  - `include_seen: false`
- Run a specific date range:
  - `start_date: 2026-01-01`
  - `end_date: 2026-01-15`

The workflow will update `seen_papers.json` after each run and commit it back to the repo.

## Required Secrets
Add these under **Settings → Secrets and variables → Actions**:
- `OPENAI_API_KEY`
- `GMAIL_ADDRESS`
- `GMAIL_APP_PASSWORD`
- `RECIPIENT_EMAIL`

## Local Usage (Optional)
For local runs, create a `config_local.py` with your secrets (ignored by git):

```python
GMAIL_ADDRESS = "you@gmail.com"
GMAIL_APP_PASSWORD = "your_app_password"
RECIPIENT_EMAIL = "you@domain.com"
OPENAI_API_KEY = "sk-..."
```

Then run:

```bash
python paper_tracker.py --days 3
```
