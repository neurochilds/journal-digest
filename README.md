# Journal Digest (Neuroscience Paper Tracker)

This repo runs a daily/weekly digest that scans neuroscience papers (OpenAlex by default), scores relevance with keywords + GPT, summarizes top papers, and emails a digest. It keeps track of previously seen papers in `seen_papers.json` so you don’t get duplicates.

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
- `historical`: Set to `true` to use the OpenAlex search (default). Set to `false` to use RSS feeds.
- `start_date`: Start date in `YYYY-MM-DD` (overrides `days`).
- `end_date`: End date in `YYYY-MM-DD` (optional).
- `max_llm_candidates`: Max papers to send for AI scoring (default 200).
- `dry_run`: Set to `true` to score and log everything without sending the email.

**Examples**
- Look back 7 days:
  - `days: 7`
  - `include_seen: false`
- Run a specific date range (recommended with OpenAlex):
  - `start_date: 2026-01-01`
  - `end_date: 2026-01-15`
  - `historical: true`

The workflow updates `seen_papers.json` and `digest_log.csv` after each run and commits them back to the repo.

## What Gets Logged

`digest_log.csv` is the permanent record: one row per paper that reached AI scoring, with the run date, title, journal, link, publication date, keyword score, AI score, combined score, whether it was emailed, and the AI's one-line reason. `seen_papers.json` is only a dedup index of opaque hashes - use the CSV to see what actually happened.

## Tuning (config.py)

| Setting | Default | What it does |
| --- | --- | --- |
| `DAYS_TO_CHECK` | 14 | Publication-date lookback window |
| `CREATED_WINDOW_DAYS` | 21 | Additional sweep by OpenAlex *index* date, catching late deposits |
| `MAX_LLM_CANDIDATES` | 200 | Cost ceiling on AI scoring. Truncation is now logged loudly |
| `MIN_KEYWORD_SCORE` | 12 | Raw keyword score needed to become a candidate |
| `MIN_LLM_SCORE` | 40 | Hard floor - the AI can veto a keyword-dense paper |
| `MIN_COMBINED_SCORE` | 40 | Final threshold on the weighted score |
| `KEYWORD_WEIGHT` | 0.3 | Keyword share of the combined score; AI gets the rest |
| `SEEN_RETENTION_DAYS` | 365 | How long a paper stays suppressed as already seen |

A paper with a core term in its **title** (hippocampal, entorhinal, theta, replay, remapping, multisensory, ...) is always ranked ahead of keyword-dense abstracts when the candidate cap bites.

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
