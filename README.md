# Journal Digest (Neuroscience Paper Tracker)

This repo runs a daily/weekly digest that scans neuroscience papers (OpenAlex by default), scores relevance with keywords + GPT, summarizes top papers, and emails a digest. It keeps track of previously seen papers in `seen_papers.json` so you don’t get duplicates.

## How It Runs (Umbrel)
Production runs through `deploy/paper-scout.service` and `deploy/paper-scout.timer` on Umbrel. See [runtime and cutover details](deploy/README.md). Persistent history lives in `/var/lib/paper-scout`; the source repository retains the history snapshot from cutover. GitHub Actions is now for manual previews only.

The Umbrel service uses **Codex subscription authentication**, with **GPT-6.1 Sol
at Medium** for relevance and summaries. It batches ten abstracts per call,
preserves completed scores, and never falls back to paid OpenAI API calls.
Quota or authentication failures retain candidates for a later run. The legacy
desktop and optional hosted AI-preview paths still explicitly use an API key;
they are not part of the native schedule. Native Codex auth lives outside Git.

### Schedule
Currently scheduled for **Mondays and Thursdays at 09:45 UTC**.
The native timer uses UTC: 10:45 during British summer time and 09:45 in winter. There is no hosted production cron. Missed native runs catch up after downtime; initial cutover skips the already-delivered hosted run.

### Manual Runs
You can [run the workflow manually](https://github.com/neurochilds/journal-digest/actions/workflows/paper-digest.yml) with custom inputs:
1. Go to **Actions** → **Neuro Paper Digest**.
2. Click **Run workflow**.
3. Fill any of the optional inputs:

- `days`: Number of days to look back (1–90). Leave blank for the default 60-day publication overlap in OpenAlex mode.
- `include_seen`: Set to `true` to include previously seen papers (ignores `seen_papers.json`).
- `historical`: Set to `true` to use the OpenAlex search (default). Set to `false` to use RSS feeds.
- `start_date`: Start date in `YYYY-MM-DD` (overrides `days`).
- `end_date`: End date in `YYYY-MM-DD` (optional).
- `max_llm_candidates`: Max papers to send for AI scoring (default 200).
- `dry_run`: Set to `true` to score and preview without sending email or changing state/logs. This still makes paid AI calls.
- `fetch_only`: Set to `true` for retrieval and keyword ranking without AI calls, email or state/log changes.

Preview runs upload a `paper-scout-preview` HTML artifact. The digest job has a 40-minute timeout; OpenAlex retrieval has a shared 5-minute budget, at most five attempts per request and a 300-page ceiling, and fails rather than returning incomplete results. Runs on the same branch are serialized.

**Examples**
- Look back 7 days:
  - `days: 7`
  - `include_seen: false`
- Run a specific date range (recommended with OpenAlex):
  - `start_date: 2026-01-01`
  - `end_date: 2026-01-15`
  - `historical: true`

Production updates persistent state on Umbrel and makes a private pre-run snapshot; previews do not change delivery history. Hosted previews require `dry_run` or `fetch_only` and cannot send email. Their repository history is the cutover snapshot, so later preview candidates may differ from the worker’s current queue.

## OpenAlex repair and late indexing

OpenAlex's `from_created_date` and `from_updated_date` filters require a paid plan. The previous mandatory created-date pass returned a permanent "Plan upgrade required" error as HTTP 429, which was retried indefinitely. Default runs now use one wider, free publication-date window. A free API key increases the daily budget but does not unlock the paid filters.

The default revisits 60 days of publications. Papers first indexed after that overlap can still be missed; it is not equivalent to a true created-date subscription. An explicit `--days` or date range uses exactly the requested publication window.

To preview the period missed since the last successful run, without consuming papers or calling the AI:

```bash
python paper_tracker.py --start-date 2026-08-28 --end-date 2026-10-03 --fetch-only --preview-file preview-backfill.html
```

After approving the repair and reviewing the preview, run an AI dry run with a bounded candidate count before a catch-up delivery. Do not use `--include-seen` for a normal catch-up unless resending old papers is intentional.

OpenAlex references: [authentication](https://help.openalex.org/api/authentication/), [paid sync filters](https://help.openalex.org/api/filtering/#sync-filters-paid-plans).

## What Gets Logged

`digest_log.csv` is the permanent record: one row per paper that reached AI scoring, with the run date, title, journal, link, publication date, keyword score, AI score, combined score, whether it was emailed, and the AI's one-line reason. `seen_papers.json` is only a dedup index of opaque hashes - use the CSV to see what actually happened.

- `first_observed.json` records when this tracker first observed a work, not when OpenAlex created it. Retrieval previews do not change it.
- `pending_papers.json` retains candidates deferred by the AI/digest caps or failed delivery, including completed scores. Default runs also drain this queue after papers leave the publication overlap. Explicit backfills preserve queued work outside their requested window.
- Relevant papers become seen after SMTP accepts the digest. Rejected scored papers are also recorded as processed on a successful run. API failures stop the run, and delivery failures return a nonzero exit code without consuming the selected papers.
- Seen and queue JSON files are replaced atomically. SMTP and local files cannot form one atomic transaction. The native wrapper durably marks uncertainty before SMTP and clears its guard only after the tracker finishes its state writes. An ambiguous/crashed delivery blocks subsequent native production runs for reconciliation, preventing automatic replay. Direct CLI runs outside that wrapper do not have this extra guard; exact-once delivery is not guaranteed.

## Tuning (config.py)

| Setting | Default | What it does |
| --- | --- | --- |
| `DAYS_TO_CHECK` | 14 | Publication-date lookback window |
| `PUBLICATION_OVERLAP_DAYS` | 60 | Wider publication window for default OpenAlex runs, catching some late deposits |
| `MAX_LLM_CANDIDATES` | 200 | Cost ceiling on AI scoring. Truncation is now logged loudly |
| `MIN_KEYWORD_SCORE` | 12 | Raw keyword score needed to become a candidate |
| `MIN_LLM_SCORE` | 50 | Hard floor - the AI can veto a keyword-dense paper |
| `MIN_COMBINED_SCORE` | 40 | Final threshold on the weighted score |
| `KEYWORD_WEIGHT` | 0.3 | Keyword share of the combined score; AI gets the rest |
| `SEEN_RETENTION_DAYS` | 365 | How long a paper stays suppressed as already seen |

A paper with a core term in its **title** (hippocampal, entorhinal, theta, replay, remapping, multisensory, ...) is always ranked ahead of keyword-dense abstracts when the candidate cap bites.

## Required Secrets
Native credentials are root-owned outside source under `/etc/paper-scout` and loaded privately by systemd. For hosted previews, add these under **Settings → Secrets and variables → Actions**:
- `OPENAI_API_KEY`
- `GMAIL_ADDRESS`
- `GMAIL_APP_PASSWORD`
- `RECIPIENT_EMAIL`
- `OPENALEX_API_KEY` — recommended free key from [OpenAlex settings](https://openalex.org/settings/api); smaller unauthenticated requests still work without it.

## Local Usage (Optional)
For local runs, create a `config_local.py` with your secrets (ignored by git):

```python
GMAIL_ADDRESS = "you@gmail.com"
GMAIL_APP_PASSWORD = "your_app_password"
RECIPIENT_EMAIL = "you@domain.com"
OPENAI_API_KEY = "sk-..."
OPENALEX_API_KEY = "your_openalex_key"
```

Then run:

```bash
python paper_tracker.py --days 3
```

Run deterministic regression tests with `python -m unittest discover -s tests -v`. They make no network requests and send no email.
