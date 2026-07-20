#!/usr/bin/env python3
"""Optional live RSS diagnostic; deterministic tests live under tests/."""

from datetime import datetime, timedelta

from config import DAYS_TO_CHECK, JOURNAL_FEEDS, MIN_KEYWORD_SCORE
from paper_tracker import calculate_keyword_score, fetch_papers_from_feed


def main() -> int:
    cutoff = datetime.now() - timedelta(days=DAYS_TO_CHECK)
    total = 0
    passing = 0
    for journal, url in JOURNAL_FEEDS.items():
        papers = fetch_papers_from_feed(journal, url, cutoff)
        total += len(papers)
        for paper in papers:
            score, _ = calculate_keyword_score(paper["title"], paper.get("abstract", ""))
            passing += score >= MIN_KEYWORD_SCORE
        print(f"{journal}: {len(papers)} recent")
    print(f"Total recent: {total}; keyword pass: {passing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
