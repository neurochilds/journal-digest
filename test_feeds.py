#!/usr/bin/env python3
"""
Test script to verify RSS feeds are working and see what papers would be found.
Run this before setting up credentials to make sure everything works.
"""

import feedparser
from datetime import datetime, timedelta
import re
import sys

# Import config (but we don't need credentials for this)
from config import JOURNAL_FEEDS, KEYWORDS, MIN_RELEVANCE_SCORE, DAYS_TO_CHECK


def calculate_relevance(title: str, abstract: str) -> tuple[int, list[str]]:
    """Calculate relevance score."""
    text = f"{title} {abstract}".lower()
    matched_terms = []
    score = 0

    weights = {
        "primary_topics": 10,
        "multisensory": 12,
        "techniques": 8,
        "theories_and_people": 6,
        "sensory": 5,
        "behavior": 4,
    }

    for category, terms in KEYWORDS.items():
        weight = weights.get(category, 5)
        for term in terms:
            if term.lower() in text:
                if term.lower() in title.lower():
                    score += weight * 2
                    matched_terms.append(f"{term} (title)")
                else:
                    score += weight
                    matched_terms.append(term)

    seen = set()
    unique_terms = []
    for term in matched_terms:
        base_term = term.replace(" (title)", "")
        if base_term not in seen:
            seen.add(base_term)
            unique_terms.append(term)

    return score, unique_terms


def main():
    print("=" * 70)
    print("NEUROSCIENCE PAPER TRACKER - FEED TEST")
    print("=" * 70)
    print(f"Testing with {DAYS_TO_CHECK} day(s) lookback")
    print(f"Minimum relevance score: {MIN_RELEVANCE_SCORE}")
    print()

    cutoff_date = datetime.now() - timedelta(days=DAYS_TO_CHECK)
    all_papers = []
    relevant_papers = []

    for journal_name, feed_url in JOURNAL_FEEDS.items():
        print(f"Testing: {journal_name}")
        print(f"  URL: {feed_url[:60]}...")

        try:
            feed = feedparser.parse(feed_url)

            if feed.bozo and feed.bozo_exception:
                print(f"  WARNING: Feed parsing issue: {feed.bozo_exception}")

            recent_count = 0
            for entry in feed.entries:
                pub_date = None
                for date_field in ['published_parsed', 'updated_parsed', 'created_parsed']:
                    if hasattr(entry, date_field) and getattr(entry, date_field):
                        try:
                            pub_date = datetime(*getattr(entry, date_field)[:6])
                            break
                        except (TypeError, ValueError):
                            continue

                if pub_date is None:
                    pub_date = datetime.now()

                if pub_date >= cutoff_date:
                    recent_count += 1

                    abstract = ""
                    if hasattr(entry, 'summary'):
                        abstract = entry.summary
                    elif hasattr(entry, 'description'):
                        abstract = entry.description

                    abstract = re.sub(r'<[^>]+>', '', abstract)
                    title = entry.get('title', 'No title')

                    score, matched = calculate_relevance(title, abstract)

                    paper = {
                        'title': title,
                        'journal': journal_name,
                        'score': score,
                        'matched': matched,
                        'link': entry.get('link', ''),
                    }
                    all_papers.append(paper)

                    if score >= MIN_RELEVANCE_SCORE:
                        relevant_papers.append(paper)

            print(f"  Found {len(feed.entries)} total entries, {recent_count} recent")

        except Exception as e:
            print(f"  ERROR: {e}")

        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total recent papers found: {len(all_papers)}")
    print(f"Relevant papers (score >= {MIN_RELEVANCE_SCORE}): {len(relevant_papers)}")
    print()

    if relevant_papers:
        print("RELEVANT PAPERS FOUND:")
        print("-" * 70)
        # Sort by score
        relevant_papers.sort(key=lambda x: x['score'], reverse=True)

        for i, paper in enumerate(relevant_papers[:20], 1):
            print(f"\n{i}. [{paper['score']}] {paper['title'][:70]}")
            print(f"   Journal: {paper['journal']}")
            print(f"   Matched: {', '.join(paper['matched'][:5])}")
            print(f"   Link: {paper['link'][:70]}")
    else:
        print("No relevant papers found in the time window.")
        print("Try increasing DAYS_TO_CHECK in config.py or lowering MIN_RELEVANCE_SCORE")

    print()
    print("=" * 70)
    print("To run the full tracker with email summaries:")
    print("1. Add your credentials to config.py")
    print("2. Run: ./venv/bin/python paper_tracker.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
