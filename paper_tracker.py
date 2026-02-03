#!/usr/bin/env python3
"""
Neuroscience Paper Tracker

Checks top neuroscience journals for new publications relevant to your research,
uses a hybrid keyword + LLM scoring system, summarizes relevant papers,
and sends you a daily email digest.
"""

import feedparser
import smtplib
import ssl
import json
import re
import hashlib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from pathlib import Path
import time
from urllib.parse import quote
from html import escape as html_escape
import argparse

from openai import OpenAI

from config import (
    GMAIL_ADDRESS, GMAIL_APP_PASSWORD, RECIPIENT_EMAIL,
    OPENAI_API_KEY, JOURNAL_FEEDS, KEYWORDS,
    MIN_KEYWORD_SCORE, MIN_COMBINED_SCORE, DAYS_TO_CHECK, MAX_PAPERS_PER_DIGEST
)

# File to track which papers we've already processed
SEEN_PAPERS_FILE = Path(__file__).parent / "seen_papers.json"

# Research profile for LLM scoring
RESEARCH_PROFILE = """PhD student investigating what hippocampal firing actually represents.

CORE RESEARCH QUESTION: Does hippocampal firing represent sensory modalities (space, time, sound),
or does it reflect more general computations like action plans, task progression, or internally
generated sequences toward goals?

KEY INTERESTS:
1. Hippocampal representations - place cells, time cells, "tone cells" - and whether these labels
   reflect unique computations or are confounded by task structure
2. Multisensory integration - how hippocampus combines auditory and visual cues to infer task state
3. The hypothesis that hippocampus encodes task-relevant variables and action plans, not raw
   sensory input (Buzsaki's "internally generated sequences" framework)
4. Remapping - what causes it? Sensory changes or changes in task structure/goals?
5. Goal coding, prospective coding, belief states in hippocampus
6. General theories: cognitive maps, relational memory, successor representations, sequence generation

LESS INTERESTED IN:
- Pure technique papers (imaging methods, probes, etc.) unless studying hippocampal cognition
- Hippocampal papers focused only on molecular mechanisms, disease, or development
- Papers about other brain regions unless directly relevant to hippocampal function or multisensory integration"""


def _require_setting(name: str, value: str):
    if not value:
        raise RuntimeError(
            f"Missing required setting: {name}. "
            "Set it as an environment variable or in config_local.py."
        )


def fetch_abstract_from_semantic_scholar_by_doi(doi: str) -> str:
    """Fetch abstract from Semantic Scholar using DOI (more reliable than title search)."""
    if not doi:
        return ""
    try:
        # Semantic Scholar accepts DOI directly
        url = f'https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=abstract,title'
        headers = {'User-Agent': 'NeuroTracker/1.0 (academic research tool)'}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 429:
            time.sleep(2)
            resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            abstract = data.get('abstract', '')
            if abstract:
                return abstract
    except Exception:
        pass
    return ""


def fetch_abstract_from_semantic_scholar(title: str) -> str:
    """Fetch abstract from Semantic Scholar API via title search (fallback)."""
    try:
        encoded_title = quote(title)
        url = f'https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_title}&fields=abstract,title&limit=1'
        headers = {'User-Agent': 'NeuroTracker/1.0 (academic research tool)'}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 429:
            time.sleep(2)
            resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('data') and len(data['data']) > 0:
                abstract = data['data'][0].get('abstract', '')
                if abstract:
                    return abstract
    except Exception:
        pass
    return ""


def fetch_doi_from_crossref(title: str) -> str:
    """Get DOI from CrossRef by title search."""
    try:
        url = "https://api.crossref.org/works"
        params = {"query.title": title, "rows": 1}
        headers = {"User-Agent": "NeuroTracker/1.0 (academic research tool)"}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("message", {}).get("items", [])
            if items:
                return items[0].get("DOI", "")
    except Exception:
        pass
    return ""


def fetch_abstract_from_openalex(doi: str) -> str:
    """Fetch abstract from OpenAlex using DOI."""
    if not doi:
        return ""
    try:
        doi_url = f"https://doi.org/{doi}" if not doi.startswith("http") else doi
        url = f"https://api.openalex.org/works/{quote(doi_url, safe=':/')}"
        headers = {"User-Agent": "NeuroTracker/1.0 (academic research tool)"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            # OpenAlex uses inverted index for abstracts
            inv = data.get("abstract_inverted_index")
            if isinstance(inv, dict) and inv:
                positions = {}
                for word, idxs in inv.items():
                    for i in idxs:
                        positions[i] = word
                if positions:
                    return " ".join(positions[i] for i in sorted(positions))
    except Exception:
        pass
    return ""


def fetch_abstract_robust(title: str) -> str:
    """Try multiple sources to get an abstract: CrossRef→OpenAlex→SemanticScholar(DOI)→SemanticScholar(title)."""
    # Step 1: Get DOI from CrossRef
    doi = fetch_doi_from_crossref(title)

    if doi:
        # Step 2: Try OpenAlex with DOI
        abstract = fetch_abstract_from_openalex(doi)
        if abstract and len(abstract) > 100:
            return abstract

        # Step 3: Try Semantic Scholar with DOI
        abstract = fetch_abstract_from_semantic_scholar_by_doi(doi)
        if abstract and len(abstract) > 100:
            return abstract

    # Step 4: Fallback to Semantic Scholar title search
    abstract = fetch_abstract_from_semantic_scholar(title)
    if abstract and len(abstract) > 100:
        return abstract

    return ""


def reconstruct_abstract_from_inverted_index(inv: dict) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not isinstance(inv, dict) or not inv:
        return ""
    positions = {}
    for word, idxs in inv.items():
        for i in idxs:
            positions[i] = word
    if positions:
        return " ".join(positions[i] for i in sorted(positions))
    return ""


def fetch_papers_from_openalex(start_date: str, end_date: str, search_terms: list[str] = None) -> list[dict]:
    """
    Fetch papers from OpenAlex API with proper date range support.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        search_terms: Optional list of search terms to filter by

    Returns:
        List of paper dicts with title, abstract, link, journal, date, authors
    """
    papers = []
    headers = {"User-Agent": "NeuroTracker/1.0 (academic research tool; mailto:quotientscience21@gmail.com)"}

    # OpenAlex concept IDs for neuroscience-related topics
    # C169760540 = Neuroscience (verified from OpenAlex API)
    neuroscience_concept = "C169760540"

    # Core search terms - keep simple for OpenAlex compatibility
    # OpenAlex limits to 10 OR terms max
    # Multi-word phrases and detailed filtering handled by keyword scoring stage
    default_terms = [
        "hippocampus", "hippocampal", "entorhinal", "CA1",
        "navigation", "multisensory", "audiovisual",
        "abstract", "representation", "state"
    ]
    terms = search_terms if search_terms else default_terms

    # OpenAlex search - use simple OR syntax (max 10 terms)
    search_query = " OR ".join(terms[:10])

    base_url = "https://api.openalex.org/works"

    # Pagination - OpenAlex returns max 200 per page
    cursor = "*"
    page_count = 0
    max_pages = 25  # Safety limit (25 * 200 = 5000 papers max)

    print(f"  Searching OpenAlex for neuroscience papers...")
    print(f"  Date range: {start_date} to {end_date}")

    while cursor and page_count < max_pages:
        params = {
            "filter": f"concepts.id:{neuroscience_concept},from_publication_date:{start_date},to_publication_date:{end_date}",
            "search": search_query,
            "select": "id,doi,title,abstract_inverted_index,publication_date,primary_location,authorships",
            "sort": "publication_date:desc",
            "per_page": 200,
            "cursor": cursor,
        }

        try:
            resp = requests.get(base_url, params=params, headers=headers, timeout=30)

            if resp.status_code == 429:
                print("  Rate limited, waiting 5 seconds...")
                time.sleep(5)
                continue

            if resp.status_code != 200:
                print(f"  OpenAlex error: {resp.status_code}")
                break

            data = resp.json()
            results = data.get("results", [])

            if not results:
                break

            for work in results:
                # Extract journal name
                journal = "Unknown"
                primary_loc = work.get("primary_location", {})
                if primary_loc:
                    source = primary_loc.get("source", {})
                    if source:
                        journal = source.get("display_name", "Unknown")

                # Extract authors
                authorships = work.get("authorships", [])
                authors = ", ".join(
                    a.get("author", {}).get("display_name", "")
                    for a in authorships[:5]  # Limit to first 5 authors
                )
                if len(authorships) > 5:
                    authors += " et al."

                # Parse publication date
                pub_date_str = work.get("publication_date", "")
                try:
                    pub_date = datetime.strptime(pub_date_str, "%Y-%m-%d")
                except (ValueError, TypeError):
                    pub_date = datetime.now()

                # Reconstruct abstract from inverted index
                abstract = reconstruct_abstract_from_inverted_index(
                    work.get("abstract_inverted_index", {})
                )

                # Get DOI link or OpenAlex ID link
                doi = work.get("doi", "")
                link = doi if doi else f"https://openalex.org/works/{work.get('id', '').split('/')[-1]}"

                papers.append({
                    "title": work.get("title", "No title"),
                    "link": link,
                    "abstract": abstract,
                    "journal": journal,
                    "date": pub_date,
                    "authors": authors if authors else "Unknown",
                })

            # Get next cursor for pagination
            meta = data.get("meta", {})
            cursor = meta.get("next_cursor")
            page_count += 1

            print(f"  Fetched page {page_count}: {len(results)} papers (total so far: {len(papers)})")

            # Small delay to be nice to the API
            time.sleep(0.2)

        except requests.exceptions.Timeout:
            print("  Request timed out, retrying...")
            time.sleep(2)
            continue
        except Exception as e:
            print(f"  Error fetching from OpenAlex: {e}")
            break

    print(f"  Total papers from OpenAlex: {len(papers)}")
    return papers


def load_seen_papers() -> set:
    """Load the set of paper IDs we've already processed."""
    if SEEN_PAPERS_FILE.exists():
        with open(SEEN_PAPERS_FILE, "r") as f:
            data = json.load(f)
            cutoff = (datetime.now() - timedelta(days=30)).isoformat()
            return {k for k, v in data.items() if v > cutoff}
    return set()


def save_seen_papers(seen: set):
    """Save the set of processed paper IDs with timestamps."""
    existing = {}
    if SEEN_PAPERS_FILE.exists():
        with open(SEEN_PAPERS_FILE, "r") as f:
            existing = json.load(f)

    now = datetime.now().isoformat()
    for paper_id in seen:
        if paper_id not in existing:
            existing[paper_id] = now

    with open(SEEN_PAPERS_FILE, "w") as f:
        json.dump(existing, f, indent=2)


def get_paper_id(entry: dict) -> str:
    """Generate a unique ID for a paper based on its title and link."""
    text = f"{entry.get('title', '')}{entry.get('link', '')}"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _normalize_title(title: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", title.lower())
    return re.sub(r"\s+", " ", text).strip()


def _title_key(entry: dict) -> str:
    title_key = _normalize_title(entry.get("title", ""))
    if not title_key:
        return ""
    year = ""
    if isinstance(entry.get("date"), datetime):
        year = str(entry["date"].year)
    return f"{title_key}|{year}" if year else title_key


def get_title_id(entry: dict) -> str:
    """Secondary ID to reduce duplicates across feeds/runs."""
    key = _title_key(entry)
    if not key:
        return ""
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _dedupe_papers_by_title(papers: list[dict]) -> list[dict]:
    deduped = {}
    for paper in papers:
        key = _title_key(paper)
        if not key:
            continue
        existing = deduped.get(key)
        if not existing:
            deduped[key] = paper
            continue

        # Prefer richer metadata
        existing_abs = existing.get("abstract", "")
        new_abs = paper.get("abstract", "")
        if len(new_abs) > len(existing_abs):
            deduped[key] = paper
            continue
        if existing.get("journal", "Unknown") == "Unknown" and paper.get("journal", "Unknown") != "Unknown":
            deduped[key] = paper
            continue
        if existing.get("authors", "Unknown") == "Unknown" and paper.get("authors", "Unknown") != "Unknown":
            deduped[key] = paper

    return list(deduped.values())


def calculate_keyword_score(title: str, abstract: str) -> tuple[int, list[str]]:
    """
    Calculate a keyword-based relevance score.
    Returns (raw_score, matched_terms).
    """
    text = f"{title} {abstract}".lower()
    matched_terms = []
    score = 0

    weights = {
        "hippocampus": 10,
        "task_and_action": 15,  # Core to thesis
        "task_space": 6,        # Reduced - these are more generic terms
        "entorhinal_grid": 9,   # MEC/LEC/grid-code language
        "multisensory": 12,
        "auditory": 8,          # Auditory terms - boosted when combined with hippocampus
        "theory": 8,
        "spatial": 5,           # Reduced - very common in neuro
        "researchers": 3,       # Reduced - author names alone shouldn't drive inclusion
    }

    categories_matched = set()
    for category, terms in KEYWORDS.items():
        weight = weights.get(category, 5)
        for term in terms:
            if term.lower() in text:
                if term.lower() not in matched_terms:
                    categories_matched.add(category)
                    if term.lower() in title.lower():
                        score += weight * 2
                        matched_terms.append(f"{term} (title)")
                    else:
                        score += weight
                        matched_terms.append(term)

    # BONUS: Hippocampus + auditory combination is highly relevant to your research
    if "hippocampus" in categories_matched and "auditory" in categories_matched:
        score += 20  # Big bonus for this combination
        matched_terms.append("BONUS: hippo+auditory")

    # Remove duplicates
    seen = set()
    unique_terms = []
    for term in matched_terms:
        base_term = term.replace(" (title)", "")
        if base_term not in seen:
            seen.add(base_term)
            unique_terms.append(term)

    return score, unique_terms


def normalize_keyword_score(raw_score: int, max_expected: int = 100) -> int:
    """Normalize raw keyword score to 0-100 scale."""
    normalized = min(100, int((raw_score / max_expected) * 100))
    return normalized


def get_llm_relevance_score(client: OpenAI, paper: dict) -> tuple[int, str]:
    """
    Use GPT to score paper relevance 0-100.
    Returns (score, brief_reason).
    """
    prompt = f"""Rate this paper's relevance (0-100) for a PhD student studying:

1. HIPPOCAMPAL / MTL REPRESENTATIONS: What does hippocampal/entorhinal/MTL activity represent - sensory features, or internal computations like task state, action plans, and goal-directed sequences?

2. MULTISENSORY INTEGRATION: How does the brain (especially hippocampus/MTL) combine sensory cues to infer task state?

3. STATE INFERENCE & BELIEF UPDATING: How does the brain infer hidden task states from sensory evidence?

Paper title: {paper['title']}

Abstract: {paper['abstract'][:1500] if paper.get('abstract') else '[No abstract available - score based on title only]'}

Respond with ONLY a JSON object, no markdown:
{{"score": <0-100>, "reason": "<one specific sentence - what does this paper ACTUALLY show? Do not exaggerate or stretch relevance.>"}}

BE STRICT AND HONEST:
- 85-100: Directly about hippocampal/MTL representations OR hippocampal/MTL multisensory integration
- 70-84: Hippocampal/entorhinal sequences, replay, remapping, or task-state/goal coding; OR strong state-inference/belief-updating work in decision-making contexts (even if not hippocampus) with clear neural/cognitive relevance
- 55-69: Navigation/spatial coding/place/grid/time-cell work in hippocampus/entorhinal; OR task-state/latent-state representation in other regions
- 40-54: General hippocampus/MTL cognition or navigation studies; or tangential cognitive neuroscience
- 0-39: Not relevant - different topic, clinical/disease focus, technique-only, or only superficially related

PENALIZE (score 0-30):
- Molecular/cellular mechanism papers (synaptic plasticity, LTP/LTD, receptor subtypes, gene expression, protein signaling)
- Disease/clinical papers (Alzheimer's, epilepsy, psychiatric disorders)
- Developmental neuroscience
- Papers focused purely on anatomy or connectivity without functional/cognitive relevance

DO NOT stretch or exaggerate relevance. If a paper is only tangentially related, score it low."""

    try:
        response = client.chat.completions.create(
            model="gpt-5.1",
            temperature=0,
            max_completion_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )

        result_text = response.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        result_text = result_text.replace("```json", "").replace("```", "").strip()

        # Try parsing JSON, with regex fallback for malformed responses
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            # Fallback: extract first JSON object with regex
            match = re.search(r'\{[^}]+\}', result_text)
            if match:
                result = json.loads(match.group())
            else:
                return 0, "Failed to parse response"

        return int(result.get("score", 0)), result.get("reason", "")
    except Exception as e:
        print(f"    LLM scoring error: {e}")
        return 0, "Error during scoring"


def summarize_paper(client: OpenAI, paper: dict) -> str:
    """Use GPT to generate a research-focused summary."""

    if not paper.get('abstract') or len(paper.get('abstract', '')) < 50:
        return "[No abstract available - visit link to read paper]"

    prompt = f"""Summarize this paper accurately in 2-3 sentences based ONLY on what the abstract actually says.
Do NOT speculate, extrapolate, or add interpretations that aren't directly stated.
Do NOT try to force connections to any particular research area.
Just give a faithful, accurate summary of the paper's methods and findings.

Title: {paper['title']}
Abstract: {paper['abstract'][:2000]}

Summary:"""

    try:
        response = client.chat.completions.create(
            model="gpt-5.1",
            max_completion_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Summary unavailable: {e}]"


def fetch_papers_from_feed(journal_name: str, feed_url: str, cutoff_date: datetime) -> list[dict]:
    """Fetch recent papers from a journal's RSS feed."""
    papers = []

    try:
        feed = feedparser.parse(feed_url)

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

            if pub_date < cutoff_date:
                continue

            abstract = ""
            if hasattr(entry, 'summary'):
                abstract = entry.summary
            elif hasattr(entry, 'description'):
                abstract = entry.description
            elif hasattr(entry, 'content') and entry.content:
                abstract = entry.content[0].get('value', '')

            abstract = re.sub(r'<[^>]+>', '', abstract)
            abstract = re.sub(r'\s+', ' ', abstract).strip()

            # Check if abstract is actually just metadata (common with ScienceDirect feeds)
            if abstract and ('Publication date:' in abstract or 'Source:' in abstract or len(abstract) < 100):
                abstract = ""  # Will try to fetch from Semantic Scholar later

            papers.append({
                'title': entry.get('title', 'No title'),
                'link': entry.get('link', ''),
                'abstract': abstract,
                'journal': journal_name,
                'date': pub_date,
                'authors': entry.get('author', entry.get('authors', 'Unknown')),
            })

    except Exception as e:
        print(f"  Error fetching {journal_name}: {e}")

    return papers


def format_email_html(papers: list[dict]) -> str:
    """Format the papers into an HTML email with scores."""
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            .paper {{ background: #f8f9fa; border-left: 4px solid #3498db; padding: 15px; margin: 20px 0; border-radius: 0 8px 8px 0; }}
            .paper.high-relevance {{ border-left-color: #27ae60; }}
            .paper.medium-relevance {{ border-left-color: #f39c12; }}
            .paper h3 {{ margin: 0 0 10px 0; color: #2c3e50; }}
            .paper h3 a {{ color: #3498db; text-decoration: none; }}
            .paper h3 a:hover {{ text-decoration: underline; }}
            .score {{ display: inline-block; background: #2c3e50; color: white; padding: 3px 10px; border-radius: 12px; font-weight: bold; font-size: 0.9em; margin-right: 10px; }}
            .score.high {{ background: #27ae60; }}
            .score.medium {{ background: #f39c12; }}
            .meta {{ font-size: 0.9em; color: #666; margin-bottom: 10px; }}
            .scores-breakdown {{ background: #e8f4f8; padding: 8px 12px; border-radius: 4px; font-size: 0.85em; margin-bottom: 10px; }}
            .llm-reason {{ color: #555; font-style: italic; }}
            .summary {{ color: #444; margin-top: 10px; }}
            .keywords {{ color: #666; font-size: 0.85em; margin-top: 5px; }}
            .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 0.85em; color: #666; }}
        </style>
    </head>
    <body>
        <h1>Neuroscience Paper Digest</h1>
        <p>{len(papers)} relevant paper{'s' if len(papers) != 1 else ''} found on {datetime.now().strftime('%B %d, %Y')}</p>
    """

    # Already sorted by combined score
    for paper in papers:
        combined = paper.get('combined_score', 0)
        score_class = "high" if combined >= 70 else "medium" if combined >= 50 else ""
        relevance_class = "high-relevance" if combined >= 70 else "medium-relevance" if combined >= 50 else ""

        keyword_score = paper.get('keyword_score', 0)
        llm_score = paper.get('llm_score', 0)
        llm_reason = html_escape(paper.get('llm_reason', ''))
        matched = html_escape(', '.join(paper.get('matched_terms', [])[:6]))
        title = html_escape(paper['title'])
        journal = html_escape(paper['journal'])
        summary = html_escape(paper.get('summary', '[No summary]'))

        html += f"""
        <div class="paper {relevance_class}">
            <h3>
                <span class="score {score_class}">{combined}/100</span>
                <a href="{paper['link']}">{title}</a>
            </h3>
            <div class="meta">
                <strong>{journal}</strong> | {paper['date'].strftime('%Y-%m-%d')}
            </div>
            <div class="scores-breakdown">
                <strong>Score:</strong> {combined}/100 (Keyword: {keyword_score} | AI: {llm_score})<br>
                <strong>Why it's relevant:</strong> <span class="llm-reason">{llm_reason}</span>
            </div>
            <div class="summary"><strong>What they found:</strong> {summary}</div>
            <div class="keywords"><strong>Matched:</strong> {matched}</div>
        </div>
        """

    html += """
        <div class="footer">
            <p>Generated by Neuroscience Paper Tracker<br>
            Scores: Keyword matching (0-100) averaged with GPT-5.1 relevance rating (0-100)</p>
        </div>
    </body>
    </html>
    """
    return html


def format_email_text(papers: list[dict]) -> str:
    """Format the papers into plain text email."""
    text = f"NEUROSCIENCE PAPER DIGEST\n"
    text += f"{len(papers)} relevant paper(s) found on {datetime.now().strftime('%B %d, %Y')}\n"
    text += "=" * 60 + "\n\n"

    for i, paper in enumerate(papers, 1):
        combined = paper.get('combined_score', 0)
        keyword_score = paper.get('keyword_score', 0)
        llm_score = paper.get('llm_score', 0)
        llm_reason = paper.get('llm_reason', '')
        matched = ', '.join(paper.get('matched_terms', [])[:6])

        text += f"{i}. [{combined}/100] {paper['title']}\n"
        text += f"   Journal: {paper['journal']} | Date: {paper['date'].strftime('%Y-%m-%d')}\n"
        text += f"   Score: {combined}/100 (Keyword: {keyword_score} | AI: {llm_score})\n"
        text += f"   Why it's relevant: {llm_reason}\n"
        text += f"   What they found: {paper.get('summary', paper['abstract'][:300] + '...')}\n"
        text += f"   Link: {paper['link']}\n"
        text += "\n" + "-" * 60 + "\n\n"

    return text


def send_email(subject: str, html_content: str, text_content: str):
    """Send the digest email via Gmail."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = RECIPIENT_EMAIL

    msg.attach(MIMEText(text_content, "plain"))
    msg.attach(MIMEText(html_content, "html"))

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_ADDRESS, RECIPIENT_EMAIL, msg.as_string())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neuroscience Paper Tracker")
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Days to look back (1-90). Defaults to DAYS_TO_CHECK from config.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (earliest) in YYYY-MM-DD format. Overrides --days.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (latest) in YYYY-MM-DD format. Defaults to today.",
    )
    parser.add_argument(
        "--include-seen",
        action="store_true",
        help="Include previously seen papers (ignore seen_papers.json)",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--historical",
        action="store_true",
        help="Use OpenAlex API (default). Best for longer date ranges.",
    )
    mode_group.add_argument(
        "--rss",
        action="store_true",
        help="Use RSS feeds instead of OpenAlex (better for very recent items).",
    )
    parser.add_argument(
        "--max-llm-candidates",
        type=int,
        default=None,
        help="Max number of papers to send for AI scoring (default: 40).",
    )
    return parser.parse_args()


# Anchors that indicate systems/cognitive neuroscience (not just any biology paper).
# Tightened to avoid matching every generic neuro paper.
NEURO_ANCHOR_TERMS = [
    # Brain regions relevant to your work
    "hippocampus", "hippocampal", "hippocampal formation",
    "entorhinal", "subiculum", "subicular", "presubiculum", "parasubiculum",
    "perirhinal", "parahippocampal", "parahippocampus", "medial temporal lobe",
    "ca1", "ca2", "ca3", "ca4", "dentate", "dentate gyrus",
    "prefrontal", "orbitofrontal", "parietal",
    # Neural recording/circuit terms
    "place cell", "grid cell", "time cell", "neuron", "neuronal", "firing",
    "electrophysiology", "spikes", "single-unit", "neuropixels",
    "lfp", "two-photon", "optogenetic", "calcium imaging",
    # Cognition terms
    "navigation", "memory", "decision", "task", "behavior", "behaviour",
]

# Fast negative filter terms to drop obviously irrelevant domains early.
NEGATIVE_DOMAIN_TERMS = [
    "cancer", "tumor", "tumour", "alzheimer", "parkinson", "stroke", "schizophrenia",
    "virus", "viral", "ebola", "hsv", "sars", "covid",
    "quantum", "neptune", "sub-neptune", "dynamo", "megacryst", "bridgmanite", "planet",
    "gibbon", "fungi", "phototrophic", "glycoprotein",
]


def _has_neuro_anchor(title: str, abstract: str) -> bool:
    text = f"{title} {abstract}".lower()
    return any(t in text for t in NEURO_ANCHOR_TERMS)


def _has_negative_domain_terms(title: str, abstract: str) -> bool:
    text = f"{title} {abstract}".lower()
    return any(t in text for t in NEGATIVE_DOMAIN_TERMS)


def main(
    days_override: int = None,
    include_seen: bool = False,
    start_date: str = None,
    end_date: str = None,
    historical: bool = False,
    use_rss: bool = False,
    max_llm_candidates: int = None,
):
    print(f"Neuroscience Paper Tracker - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    use_openalex = historical or not use_rss
    if use_openalex:
        print("MODE: OpenAlex search")
    else:
        print("MODE: RSS feeds")

    # Determine date range
    if start_date:
        # Date range mode
        try:
            cutoff_date = datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            print(f"Error: Invalid start date format '{start_date}'. Use YYYY-MM-DD.")
            return

        if end_date:
            try:
                end_cutoff = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
            except ValueError:
                print(f"Error: Invalid end date format '{end_date}'. Use YYYY-MM-DD.")
                return
        else:
            end_cutoff = None  # No end date filter (up to today)

        if end_cutoff and end_cutoff < cutoff_date:
            print("Error: End date cannot be before start date.")
            return

        date_range_str = f"{cutoff_date.strftime('%Y-%m-%d')} to {end_cutoff.strftime('%Y-%m-%d') if end_cutoff else 'today'}"
        print(f"Checking papers from {date_range_str}")
    else:
        # Days mode (original behavior)
        days_to_check = days_override if days_override else DAYS_TO_CHECK
        # Cap for safety
        if days_to_check < 1:
            days_to_check = 1
        if days_to_check > 90:
            days_to_check = 90

        cutoff_date = datetime.now() - timedelta(days=days_to_check)
        end_cutoff = None
        print(f"Checking papers from {cutoff_date.strftime('%Y-%m-%d')} onwards ({days_to_check} days)")

    _require_setting("OPENAI_API_KEY", OPENAI_API_KEY)
    _require_setting("GMAIL_ADDRESS", GMAIL_ADDRESS)
    _require_setting("GMAIL_APP_PASSWORD", GMAIL_APP_PASSWORD)
    _require_setting("RECIPIENT_EMAIL", RECIPIENT_EMAIL)

    client = OpenAI(api_key=OPENAI_API_KEY)
    seen_papers = load_seen_papers() if not include_seen else set()
    if include_seen:
        print("Including ALL papers (ignoring previously seen)")
    print()

    # Stage 1: Fetch all papers
    all_papers = []

    if use_openalex:
        # Use OpenAlex API for historical search
        start_str = cutoff_date.strftime("%Y-%m-%d")
        end_str = end_cutoff.strftime("%Y-%m-%d") if end_cutoff else datetime.now().strftime("%Y-%m-%d")
        all_papers = fetch_papers_from_openalex(start_str, end_str)
    else:
        # Use RSS feeds (original behavior)
        for journal_name, feed_url in JOURNAL_FEEDS.items():
            print(f"Fetching: {journal_name}...")
            papers = fetch_papers_from_feed(journal_name, feed_url, cutoff_date)
            print(f"  Found {len(papers)} recent papers")
            all_papers.extend(papers)
            time.sleep(0.3)

        # Filter by end date if specified (RSS mode only - OpenAlex already filters)
        if end_cutoff:
            all_papers = [p for p in all_papers if p['date'] <= end_cutoff]

    print(f"\nTotal papers fetched: {len(all_papers)}")

    # Dedupe by normalized title (+ year) to avoid duplicates across feeds
    deduped_papers = _dedupe_papers_by_title(all_papers)
    if len(deduped_papers) != len(all_papers):
        print(f"Deduplicated by title: {len(all_papers)} -> {len(deduped_papers)}")
    all_papers = deduped_papers

    # Filter out seen papers (do not mark as seen yet)
    new_papers = []
    for paper in all_papers:
        paper_id = get_paper_id(paper)
        title_id = get_title_id(paper)
        if paper_id not in seen_papers and (not title_id or title_id not in seen_papers):
            new_papers.append(paper)

    print(f"New papers (not seen before): {len(new_papers)}")

    # Stage 2: Keyword scoring (broad filter)
    print(f"\nStage 1: Keyword scoring...")
    keyword_candidates = []
    for paper in new_papers:
        # Hard exclude obvious non-neuro domains before any scoring.
        if _has_negative_domain_terms(paper.get('title', ''), paper.get('abstract', '')):
            continue

        raw_score, matched = calculate_keyword_score(paper['title'], paper['abstract'])

        # Require at least one neuroscience anchor term to avoid generic matches.
        if raw_score >= MIN_KEYWORD_SCORE and _has_neuro_anchor(paper['title'], paper.get('abstract', '')):
            paper['keyword_score_raw'] = raw_score
            paper['keyword_score'] = normalize_keyword_score(raw_score)
            paper['matched_terms'] = matched
            keyword_candidates.append(paper)
            # Debug: show why each paper passed
            print(f"  PASS (raw={raw_score}): {paper['title'][:60]}...")
            print(f"       Matched: {', '.join(matched[:5])}")

    print(f"\nPapers passing keyword filter (>= {MIN_KEYWORD_SCORE}) + neuro-anchor: {len(keyword_candidates)}")

    # Limit to top N by keyword score to avoid excessive LLM API usage
    llm_cap = max_llm_candidates if max_llm_candidates is not None else 40
    if llm_cap < 1:
        llm_cap = 1
    if len(keyword_candidates) > llm_cap:
        keyword_candidates.sort(key=lambda x: x['keyword_score_raw'], reverse=True)
        keyword_candidates = keyword_candidates[:llm_cap]
        print(f"Limited to top {llm_cap} by keyword score for LLM scoring")

    # Fetch missing abstracts from Semantic Scholar (only for papers that passed keyword filter)
    papers_needing_abstract = [p for p in keyword_candidates if not p.get('abstract')]
    if papers_needing_abstract:
        print(f"\nFetching {len(papers_needing_abstract)} missing abstracts (CrossRef→OpenAlex→SemanticScholar)...")
        for paper in papers_needing_abstract:
            abstract = fetch_abstract_robust(paper['title'])
            if abstract:
                paper['abstract'] = abstract
                print(f"  Found: {paper['title'][:50]}...")
            else:
                print(f"  Not found: {paper['title'][:50]}...")
            time.sleep(0.4)  # Rate limiting for multiple API calls

    # Stage 3: LLM scoring
    print(f"\nStage 2: AI relevance scoring ({len(keyword_candidates)} papers)...")
    scored_papers = []
    for i, paper in enumerate(keyword_candidates):
        print(f"  Scoring {i+1}/{len(keyword_candidates)}: {paper['title'][:50]}...")
        llm_score, llm_reason = get_llm_relevance_score(client, paper)
        paper['llm_score'] = llm_score
        paper['llm_reason'] = llm_reason

        # Combine Score
        paper['combined_score'] = (paper['keyword_score'] + llm_score) // 2
        scored_papers.append(paper)
        time.sleep(0.2)

    # Stage 4: Filter by combined score
    relevant_papers = [p for p in scored_papers if p['combined_score'] >= MIN_COMBINED_SCORE]
    relevant_papers.sort(key=lambda x: x['combined_score'], reverse=True)

    print(f"\nPapers with combined score >= {MIN_COMBINED_SCORE}: {len(relevant_papers)}")

    # Limit to top N
    if len(relevant_papers) > MAX_PAPERS_PER_DIGEST:
        relevant_papers = relevant_papers[:MAX_PAPERS_PER_DIGEST]
        print(f"Limited to top {MAX_PAPERS_PER_DIGEST}")

    # Mark as seen only after LLM scoring (track both link-based and title-based IDs)
    for paper in scored_papers:
        seen_papers.add(get_paper_id(paper))
        title_id = get_title_id(paper)
        if title_id:
            seen_papers.add(title_id)
    save_seen_papers(seen_papers)

    if not relevant_papers:
        print("\nNo relevant papers found today. No email sent.")
        return

    # Stage 5: Generate summaries with Sonnet
    print(f"\nStage 3: Generating summaries...")
    for i, paper in enumerate(relevant_papers):
        print(f"  Summarizing {i+1}/{len(relevant_papers)}: {paper['title'][:50]}...")
        paper['summary'] = summarize_paper(client, paper)
        time.sleep(0.3)

    # Send email
    print("\nSending email...")
    subject = f"Neuro Papers: {len(relevant_papers)} found - {datetime.now().strftime('%b %d')}"
    html_content = format_email_html(relevant_papers)
    text_content = format_email_text(relevant_papers)

    try:
        send_email(subject, html_content, text_content)
        print(f"Email sent successfully to {RECIPIENT_EMAIL}")
    except Exception as e:
        print(f"Error sending email: {e}")
        backup_file = Path(__file__).parent / f"digest_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(backup_file, "w") as f:
            f.write(text_content)
        print(f"Digest saved to {backup_file}")

    print("\nDone!")


if __name__ == "__main__":
    args = _parse_args()
    main(
        days_override=args.days,
        include_seen=args.include_seen,
        start_date=args.start_date,
        end_date=args.end_date,
        historical=args.historical,
        use_rss=args.rss,
        max_llm_candidates=args.max_llm_candidates,
    )
