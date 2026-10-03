"""Bounded OpenAlex requests. Credentials never appear in URLs or errors."""

import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests

from config import OPENALEX_API_KEY


MAX_ATTEMPTS = 5
MAX_RETRY_DELAY = 60
REQUEST_TIMEOUT = 30
FETCH_BUDGET_SECONDS = 300


def get_json(url, params=None, deadline=None):
    if deadline is None:
        deadline = time.monotonic() + FETCH_BUDGET_SECONDS
    headers = {"User-Agent": "NeuroTracker/1.0 (academic research tool)"}
    if OPENALEX_API_KEY:
        headers["Authorization"] = f"Bearer {OPENALEX_API_KEY}"

    for attempt in range(MAX_ATTEMPTS):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("OpenAlex retrieval exceeded its time budget")
        retry_after = None
        try:
            response = requests.get(
                url, params=params, headers=headers,
                timeout=min(REQUEST_TIMEOUT, remaining),
            )
        except requests.exceptions.RequestException as error:
            reason = f"network failure ({type(error).__name__})"
        else:
            try:
                data = response.json()
            except ValueError:
                data = None
            if response.status_code == 200:
                if not isinstance(data, dict):
                    raise RuntimeError("OpenAlex returned invalid JSON")
                return data

            message = " ".join(str(data.get(k, "")) for k in ("error", "message")) if isinstance(data, dict) else ""
            if "plan upgrade required" in message.lower() or "requires a premium" in message.lower():
                raise RuntimeError(
                    "OpenAlex query requires a paid plan; retrying cannot unlock it. "
                    "Use publication-date filtering with a free API key."
                )
            if response.status_code == 429 and (
                any(term in message.lower() for term in ("daily", "budget", "credits", "quota"))
                or response.headers.get("X-RateLimit-Remaining") == "0"
            ):
                raise RuntimeError("OpenAlex daily budget exhausted; retry after reset or configure OPENALEX_API_KEY")
            if response.status_code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"OpenAlex HTTP {response.status_code}; retrieval aborted")
            reason = f"HTTP {response.status_code}"
            value = response.headers.get("Retry-After")
            if value:
                try:
                    retry_after = max(0, float(value))
                except ValueError:
                    try:
                        when = parsedate_to_datetime(value)
                        if when.tzinfo is None:
                            when = when.replace(tzinfo=timezone.utc)
                        retry_after = max(0, (when - datetime.now(timezone.utc)).total_seconds())
                    except (TypeError, ValueError, OverflowError):
                        pass

        if attempt == MAX_ATTEMPTS - 1:
            raise RuntimeError(f"OpenAlex {reason}; exhausted {MAX_ATTEMPTS} attempts")
        delay = max(2 ** attempt, retry_after or 0)
        if delay > MAX_RETRY_DELAY or delay >= deadline - time.monotonic():
            raise RuntimeError(f"OpenAlex {reason}; retry delay exceeds the retrieval budget")
        print(f"  OpenAlex {reason}; retry {attempt + 1}/{MAX_ATTEMPTS - 1} in {delay:g}s")
        time.sleep(delay)

