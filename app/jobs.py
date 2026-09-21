"""Fetch real, live job postings from a public job board.

Uses the Remotive API (https://remotive.com/api/remote-jobs) - free, no API key,
returns real remote software/ML roles with a real apply URL per posting. Pure
standard library (urllib) so the app runs with zero extra dependencies.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request

REMOTIVE_URL = "https://remotive.com/api/remote-jobs"
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def _strip_html(text):
    """Turn an HTML job description into plain text for scoring/preview."""
    text = _TAG.sub(" ", text or "")
    text = (text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                .replace("&nbsp;", " ").replace("&#39;", "'").replace("&quot;", '"'))
    return _WS.sub(" ", text).strip()


def fetch_jobs(query="machine learning", limit=15, timeout=12):
    """Return a list of live job dicts. Never raises: on failure returns []."""
    params = {"search": query, "limit": max(1, min(limit, 100))}
    url = REMOTIVE_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "jobfit-agent/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as exc:  # network, JSON, timeout - stay graceful for the demo
        raise JobFetchError(str(exc))

    jobs = []
    for j in data.get("jobs", [])[:limit]:
        desc = _strip_html(j.get("description", ""))
        jobs.append({
            "id": j.get("id"),
            "title": j.get("title", "").strip(),
            "company": (j.get("company_name") or "").strip(),
            "location": (j.get("candidate_required_location") or "Remote").strip(),
            "category": (j.get("category") or "").strip(),
            "url": j.get("url", ""),
            "published": (j.get("publication_date") or "")[:10],
            "description": desc,
            # a compact text blob to match a resume against
            "match_text": " ".join([j.get("title", ""), j.get("category", ""), desc]),
        })
    return jobs


class JobFetchError(Exception):
    """Raised when the live job feed cannot be reached or parsed."""


import os

_SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_jobs.json")


def sample_jobs():
    """Bundled illustrative roles, shown when the live feed is unreachable."""
    with open(_SAMPLE_PATH, encoding="utf-8") as fh:
        jobs = json.load(fh)
    for j in jobs:
        j["match_text"] = " ".join([j.get("title", ""), j.get("category", ""),
                                     j.get("description", "")])
    return jobs


_GENERIC_QUERY_WORDS = {
    "engineer", "engineering", "developer", "dev", "senior", "junior", "lead",
    "staff", "principal", "remote", "role", "roles", "job", "jobs", "position",
    "specialist", "analyst", "consultant", "manager", "intern", "internship",
    "i", "ii", "iii", "the", "a", "an", "of", "and", "for", "in", "with",
}


def _query_terms(query):
    toks = re.findall(r"[a-z0-9+#.]+", (query or "").lower())
    significant = [t for t in toks if len(t) >= 2 and t not in _GENERIC_QUERY_WORDS]
    return significant, " ".join(toks)


def _word_in(term, haystack):
    return re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])",
                     haystack) is not None


def matches_query(job, query):
    """True if the posting is actually about the searched role.

    Remotive's ?search= often returns the same recent list regardless of query,
    so we filter here: keep a job when the full phrase, or every significant
    query word, appears in its title or description.
    """
    significant, phrase = _query_terms(query)
    if not significant:
        return True  # only generic words (e.g. "engineer") - can't narrow
    hay = (job.get("title", "") + " " + job.get("match_text", "")).lower()
    if phrase and phrase in hay:
        return True
    return all(_word_in(t, hay) for t in significant)
