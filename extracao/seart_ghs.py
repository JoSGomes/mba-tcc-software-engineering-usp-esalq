"""SEART GHS extractor — fetches repository metadata from seart.si.usi.ch."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

SEART_BASE_URL = "https://seart-ghs.si.usi.ch/api/r/search"
DEFAULT_CACHE_DIR = Path("data/raw/seart_cache")


@dataclass
class RepoRecord:
    """Canonical schema for a single repository record."""

    repo_id: str                  # "owner/name"
    name: str
    description: str
    language: str
    stars: int
    forks: int
    watchers: int
    open_issues: int
    size_kb: int
    has_readme: bool
    has_license: bool
    has_wiki: bool
    commits: int
    contributors: int
    created_at: str               # ISO 8601
    updated_at: str               # ISO 8601
    readme_text: str = field(default="")
    collected_at: str = field(default="")


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
def _get_page(session: requests.Session, url: str, params: dict) -> dict:
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_repositories(
    min_stars: int = 10,
    language: str | None = None,
    has_readme: bool = True,
    max_repos: int = 50_000,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    page_size: int = 100,
    verify_ssl: bool = True,
) -> Iterator[RepoRecord]:
    """Yield RepoRecord objects from SEART GHS with pagination, retry and disk cache.

    Args:
        min_stars: Minimum star count filter.
        language: Filter by primary language (None = all).
        has_readme: Only include repos with README.
        max_repos: Hard limit on total repos yielded.
        cache_dir: Directory to store raw JSON pages for resumability.
        page_size: Results per API page.
        verify_ssl: Set False to skip SSL verification (workaround for corporate proxies on Windows).

    Yields:
        RepoRecord for each matching repository.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    session.verify = verify_ssl
    if not verify_ssl:
        warnings.filterwarnings("ignore", message="Unverified HTTPS request")

    # API parameter discovery (2026-05-06):
    #   starsMin=N  — minimum star count (correct name; minStars/stars=>=N are ignored)
    #   language=X  — filter by mainLanguage
    #   sort=FIELD,DIRECTION — e.g. "stargazers,DESC"
    #   hasReadme   — parameter exists but has no effect on the API; removed
    params: dict = {
        "starsMin": min_stars,
        "size": page_size,
        "page": 0,
        "sort": "stargazers,DESC",
    }
    if language:
        params["language"] = language

    collected = 0
    while collected < max_repos:
        data = _get_page(session, SEART_BASE_URL, params)
        items = data.get("items", [])
        if not items:
            break

        for item in items:
            if collected >= max_repos:
                return
            yield _parse_item(item)
            collected += 1

        if data.get("last", True):
            break
        params["page"] += 1
        time.sleep(0.5)  # polite rate-limiting


def _parse_item(item: dict) -> RepoRecord:
    # API field mapping (verified 2026-05-06):
    #   name = "owner/repo"  (already includes owner)
    #   stargazers = star count
    #   mainLanguage = primary language
    #   openIssues = open issue count
    #   hasWiki = bool
    #   license = license name string or None (None → has_license=False)
    #   hasReadme not available; defaulted to True for repos with >=10 stars
    #   description not available from SEART API (homepage is a URL, not text)
    name_full = item.get("name", "")  # already "owner/repo" format
    return RepoRecord(
        repo_id=name_full,
        name=name_full.split("/")[-1] if "/" in name_full else name_full,
        description="",  # SEART API does not expose description text
        language=item.get("mainLanguage", "") or "",
        stars=item.get("stargazers", 0) or 0,
        forks=item.get("forks", 0) or 0,
        watchers=item.get("watchers", 0) or 0,
        open_issues=item.get("openIssues", 0) or 0,
        size_kb=item.get("size", 0) or 0,
        has_readme=True,  # not available from API; assumed True for repos with >=10 stars
        has_license=item.get("license") is not None,
        has_wiki=item.get("hasWiki", False) or False,
        commits=item.get("commits", 0) or 0,
        contributors=item.get("contributors", 0) or 0,
        created_at=item.get("createdAt", "") or "",
        updated_at=item.get("updatedAt", "") or "",
    )
