"""WoC (World of Code) extractor — stub with same interface as seart_ghs.

WoC access requires SSH to da1.eecs.utk.edu and is currently unavailable.
This stub preserves the interface so tests and pipelines remain valid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from extracao.seart_ghs import RepoRecord

DEFAULT_CACHE_DIR = Path("data/raw/woc_cache")


def fetch_repositories(
    min_stars: int = 10,
    language: str | None = None,
    has_readme: bool = True,
    max_repos: int = 50_000,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    page_size: int = 100,
    verify_ssl: bool = True,
) -> Iterator[RepoRecord]:
    """Yield RepoRecord objects from WoC (stub — not yet implemented).

    Args:
        min_stars: Minimum star count filter.
        language: Filter by primary language (None = all).
        has_readme: Only include repos with README.
        max_repos: Hard limit on total repos yielded.
        cache_dir: Directory to store raw data for resumability.
        page_size: Results per page/batch.
        verify_ssl: Set False to skip SSL verification (workaround for corporate proxies on Windows).

    Yields:
        RepoRecord for each matching repository.

    Raises:
        NotImplementedError: Always — WoC access not yet configured.
    """
    raise NotImplementedError(
        "WoC extractor is not yet implemented. "
        "Access to da1.eecs.utk.edu required. "
        "Use extracao.seart_ghs.fetch_repositories() instead."
    )
    yield  # make this a generator to satisfy the return type
