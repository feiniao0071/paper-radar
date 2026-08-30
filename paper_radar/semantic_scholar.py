from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import httpx

from paper_radar.config import SemanticScholarConfig
from paper_radar.models import Paper

LOGGER = logging.getLogger(__name__)

FIELDS = (
    "paperId,externalIds,title,venue,citationCount,influentialCitationCount,"
    "publicationTypes,fieldsOfStudy"
)
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
# Semantic Scholar can temporarily throttle a valid key without Retry-After.
# This gives the shared quota window about one minute to recover.
MAX_ATTEMPTS = 7


def _batches(papers: list[Paper], size: int) -> list[list[Paper]]:
    return [papers[index : index + size] for index in range(0, len(papers), size)]


def _identifier(paper: Paper) -> str:
    if paper.doi:
        return f"DOI:{paper.doi}"
    if paper.source.startswith("arXiv"):
        return f"ARXIV:{paper.paper_id}"
    return ""


def _enrich(paper: Paper, metadata: dict[str, Any]) -> Paper:
    external_ids = metadata.get("externalIds")
    doi = paper.doi
    if not doi and isinstance(external_ids, dict):
        doi = str(external_ids.get("DOI", "")).strip().lower()

    citation_count = metadata.get("citationCount")
    influential_count = metadata.get("influentialCitationCount")
    publication_types = metadata.get("publicationTypes")
    return replace(
        paper,
        doi=doi,
        venue=paper.venue or str(metadata.get("venue", "")).strip(),
        citation_count=(
            max(paper.citation_count or 0, citation_count)
            if isinstance(citation_count, int)
            else paper.citation_count
        ),
        influential_citation_count=(
            influential_count if isinstance(influential_count, int) else None
        ),
        publication_types=(
            tuple(str(item) for item in publication_types if str(item).strip())
            if isinstance(publication_types, list)
            else paper.publication_types
        ),
    )


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After", "").strip()
    if retry_after:
        try:
            return min(max(float(retry_after), 0.0), 60.0)
        except ValueError:
            pass
    return min(2.0**attempt, 30.0)


def _fetch_batch(
    http_client: httpx.Client,
    api_url: str,
    identifiers: list[str],
    headers: dict[str, str],
    sleep: Callable[[float], None],
) -> list[Any]:
    for attempt in range(MAX_ATTEMPTS):
        response = http_client.post(
            f"{api_url.rstrip('/')}/paper/batch",
            params={"fields": FIELDS},
            json={"ids": identifiers},
            headers=headers,
        )
        if response.status_code not in RETRYABLE_STATUS_CODES:
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError("Semantic Scholar returned a non-list response")
            return payload

        if attempt == MAX_ATTEMPTS - 1:
            response.raise_for_status()
        delay = _retry_delay(response, attempt)
        LOGGER.warning(
            "Semantic Scholar request returned %d; retrying in %.1fs (%d/%d)",
            response.status_code,
            delay,
            attempt + 1,
            MAX_ATTEMPTS,
        )
        sleep(delay)

    raise RuntimeError("Semantic Scholar retry loop ended unexpectedly")


def enrich_papers(
    papers: list[Paper],
    config: SemanticScholarConfig,
    *,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Paper]:
    if not config.enabled or not papers:
        return papers

    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    headers = {"User-Agent": "paper-radar/0.2"}
    if api_key:
        headers["x-api-key"] = api_key
    owns_client = client is None
    http_client = client or httpx.Client(timeout=45)
    enriched: list[Paper] = []
    try:
        for batch in _batches(papers, config.batch_size):
            identified = [(paper, _identifier(paper)) for paper in batch]
            request_items = [(paper, identifier) for paper, identifier in identified if identifier]
            metadata_by_id: dict[str, dict[str, Any]] = {}
            if request_items:
                payload = _fetch_batch(
                    http_client,
                    config.api_url,
                    [identifier for _, identifier in request_items],
                    headers,
                    sleep,
                )
                for (_, identifier), item in zip(request_items, payload, strict=False):
                    if isinstance(item, dict):
                        metadata_by_id[identifier] = item
            for paper, identifier in identified:
                metadata = metadata_by_id.get(identifier)
                enriched.append(_enrich(paper, metadata) if metadata else paper)
        return enriched
    finally:
        if owns_client:
            http_client.close()
