from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from contextlib import suppress
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
# Six attempts with the configured backoff give the shared quota time to recover
# without keeping a scheduled workflow occupied indefinitely.
MAX_ATTEMPTS = 6
MAX_RESPONSE_DETAIL_LENGTH = 300


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


def _retry_delay(
    response: httpx.Response,
    attempt: int,
    *,
    initial_retry_delay_seconds: float,
    max_retry_delay_seconds: float,
) -> float:
    exponential_delay = initial_retry_delay_seconds * (2.0**attempt)
    retry_after = response.headers.get("Retry-After", "").strip()
    if retry_after:
        with suppress(ValueError):
            exponential_delay = max(float(retry_after), exponential_delay)
    return min(exponential_delay, max_retry_delay_seconds)


def _response_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = response.text
    if isinstance(payload, dict):
        detail = payload.get("message") or payload.get("error") or payload
    else:
        detail = payload
    normalized = " ".join(str(detail).split())
    return normalized[:MAX_RESPONSE_DETAIL_LENGTH] or "no response detail"


def describe_error(error: Exception) -> str:
    if isinstance(error, httpx.HTTPStatusError):
        return f"HTTP {error.response.status_code}"
    return type(error).__name__


def _fetch_batch(
    http_client: httpx.Client,
    api_url: str,
    identifiers: list[str],
    headers: dict[str, str],
    initial_retry_delay_seconds: float,
    max_retry_delay_seconds: float,
    sleep: Callable[[float], None],
) -> list[Any]:
    previous_status: int | None = None
    retried_transient_bad_request = False
    for attempt in range(MAX_ATTEMPTS):
        response = http_client.post(
            f"{api_url.rstrip('/')}/paper/batch",
            params={"fields": FIELDS},
            json={"ids": identifiers},
            headers=headers,
        )
        retryable = response.status_code in RETRYABLE_STATUS_CODES
        if (
            response.status_code == 400
            and previous_status in RETRYABLE_STATUS_CODES
            and not retried_transient_bad_request
        ):
            # The API has been observed returning 400 immediately after a 429
            # for the same valid payload. Give that inconsistent response one
            # retry before treating it as a genuine bad request.
            retryable = True
            retried_transient_bad_request = True
        if not retryable:
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError("Semantic Scholar returned a non-list response")
            return payload

        if attempt == MAX_ATTEMPTS - 1:
            LOGGER.error(
                "Semantic Scholar retries exhausted with HTTP %d: %s",
                response.status_code,
                _response_detail(response),
            )
            response.raise_for_status()
        delay = _retry_delay(
            response,
            attempt,
            initial_retry_delay_seconds=initial_retry_delay_seconds,
            max_retry_delay_seconds=max_retry_delay_seconds,
        )
        LOGGER.warning(
            "Semantic Scholar request returned %d: %s; retrying in %.1fs (%d/%d)",
            response.status_code,
            _response_detail(response),
            delay,
            attempt + 1,
            MAX_ATTEMPTS,
        )
        sleep(delay)
        previous_status = response.status_code

    raise RuntimeError("Semantic Scholar retry loop ended unexpectedly")


def _fetch_resilient_batch(
    http_client: httpx.Client,
    api_url: str,
    identifiers: list[str],
    headers: dict[str, str],
    config: SemanticScholarConfig,
    sleep: Callable[[float], None],
) -> list[Any]:
    try:
        return _fetch_batch(
            http_client,
            api_url,
            identifiers,
            headers,
            config.initial_retry_delay_seconds,
            config.max_retry_delay_seconds,
            sleep,
        )
    except httpx.HTTPStatusError as error:
        if error.response.status_code != 400:
            raise
        detail = _response_detail(error.response)
        if len(identifiers) == 1:
            LOGGER.warning(
                "Semantic Scholar rejected identifier %s with HTTP 400: %s; "
                "leaving its optional metadata unchanged",
                identifiers[0],
                detail,
            )
            return [None]

        midpoint = len(identifiers) // 2
        LOGGER.warning(
            "Semantic Scholar rejected a batch of %d identifiers with HTTP 400: %s; "
            "splitting the batch to isolate invalid identifiers",
            len(identifiers),
            detail,
        )
        left = _fetch_resilient_batch(
            http_client,
            api_url,
            identifiers[:midpoint],
            headers,
            config,
            sleep,
        )
        if config.request_interval_seconds > 0:
            sleep(config.request_interval_seconds)
        right = _fetch_resilient_batch(
            http_client,
            api_url,
            identifiers[midpoint:],
            headers,
            config,
            sleep,
        )
        return left + right


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
        batches = _batches(papers, config.batch_size)
        for batch_index, batch in enumerate(batches):
            identified = [(paper, _identifier(paper)) for paper in batch]
            request_items = [(paper, identifier) for paper, identifier in identified if identifier]
            made_request = False
            metadata_by_id: dict[str, dict[str, Any]] = {}
            if request_items:
                made_request = True
                payload = _fetch_resilient_batch(
                    http_client,
                    config.api_url,
                    [identifier for _, identifier in request_items],
                    headers,
                    config,
                    sleep,
                )
                for (_, identifier), item in zip(request_items, payload, strict=False):
                    if isinstance(item, dict):
                        metadata_by_id[identifier] = item
            for paper, identifier in identified:
                metadata = metadata_by_id.get(identifier)
                enriched.append(_enrich(paper, metadata) if metadata else paper)
            if (
                made_request
                and config.request_interval_seconds > 0
                and batch_index < len(batches) - 1
            ):
                sleep(config.request_interval_seconds)
        return enriched
    finally:
        if owns_client:
            http_client.close()
