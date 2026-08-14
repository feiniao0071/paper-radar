from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

import httpx

from paper_radar.config import SemanticScholarConfig
from paper_radar.models import Paper

FIELDS = (
    "paperId,externalIds,title,venue,citationCount,influentialCitationCount,"
    "publicationTypes,fieldsOfStudy"
)


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


def enrich_papers(
    papers: list[Paper],
    config: SemanticScholarConfig,
    *,
    client: httpx.Client | None = None,
) -> list[Paper]:
    if not config.enabled or not papers:
        return papers

    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    headers = {"User-Agent": "paper-radar/0.2"}
    if api_key:
        headers["x-api-key"] = api_key
    owns_client = client is None
    http_client = client or httpx.Client(timeout=45, headers=headers)
    enriched: list[Paper] = []
    try:
        for batch in _batches(papers, config.batch_size):
            identified = [(paper, _identifier(paper)) for paper in batch]
            request_items = [(paper, identifier) for paper, identifier in identified if identifier]
            metadata_by_id: dict[str, dict[str, Any]] = {}
            if request_items:
                response = http_client.post(
                    f"{config.api_url.rstrip('/')}/paper/batch",
                    params={"fields": FIELDS},
                    json={"ids": [identifier for _, identifier in request_items]},
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list):
                    raise ValueError("Semantic Scholar returned a non-list response")
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
