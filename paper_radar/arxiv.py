from __future__ import annotations

import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta

import httpx

from paper_radar.config import ArxivConfig
from paper_radar.models import Paper

LOGGER = logging.getLogger(__name__)
ATOM = "http://www.w3.org/2005/Atom"


def build_search_query(terms: tuple[str, ...]) -> str:
    clauses = []
    for term in terms:
        escaped = term.replace('"', "")
        clauses.append(f'all:"{escaped}"')
    return " OR ".join(clauses)


def _batches(terms: tuple[str, ...], batch_size: int) -> list[tuple[str, ...]]:
    return [terms[index : index + batch_size] for index in range(0, len(terms), batch_size)]


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(UTC)


def _text(entry: ET.Element, name: str) -> str:
    value = entry.findtext(f"{{{ATOM}}}{name}", default="")
    return re.sub(r"\s+", " ", value).strip()


def parse_feed(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    papers: list[Paper] = []
    for entry in root.findall(f"{{{ATOM}}}entry"):
        raw_id = _text(entry, "id")
        paper_id = raw_id.rsplit("/abs/", 1)[-1]
        paper_id = re.sub(r"v\d+$", "", paper_id)
        authors = tuple(
            re.sub(r"\s+", " ", author.findtext(f"{{{ATOM}}}name", default="")).strip()
            for author in entry.findall(f"{{{ATOM}}}author")
        )
        categories = tuple(
            category.attrib.get("term", "")
            for category in entry.findall(f"{{{ATOM}}}category")
            if category.attrib.get("term")
        )
        links = entry.findall(f"{{{ATOM}}}link")
        abstract_url = next(
            (
                link.attrib["href"]
                for link in links
                if link.attrib.get("rel") == "alternate" and link.attrib.get("href")
            ),
            raw_id,
        )
        pdf_url = next(
            (
                link.attrib["href"]
                for link in links
                if link.attrib.get("title") == "pdf" and link.attrib.get("href")
            ),
            abstract_url.replace("/abs/", "/pdf/"),
        )
        papers.append(
            Paper(
                paper_id=paper_id,
                title=_text(entry, "title"),
                authors=authors,
                abstract=_text(entry, "summary"),
                published=_parse_datetime(_text(entry, "published")),
                updated=_parse_datetime(_text(entry, "updated")),
                categories=categories,
                abstract_url=abstract_url,
                pdf_url=pdf_url,
            )
        )
    return papers


def fetch_recent_papers(
    config: ArxivConfig,
    *,
    now: datetime | None = None,
    client: httpx.Client | None = None,
) -> list[Paper]:
    current_time = (now or datetime.now(UTC)).astimezone(UTC)
    cutoff = current_time - timedelta(days=config.lookback_days)
    contact = os.getenv("ARXIV_CONTACT", "").strip()
    user_agent = "paper-radar/0.1"
    if contact:
        user_agent += f" ({contact})"

    owns_client = client is None
    http_client = client or httpx.Client(
        timeout=45,
        follow_redirects=True,
        headers={"User-Agent": user_agent},
    )
    try:
        papers_by_id: dict[str, Paper] = {}
        term_batches = _batches(config.query_terms, config.query_batch_size)
        for batch_index, terms in enumerate(term_batches):
            params = {
                "search_query": build_search_query(terms),
                "start": 0,
                "max_results": config.max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            for attempt in range(3):
                try:
                    response = http_client.get(config.api_url, params=params)
                    response.raise_for_status()
                    for paper in parse_feed(response.text):
                        papers_by_id[paper.paper_id] = paper
                    break
                except (httpx.HTTPError, ET.ParseError):
                    if attempt == 2:
                        raise
                    delay = 2**attempt
                    LOGGER.warning("arXiv request failed; retrying in %d second(s)", delay)
                    time.sleep(delay)
            if batch_index < len(term_batches) - 1:
                time.sleep(3)

        papers = sorted(papers_by_id.values(), key=lambda paper: paper.published, reverse=True)
        recent = [paper for paper in papers if paper.published >= cutoff]
        LOGGER.info(
            "Fetched %d unique papers; %d are within the lookback window",
            len(papers),
            len(recent),
        )
        return recent
    finally:
        if owns_client:
            http_client.close()

    return []
