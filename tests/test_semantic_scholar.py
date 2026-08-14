from datetime import UTC, datetime

import httpx

from paper_radar.config import SemanticScholarConfig
from paper_radar.models import Paper
from paper_radar.semantic_scholar import enrich_papers


def test_enrichment_adds_quality_metadata() -> None:
    now = datetime(2026, 8, 14, tzinfo=UTC)
    paper = Paper(
        paper_id="2608.00001",
        title="Graphene paper",
        authors=("Author",),
        abstract="Abstract",
        published=now,
        updated=now,
        categories=(),
        abstract_url="https://arxiv.org/abs/2608.00001",
        pdf_url="https://arxiv.org/pdf/2608.00001",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {
                    "externalIds": {"DOI": "10.1234/example"},
                    "venue": "Nature Physics",
                    "citationCount": 4,
                    "influentialCitationCount": 1,
                    "publicationTypes": ["JournalArticle"],
                }
            ],
            request=request,
        )

    config = SemanticScholarConfig(
        enabled=True,
        api_url="https://example.test/graph/v1",
        batch_size=100,
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        enriched = enrich_papers([paper], config, client=client)[0]

    assert enriched.doi == "10.1234/example"
    assert enriched.venue == "Nature Physics"
    assert enriched.citation_count == 4
    assert enriched.influential_citation_count == 1
