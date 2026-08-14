from datetime import UTC, datetime

from paper_radar.models import Paper
from paper_radar.sources import deduplicate_papers


def _paper(paper_id: str, source: str, *, doi: str = "", abstract: str = "") -> Paper:
    now = datetime(2026, 8, 14, tzinfo=UTC)
    return Paper(
        paper_id=paper_id,
        title="The same graphene result",
        authors=("Author",),
        abstract=abstract,
        published=now,
        updated=now,
        categories=(),
        abstract_url="https://example.test/paper",
        pdf_url="https://example.test/paper.pdf",
        source=source,
        doi=doi,
    )


def test_deduplication_merges_arxiv_and_crossref_metadata() -> None:
    arxiv = _paper("2608.00001", "arXiv", abstract="Detailed abstract")
    crossref = _paper(
        "crossref:10.1234/example",
        "Crossref",
        doi="10.1234/example",
    )

    papers = deduplicate_papers([arxiv, crossref])

    assert len(papers) == 1
    assert papers[0].paper_id == arxiv.paper_id
    assert papers[0].doi == "10.1234/example"
    assert papers[0].source == "arXiv + Crossref"
