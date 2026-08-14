from datetime import UTC, datetime

from paper_radar.config import MatchingConfig
from paper_radar.matcher import match_paper
from paper_radar.models import Paper


def make_paper(title: str, abstract: str) -> Paper:
    now = datetime(2026, 8, 14, tzinfo=UTC)
    return Paper(
        paper_id="2608.00001",
        title=title,
        authors=("A. Researcher",),
        abstract=abstract,
        published=now,
        updated=now,
        categories=("cond-mat.mtrl-sci",),
        abstract_url="https://arxiv.org/abs/2608.00001",
        pdf_url="https://arxiv.org/pdf/2608.00001",
    )


def config() -> MatchingConfig:
    return MatchingConfig(
        require_core_term=True,
        minimum_score=3,
        core_term_weight=3,
        supporting_term_weight=1,
        core_terms=("graphene", "MoS2", "two-dimensional materials"),
        supporting_terms=("DFT", "sensor", "quantum transport"),
        excluded_terms=("2D object detection",),
    )


def test_core_and_supporting_terms_match() -> None:
    paper = make_paper("Quantum transport in graphene", "A DFT study of a graphene device.")
    result = match_paper(paper, config())
    assert result is not None
    assert result.score == 5
    assert "graphene" in result.core_terms
    assert "DFT" in result.supporting_terms


def test_generic_supporting_term_is_not_enough() -> None:
    paper = make_paper("A flexible chemical sensor", "We report a sensitive device.")
    assert match_paper(paper, config()) is None


def test_exclusion_wins() -> None:
    paper = make_paper(
        "Graphene features for 2D object detection",
        "A computer vision benchmark.",
    )
    assert match_paper(paper, config()) is None


def test_hyphenated_two_dimensional_phrase_matches() -> None:
    paper = make_paper(
        "Two-dimensional materials for electronics",
        "Electronic transport is measured.",
    )
    assert match_paper(paper, config()) is not None
