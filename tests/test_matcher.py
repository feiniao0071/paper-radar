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
        minimum_score=5,
        core_term_weight=3,
        supporting_term_weight=1,
        core_terms=("graphene", "MoS2", "two-dimensional materials"),
        supporting_terms=("DFT", "sensor"),
        excluded_terms=("2D object detection", "machine learning", "gas sensor"),
        require_focus_term=True,
        focus_term_weight=2,
        focus_terms=("quantum transport", "moire", "superconductivity"),
    )


def test_core_and_supporting_terms_match() -> None:
    paper = make_paper("Quantum transport in graphene", "A DFT study of a graphene device.")
    result = match_paper(paper, config())
    assert result is not None
    assert result.score == 6
    assert "graphene" in result.core_terms
    assert "quantum transport" in result.focus_terms
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
        "Two-dimensional materials for quantum electronics",
        "Quantum transport is measured.",
    )
    assert match_paper(paper, config()) is not None


def test_generic_2d_application_is_rejected_without_quantum_focus() -> None:
    paper = make_paper(
        "Graphene field-effect transistor",
        "We optimize room-temperature mobility for a flexible device.",
    )
    assert match_paper(paper, config()) is None


def test_quantum_ai_materials_are_reserved_for_separate_radar() -> None:
    paper = make_paper(
        "Machine learning for graphene quantum transport",
        "A neural network predicts correlated transport.",
    )
    assert match_paper(paper, config()) is None


def test_accented_moire_matches_quantum_focus() -> None:
    paper = make_paper(
        "Moiré bands in graphene",
        "We study interaction-driven phases.",
    )
    result = match_paper(paper, config())
    assert result is not None
    assert "moire" in result.focus_terms
