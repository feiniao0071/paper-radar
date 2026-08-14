from datetime import UTC, datetime

from paper_radar.feishu import build_card, build_signature
from paper_radar.models import MatchResult, Paper, Recommendation


def test_signature_is_stable() -> None:
    expected = "mbm4Y4oluIPQ00qlBIhX8vAZ0EKv3nw0LuTb91jPL84="
    assert build_signature("test-secret", 1_700_000_000) == expected


def test_card_contains_links_and_priority() -> None:
    now = datetime(2026, 8, 14, tzinfo=UTC)
    paper = Paper(
        paper_id="2608.00001",
        title="Graphene paper",
        authors=("A. Researcher",),
        abstract="Abstract",
        published=now,
        updated=now,
        categories=("cond-mat.mtrl-sci",),
        abstract_url="https://arxiv.org/abs/2608.00001",
        pdf_url="https://arxiv.org/pdf/2608.00001",
    )
    recommendation = Recommendation(
        paper=paper,
        relevance_score=3,
        reason="Directly relevant",
        key_relevance=("graphene",),
        title_zh="",
        summary_zh="",
        used_ai=False,
    )
    card = build_card(recommendation, MatchResult(3, ("graphene",), ()))
    assert card["header"]["template"] == "red"
    buttons = card["elements"][1]["actions"]
    assert buttons[0]["url"] == paper.abstract_url
    assert buttons[1]["url"] == paper.pdf_url
