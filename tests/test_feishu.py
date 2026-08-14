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


def test_card_uses_complete_ai_recommendation() -> None:
    now = datetime(2026, 8, 14, tzinfo=UTC)
    paper = Paper(
        paper_id="2608.00002",
        title="Excitons in a two-dimensional semiconductor",
        authors=("A. Researcher",),
        abstract="Original abstract",
        published=now,
        updated=now,
        categories=("cond-mat.mtrl-sci",),
        abstract_url="https://arxiv.org/abs/2608.00002",
        pdf_url="https://arxiv.org/pdf/2608.00002",
    )
    recommendation = Recommendation(
        paper=paper,
        relevance_score=3,
        reason="与二维材料激子研究直接相关。",
        key_relevance=("二维半导体", "激子"),
        title_zh="二维半导体中的激子",
        summary_zh="本文研究单层半导体中的激子和光学性质。",
        used_ai=True,
    )

    card = build_card(
        recommendation,
        MatchResult(6, ("two-dimensional materials",), ("exciton",)),
    )

    assert card["header"]["title"]["content"] == recommendation.title_zh
    content = card["elements"][0]["text"]["content"]
    assert recommendation.summary_zh in content
    assert recommendation.reason in content
    assert "二维半导体, 激子" in content
