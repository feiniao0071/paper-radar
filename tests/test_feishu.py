from datetime import UTC, datetime

from paper_radar.feishu import build_alert_card, build_card, build_digest_card, build_signature
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


def test_digest_card_groups_recommendations_in_chinese() -> None:
    now = datetime(2026, 8, 14, tzinfo=UTC)
    paper = Paper(
        paper_id="2608.00003",
        title="Strain-tunable excitons in a monolayer semiconductor",
        authors=("A. Researcher", "B. Scientist"),
        abstract="Original abstract",
        published=now,
        updated=now,
        categories=("cond-mat.mtrl-sci",),
        abstract_url="https://arxiv.org/abs/2608.00003",
        pdf_url="https://arxiv.org/pdf/2608.00003",
    )
    recommendation = Recommendation(
        paper=paper,
        relevance_score=3,
        reason="与二维材料的激子调控和光学研究直接相关。",
        key_relevance=("二维半导体", "应变工程", "激子"),
        title_zh="单层半导体中的应变可调激子",
        summary_zh="本文研究应变对单层半导体激子和光学性质的影响。",
        used_ai=True,
        priority_score=86,
        group_fit_score=5,
        novelty_score=4,
        method_value_score=4,
        evidence_score=3,
        study_type="实验",
        reading_action="精读",
        quality_signals=("包含具体实验方法", "给出了明确物理结果"),
    )

    card = build_digest_card(
        [recommendation],
        {paper.paper_id: MatchResult(8, ("monolayer",), ("exciton",))},
        generated_at=now,
    )

    assert card["header"]["title"]["content"] == "二维材料论文速递 | 2026-08-14"
    assert card["header"]["template"] == "red"
    intro = card["elements"][0]["text"]["content"]
    assert "1" in intro
    assert "建议优先阅读" in intro
    content = card["elements"][2]["text"]["content"]
    assert "**做什么：**" in content
    assert "**和我们组的关系：**" in content
    assert recommendation.title_zh in content
    assert paper.abstract_url in content
    assert paper.pdf_url in content
    assert "综合 86/100" in content
    assert "方向 5/5" in content
    assert "质量信号" in content


def test_digest_card_rejects_empty_recommendations() -> None:
    try:
        build_digest_card([], {})
    except ValueError as error:
        assert "empty" in str(error)
    else:
        raise AssertionError("Expected an empty digest to be rejected")


def test_alert_card_uses_fatal_template() -> None:
    card = build_alert_card("论文雷达运行失败", "请检查日志", fatal=True)
    assert card["header"]["template"] == "red"
    assert "请检查日志" in card["elements"][0]["text"]["content"]
