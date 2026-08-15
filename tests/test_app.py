from datetime import UTC, datetime, timedelta

from paper_radar import app
from paper_radar.models import MatchResult, Paper
from paper_radar.sources import FetchResult
from paper_radar.state import StateStore


def _paper() -> Paper:
    now = datetime.now(UTC)
    return Paper(
        paper_id="2608.00001",
        title="Graphene transport",
        authors=("Test Author",),
        abstract="Quantum transport in graphene.",
        published=now,
        updated=now,
        categories=("cond-mat.mes-hall",),
        abstract_url="https://arxiv.org/abs/2608.00001",
        pdf_url="https://arxiv.org/pdf/2608.00001",
    )


def test_resend_latest_includes_seen_paper_without_changing_state(
    tmp_path, monkeypatch
) -> None:
    paper = _paper()
    state_path = tmp_path / "seen.json"
    state = StateStore(state_path)
    state.mark(paper, "sent")
    state.save()
    original_state = state_path.read_bytes()
    previewed = []

    monkeypatch.setattr(
        app,
        "fetch_all_papers",
        lambda config: FetchResult(papers=[paper], warnings=()),
    )
    monkeypatch.setattr(
        app,
        "match_paper",
        lambda candidate, config: MatchResult(3, ("graphene",), ()),
    )
    monkeypatch.setattr(
        app,
        "_print_preview",
        lambda recommendations, matches, notices, **kwargs: previewed.extend(
            recommendations
        ),
    )

    args = app._arguments(
        [
            "--state",
            str(state_path),
            "--dry-run",
            "--no-ai",
            "--resend-latest",
        ]
    )

    assert app.run(args) == 0
    assert [item.paper.paper_id for item in previewed] == [paper.paper_id]
    assert state_path.read_bytes() == original_state


def test_calibration_caps_high_priority_recommendations() -> None:
    papers = [
        Paper(
            paper_id=f"paper-{index}",
            title=f"Paper {index}",
            authors=("Author",),
            abstract="Graphene",
            published=datetime(2026, 8, 14, index, tzinfo=UTC),
            updated=datetime(2026, 8, 14, index, tzinfo=UTC),
            categories=(),
            abstract_url="https://example.test",
            pdf_url="https://example.test/paper.pdf",
        )
        for index in range(5)
    ]
    recommendations = [
        app.Recommendation(
            paper=paper,
            relevance_score=3,
            reason="相关",
            key_relevance=("石墨烯",),
            title_zh="论文",
            summary_zh="摘要",
            used_ai=True,
            priority_score=90 - index,
            reading_action="精读",
        )
        for index, paper in enumerate(papers)
    ]
    matches = {
        paper.paper_id: MatchResult(3, ("graphene",), ()) for paper in papers
    }

    calibrated = app._calibrate_priorities(
        recommendations,
        matches,
        max_high_priority=3,
    )

    assert [item.relevance_score for item in calibrated] == [3, 3, 3, 2, 2]
    assert calibrated[-1].reading_action == "速读"


def test_delivery_selection_caps_non_preferred_machine_learning() -> None:
    papers = [
        Paper(
            paper_id=f"paper-{index}",
            title=f"Paper {index}",
            authors=("Author",),
            abstract="Physics AI",
            published=datetime(2026, 8, 14, index, tzinfo=UTC),
            updated=datetime(2026, 8, 14, index, tzinfo=UTC),
            categories=(),
            abstract_url="https://example.test",
            pdf_url="https://example.test/paper.pdf",
        )
        for index in range(6)
    ]
    recommendations = [
        app.Recommendation(
            paper=paper,
            relevance_score=2,
            reason="相关",
            key_relevance=("凝聚态物理",),
            title_zh="论文",
            summary_zh="摘要",
            used_ai=True,
        )
        for paper in papers
    ]
    matches = {
        paper.paper_id: MatchResult(
            10,
            ("large language model",) if index < 2 else ("machine learning",),
            (),
            ("condensed matter",),
            ("large language model",) if index < 2 else (),
        )
        for index, paper in enumerate(papers)
    }

    selected, quota_skipped = app._select_for_delivery(
        recommendations,
        matches,
        minimum_relevance=2,
        max_papers=10,
        max_non_preferred_papers=2,
    )

    assert [item.paper.paper_id for item in selected] == [
        "paper-0",
        "paper-1",
        "paper-2",
        "paper-3",
    ]
    assert quota_skipped == {"paper-4", "paper-5"}

    traditional_matches = {
        paper.paper_id: MatchResult(
            5,
            ("machine learning",),
            (),
            ("condensed matter",),
        )
        for paper in papers
    }
    selected, quota_skipped = app._select_for_delivery(
        recommendations,
        traditional_matches,
        minimum_relevance=2,
        max_papers=10,
        max_non_preferred_papers=2,
    )

    assert [item.paper.paper_id for item in selected] == ["paper-0", "paper-1"]
    assert quota_skipped == {"paper-2", "paper-3", "paper-4", "paper-5"}


def test_delivery_selection_preserves_profiles_without_preferred_terms() -> None:
    paper = _paper()
    recommendation = app.Recommendation(
        paper=paper,
        relevance_score=2,
        reason="相关",
        key_relevance=("二维材料",),
        title_zh="论文",
        summary_zh="摘要",
        used_ai=True,
    )
    matches = {paper.paper_id: MatchResult(5, ("graphene",), ())}

    selected, quota_skipped = app._select_for_delivery(
        [recommendation],
        matches,
        minimum_relevance=2,
        max_papers=10,
        max_non_preferred_papers=10,
    )

    assert selected == [recommendation]
    assert quota_skipped == set()


def test_deep_read_candidate_requires_inspiring_ai_evaluation() -> None:
    paper = _paper()
    routine = app.Recommendation(
        paper=paper,
        relevance_score=3,
        reason="相关",
        key_relevance=("石墨烯",),
        title_zh="常规论文",
        summary_zh="摘要",
        used_ai=True,
        priority_score=86,
        method_value_score=3,
        evidence_score=4,
    )
    inspiring = app.Recommendation(
        paper=paper,
        relevance_score=3,
        reason="高度相关",
        key_relevance=("石墨烯",),
        title_zh="启发性论文",
        summary_zh="摘要",
        used_ai=True,
        priority_score=84,
        method_value_score=4,
        evidence_score=3,
    )

    selected = app._select_deep_read_candidate(
        [routine, inspiring],
        enabled=True,
        minimum_priority_score=82,
    )

    assert selected is inspiring
    assert (
        app._select_deep_read_candidate(
            [inspiring],
            enabled=False,
            minimum_priority_score=82,
        )
        is None
    )


def test_delivery_limit_marks_remaining_paper_deferred(tmp_path, monkeypatch) -> None:
    older = _paper()
    newer = Paper(
        paper_id="2608.00002",
        title="New graphene transport",
        authors=older.authors,
        abstract=older.abstract,
        published=older.published + timedelta(seconds=1),
        updated=older.updated,
        categories=older.categories,
        abstract_url="https://arxiv.org/abs/2608.00002",
        pdf_url="https://arxiv.org/pdf/2608.00002",
    )

    class FakeClient:
        def send_digest(
            self,
            recommendations,
            matches,
            *,
            notices=(),
            digest_title="",
            digest_intro="",
        ) -> None:
            return None

    monkeypatch.setattr(
        app,
        "fetch_all_papers",
        lambda config: FetchResult(papers=[newer, older], warnings=()),
    )
    monkeypatch.setattr(
        app,
        "match_paper",
        lambda candidate, config: MatchResult(3, ("graphene",), ()),
    )
    monkeypatch.setattr(app, "_feishu_client", lambda: FakeClient())
    state_path = tmp_path / "seen.json"
    args = app._arguments(
        ["--state", str(state_path), "--no-ai", "--max-papers", "1"]
    )

    assert app.run(args) == 0

    state = StateStore(state_path)
    assert state.status(newer.paper_id) == "sent"
    assert state.status(older.paper_id) == "deferred"
    assert state.should_consider(older.paper_id)
