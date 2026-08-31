import json
from datetime import UTC, datetime, timedelta

from paper_radar.models import Paper, Recommendation
from paper_radar.state import StateStore


def make_paper(paper_id: str) -> Paper:
    now = datetime(2026, 8, 14, tzinfo=UTC)
    return Paper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        authors=("A. Researcher",),
        abstract="Abstract",
        published=now,
        updated=now,
        categories=("cond-mat.mtrl-sci",),
        abstract_url=f"https://arxiv.org/abs/{paper_id}",
        pdf_url=f"https://arxiv.org/pdf/{paper_id}",
    )


def make_recommendation(paper: Paper) -> Recommendation:
    return Recommendation(
        paper=paper,
        relevance_score=3,
        reason="cached reason",
        key_relevance=("graphene",),
        title_zh="cached title",
        summary_zh="cached summary",
        used_ai=True,
        priority_score=88,
        group_fit_score=5,
        novelty_score=4,
        method_value_score=4,
        evidence_score=4,
        study_type="experiment",
        reading_action="read",
        quality_signals=("clear method",),
    )


def test_state_round_trip_and_prune(tmp_path) -> None:
    path = tmp_path / "seen.json"
    store = StateStore(path)
    old_time = datetime(2025, 1, 1, tzinfo=UTC)
    current_time = datetime(2026, 8, 14, tzinfo=UTC)
    old_paper = make_paper("old")
    new_paper = make_paper("new")
    store.mark(old_paper, "sent", now=old_time)
    store.mark(new_paper, "sent", now=current_time)
    store.cache_ai_evaluation("old-cache", make_recommendation(old_paper), now=old_time)
    store.cache_ai_evaluation(
        "new-cache",
        make_recommendation(new_paper),
        now=current_time,
    )
    store.prune(365, now=current_time + timedelta(days=1))
    store.save()

    loaded = StateStore(path)
    assert not loaded.contains("old")
    assert loaded.contains("new")
    assert loaded.cached_ai_evaluation(old_paper, "old-cache") is None
    assert loaded.cached_ai_evaluation(new_paper, "new-cache") is not None


def test_deferred_paper_is_reconsidered() -> None:
    store = StateStore.__new__(StateStore)
    store.path = None
    store.data = {"version": 2, "papers": {}}
    paper = make_paper("deferred")

    store.mark(paper, "deferred")

    assert store.contains(paper.paper_id)
    assert store.should_consider(paper.paper_id)
    store.mark(paper, "sent")
    assert not store.should_consider(paper.paper_id)


def test_ai_evaluation_cache_round_trip(tmp_path) -> None:
    path = tmp_path / "seen.json"
    paper = make_paper("cached")
    store = StateStore(path)

    store.cache_ai_evaluation("cache-key", make_recommendation(paper))
    store.save()

    loaded = StateStore(path)
    cached = loaded.cached_ai_evaluation(paper, "cache-key")

    assert cached is not None
    assert cached.paper is paper
    assert cached.used_ai is True
    assert cached.priority_score == 88
    assert cached.key_relevance == ("graphene",)
    assert cached.quality_signals == ("clear method",)
    assert loaded.cached_ai_evaluation(make_paper("other"), "cache-key") is None


def test_version_one_skipped_records_migrate_to_deferred(tmp_path) -> None:
    path = tmp_path / "seen.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "papers": {
                    "old-skipped": {"status": "skipped"},
                    "already-sent": {"status": "sent"},
                },
            }
        ),
        encoding="utf-8",
    )

    store = StateStore(path)

    assert store.data["version"] == 2
    assert store.status("old-skipped") == "deferred"
    assert store.status("already-sent") == "sent"
