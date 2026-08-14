import json
from datetime import UTC, datetime, timedelta

from paper_radar.models import Paper
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


def test_state_round_trip_and_prune(tmp_path) -> None:
    path = tmp_path / "seen.json"
    store = StateStore(path)
    old_time = datetime(2025, 1, 1, tzinfo=UTC)
    current_time = datetime(2026, 8, 14, tzinfo=UTC)
    store.mark(make_paper("old"), "sent", now=old_time)
    store.mark(make_paper("new"), "sent", now=current_time)
    store.prune(365, now=current_time + timedelta(days=1))
    store.save()

    loaded = StateStore(path)
    assert not loaded.contains("old")
    assert loaded.contains("new")


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
