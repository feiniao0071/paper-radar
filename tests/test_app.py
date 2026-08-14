from datetime import UTC, datetime

from paper_radar import app
from paper_radar.models import MatchResult, Paper
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

    monkeypatch.setattr(app, "fetch_recent_papers", lambda config: [paper])
    monkeypatch.setattr(
        app,
        "match_paper",
        lambda candidate, config: MatchResult(3, ("graphene",), ()),
    )
    monkeypatch.setattr(
        app,
        "_print_preview",
        lambda recommendations, matches: previewed.extend(recommendations),
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
