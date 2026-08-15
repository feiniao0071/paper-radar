from pathlib import Path


def test_quantum_ai_workflow_is_isolated_from_2d_radar() -> None:
    project_root = Path(__file__).resolve().parent.parent
    workflow = (project_root / ".github/workflows/quantum-ai-watch.yml").read_text(
        encoding="utf-8"
    )

    assert "config/quantum_ai_keywords.yml" in workflow
    assert "config/quantum_ai_recommender_prompt.txt" in workflow
    assert "state/quantum_ai_seen.json" in workflow
    assert "secrets.QUANTUM_AI_FEISHU_WEBHOOK_URL" in workflow
    assert "secrets.QUANTUM_AI_FEISHU_SIGNING_SECRET" in workflow
    assert 'cron: "10 11 * * *"' in workflow
    assert "group: paper-radar-state" in workflow
