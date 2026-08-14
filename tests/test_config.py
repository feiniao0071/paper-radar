from pathlib import Path

from paper_radar.config import load_config


def test_project_configuration_loads() -> None:
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / "config" / "keywords.yml")
    assert "graphene" in config.matching.core_terms
    assert "DFT" in config.matching.supporting_terms
    assert config.matching.require_core_term is True
    assert config.arxiv.query_batch_size == 6
    assert config.crossref.enabled is True
    assert config.semantic_scholar.enabled is True
    assert config.run.max_papers_per_run == 10
    assert config.run.max_high_priority_per_run == 3
