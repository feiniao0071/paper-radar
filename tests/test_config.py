from pathlib import Path

from paper_radar.config import load_config


def test_project_configuration_loads() -> None:
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / "config" / "keywords.yml")
    assert "graphene" in config.matching.core_terms
    assert "DFT" in config.matching.supporting_terms
    assert config.matching.require_core_term is True
    assert config.matching.require_focus_term is True
    assert "quantum transport" in config.matching.focus_terms
    assert "MnBi2Te4" in config.matching.core_terms
    assert "fractional Chern insulator" in config.matching.focus_terms
    assert "twisted MoTe2 fractional Chern" in config.arxiv.query_terms
    assert config.arxiv.query_batch_size == 6
    assert config.crossref.enabled is True
    assert config.semantic_scholar.enabled is True
    assert config.run.max_papers_per_run == 10
    assert config.run.max_high_priority_per_run == 3
    assert config.run.deep_read_enabled is True
    assert config.run.deep_read_min_priority_score == 82
    assert config.profile.digest_title == "二维量子材料论文速递"


def test_quantum_ai_configuration_loads() -> None:
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / "config" / "quantum_ai_keywords.yml")

    assert config.profile.name == "Quantum AI Materials"
    assert config.profile.digest_title == "Quantum AI Materials 论文速递"
    assert "machine learning" in config.matching.core_terms
    assert "materials discovery" in config.matching.focus_terms
    assert "agentic AI" in config.matching.core_terms
    assert "large language model" in config.matching.preferred_terms
    assert "scientific agent" in config.matching.preferred_terms
    assert "instrument control" in config.matching.focus_terms
    assert "twisted MoTe2" in config.matching.focus_terms
    assert "quantum sensing" in config.matching.focus_terms
    assert "relative binding free energy" in config.matching.excluded_terms
    assert "protein" in config.matching.excluded_terms
    assert "molecule" not in config.matching.focus_terms
    assert "quantum chemistry" not in config.matching.focus_terms
    assert "large language model condensed matter" in config.arxiv.query_terms
    assert "large language model quantum materials" in config.crossref.query_terms
    assert "machine learning materials discovery" not in config.arxiv.query_terms
    assert config.matching.require_core_term is True
    assert config.matching.require_focus_term is True
    assert config.run.deep_read_enabled is True
    assert config.run.deep_read_min_priority_score == 82
    assert config.run.max_non_preferred_papers == 3
