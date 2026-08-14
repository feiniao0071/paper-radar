from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class ArxivConfig:
    api_url: str
    max_results: int
    lookback_days: int
    query_batch_size: int
    query_terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MatchingConfig:
    require_core_term: bool
    minimum_score: int
    core_term_weight: int
    supporting_term_weight: int
    core_terms: tuple[str, ...]
    supporting_terms: tuple[str, ...]
    excluded_terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RunConfig:
    max_papers_per_run: int
    ai_candidate_limit: int
    minimum_ai_relevance: int
    state_retention_days: int


@dataclass(frozen=True, slots=True)
class RadarConfig:
    arxiv: ArxivConfig
    matching: MatchingConfig
    run: RunConfig


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result = tuple(str(item).strip() for item in value if str(item).strip())
    if not result:
        raise ValueError(f"{name} must contain at least one non-empty string")
    return result


def load_config(path: Path) -> RadarConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = _mapping(yaml.safe_load(handle), "configuration")

    arxiv_raw = _mapping(raw.get("arxiv"), "arxiv")
    matching_raw = _mapping(raw.get("matching"), "matching")
    run_raw = _mapping(raw.get("run"), "run")

    config = RadarConfig(
        arxiv=ArxivConfig(
            api_url=str(arxiv_raw["api_url"]),
            max_results=int(arxiv_raw["max_results"]),
            lookback_days=int(arxiv_raw["lookback_days"]),
            query_batch_size=int(arxiv_raw.get("query_batch_size", 6)),
            query_terms=_strings(arxiv_raw.get("query_terms"), "arxiv.query_terms"),
        ),
        matching=MatchingConfig(
            require_core_term=bool(matching_raw.get("require_core_term", True)),
            minimum_score=int(matching_raw.get("minimum_score", 3)),
            core_term_weight=int(matching_raw.get("core_term_weight", 3)),
            supporting_term_weight=int(matching_raw.get("supporting_term_weight", 1)),
            core_terms=_strings(matching_raw.get("core_terms"), "matching.core_terms"),
            supporting_terms=_strings(
                matching_raw.get("supporting_terms"), "matching.supporting_terms"
            ),
            excluded_terms=tuple(
                str(item).strip()
                for item in matching_raw.get("excluded_terms", [])
                if str(item).strip()
            ),
        ),
        run=RunConfig(
            max_papers_per_run=int(run_raw.get("max_papers_per_run", 10)),
            ai_candidate_limit=int(run_raw.get("ai_candidate_limit", 20)),
            minimum_ai_relevance=int(run_raw.get("minimum_ai_relevance", 2)),
            state_retention_days=int(run_raw.get("state_retention_days", 365)),
        ),
    )

    if (
        config.arxiv.max_results < 1
        or config.arxiv.lookback_days < 1
        or config.arxiv.query_batch_size < 1
    ):
        raise ValueError("arxiv limits must be positive")
    if not 1 <= config.run.minimum_ai_relevance <= 3:
        raise ValueError("run.minimum_ai_relevance must be between 1 and 3")
    return config
