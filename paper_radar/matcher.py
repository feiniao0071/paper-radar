from __future__ import annotations

import re
import unicodedata

from paper_radar.config import MatchingConfig
from paper_radar.models import MatchResult, Paper


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).casefold()
    value = "".join(character for character in value if not unicodedata.combining(character))
    value = re.sub(r"[-_\u2010-\u2015]", " ", value)
    value = re.sub(r"[^a-z0-9+]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def contains_term(normalized_text: str, term: str) -> bool:
    normalized_term = normalize_text(term)
    return f" {normalized_term} " in f" {normalized_text} "


def _matched_terms(normalized_text: str, terms: tuple[str, ...]) -> tuple[str, ...]:
    matched = []
    seen_normalized = set()
    for term in terms:
        normalized_term = normalize_text(term)
        if normalized_term in seen_normalized:
            continue
        if contains_term(normalized_text, term):
            matched.append(term)
            seen_normalized.add(normalized_term)
    return tuple(matched)


def match_paper(paper: Paper, config: MatchingConfig) -> MatchResult | None:
    searchable = normalize_text(f"{paper.title} {paper.abstract}")
    if any(contains_term(searchable, term) for term in config.excluded_terms):
        return None

    core_terms = _matched_terms(searchable, config.core_terms)
    focus_terms = _matched_terms(searchable, config.focus_terms)
    focus_normalized = {normalize_text(term) for term in focus_terms}
    supporting_terms = tuple(
        term
        for term in _matched_terms(searchable, config.supporting_terms)
        if normalize_text(term) not in focus_normalized
    )
    if config.require_core_term and not core_terms:
        return None
    if config.require_focus_term and not focus_terms:
        return None

    score = (
        min(len(core_terms), 3) * config.core_term_weight
        + min(len(focus_terms), 3) * config.focus_term_weight
        + min(len(supporting_terms), 5) * config.supporting_term_weight
    )
    if score < config.minimum_score:
        return None
    return MatchResult(
        score=score,
        core_terms=core_terms,
        supporting_terms=supporting_terms,
        focus_terms=focus_terms,
    )
