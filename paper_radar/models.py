from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Paper:
    paper_id: str
    title: str
    authors: tuple[str, ...]
    abstract: str
    published: datetime
    updated: datetime
    categories: tuple[str, ...]
    abstract_url: str
    pdf_url: str

    def prompt_text(self) -> str:
        authors = ", ".join(self.authors)
        categories = ", ".join(self.categories)
        return (
            f"Paper ID: {self.paper_id}\n"
            f"Title: {self.title}\n"
            f"Authors: {authors}\n"
            f"Categories: {categories}\n"
            f"Published: {self.published.date().isoformat()}\n"
            f"Abstract: {self.abstract}"
        )


@dataclass(frozen=True, slots=True)
class MatchResult:
    score: int
    core_terms: tuple[str, ...]
    supporting_terms: tuple[str, ...]

    @property
    def matched_terms(self) -> tuple[str, ...]:
        return self.core_terms + self.supporting_terms


@dataclass(frozen=True, slots=True)
class Recommendation:
    paper: Paper
    relevance_score: int
    reason: str
    key_relevance: tuple[str, ...]
    title_zh: str
    summary_zh: str
    used_ai: bool

