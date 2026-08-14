from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from paper_radar.models import MatchResult, Paper, Recommendation

LOGGER = logging.getLogger(__name__)

RECOMMENDATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "evaluations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string"},
                    "score": {"type": "integer", "enum": [1, 2, 3]},
                    "reason": {"type": "string"},
                    "key_relevance": {"type": "array", "items": {"type": "string"}},
                    "title_zh": {"type": "string"},
                    "summary_zh": {"type": "string"},
                },
                "required": [
                    "paper_id",
                    "score",
                    "reason",
                    "key_relevance",
                    "title_zh",
                    "summary_zh",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["evaluations"],
    "additionalProperties": False,
}


def fallback_recommendation(paper: Paper, match: MatchResult) -> Recommendation:
    relevance_score = 3 if match.score >= 6 or len(match.core_terms) >= 2 else 2
    matched = match.matched_terms[:6]
    reason = "Matched configured 2D-material research terms: " + ", ".join(matched)
    return Recommendation(
        paper=paper,
        relevance_score=relevance_score,
        reason=reason,
        key_relevance=matched,
        title_zh="",
        summary_zh="",
        used_ai=False,
    )


def _extract_json(value: str) -> dict[str, Any]:
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("AI response must be a JSON object")
    return parsed


class AIRecommender:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None,
        model: str,
        reasoning_effort: str,
        prompt_path: Path,
    ) -> None:
        client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": 120.0}
        if base_url:
            client_kwargs["base_url"] = base_url.rstrip("/")
        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.prompt_template = prompt_path.read_text(encoding="utf-8")

    @classmethod
    def from_environment(cls, prompt_path: Path) -> AIRecommender | None:
        api_key = os.getenv("LLM_API_KEY", "").strip()
        if not api_key:
            return None
        return cls(
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL", "").strip() or None,
            model=os.getenv("LLM_MODEL", "gpt-5.6-sol").strip(),
            reasoning_effort=os.getenv("LLM_REASONING_EFFORT", "low").strip(),
            prompt_path=prompt_path,
        )

    def _request(self, prompt: str, *, structured: bool) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
            "reasoning": {"effort": self.reasoning_effort},
            "max_output_tokens": 6000,
            "store": False,
            "stream": True,
        }
        if structured:
            request["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "paper_recommendations",
                    "strict": True,
                    "schema": RECOMMENDATION_SCHEMA,
                }
            }
        output_parts: list[str] = []
        completed_response = None
        with self.client.responses.create(**request) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    output_parts.append(event.delta)
                elif event.type == "response.completed":
                    completed_response = event.response

        if completed_response is None or completed_response.status != "completed":
            status = getattr(completed_response, "status", "stream ended early")
            raise RuntimeError(f"AI response was not completed: {status}")
        return "".join(output_parts) or completed_response.output_text

    def evaluate(
        self,
        papers: list[Paper],
        matches: dict[str, MatchResult],
    ) -> list[Recommendation]:
        paper_info = "\n\n---\n\n".join(paper.prompt_text() for paper in papers)
        prompt = self.prompt_template.replace("{PAPER_INFO}", paper_info)
        try:
            output_text = self._request(prompt, structured=True)
        except Exception as error:
            LOGGER.warning("Structured AI evaluation failed; retrying with plain JSON: %s", error)
            fallback_prompt = (
                prompt
                + "\n\nReturn exactly one JSON object with an evaluations array. "
                "Do not wrap it in commentary."
            )
            output_text = self._request(fallback_prompt, structured=False)

        raw = _extract_json(output_text)
        evaluations = raw.get("evaluations")
        if not isinstance(evaluations, list):
            raise ValueError("AI response does not contain an evaluations array")

        by_id = {paper.paper_id: paper for paper in papers}
        recommendations: dict[str, Recommendation] = {}
        for evaluation in evaluations:
            if not isinstance(evaluation, dict):
                continue
            paper_id = str(evaluation.get("paper_id", ""))
            paper = by_id.get(paper_id)
            if paper is None:
                continue
            score = int(evaluation.get("score", 1))
            if score not in {1, 2, 3}:
                score = 1
            relevance = evaluation.get("key_relevance", [])
            if not isinstance(relevance, list):
                relevance = []
            recommendations[paper_id] = Recommendation(
                paper=paper,
                relevance_score=score,
                reason=str(evaluation.get("reason", "")).strip(),
                key_relevance=tuple(str(item).strip() for item in relevance if str(item).strip()),
                title_zh=str(evaluation.get("title_zh", "")).strip(),
                summary_zh=str(evaluation.get("summary_zh", "")).strip(),
                used_ai=True,
            )

        return [
            recommendations.get(paper.paper_id)
            or fallback_recommendation(paper, matches[paper.paper_id])
            for paper in papers
        ]
