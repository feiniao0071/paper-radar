from __future__ import annotations

import base64
import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Any

import httpx

from paper_radar.models import MatchResult, Recommendation


def build_signature(secret: str, timestamp: int) -> str:
    string_to_sign = f"{timestamp}\n{secret}"
    digest = hmac.new(string_to_sign.encode("utf-8"), digestmod=hashlib.sha256).digest()
    return base64.b64encode(digest).decode("ascii")


def _truncate(value: str, limit: int) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def build_card(recommendation: Recommendation, match: MatchResult) -> dict[str, Any]:
    paper = recommendation.paper
    display_title = recommendation.title_zh or paper.title
    summary = recommendation.summary_zh or _truncate(paper.abstract, 700)
    authors = _truncate(", ".join(paper.authors), 300)
    matched_terms = ", ".join(match.matched_terms[:8])
    relevance = ", ".join(recommendation.key_relevance[:6]) or matched_terms
    score_icons = "*" * recommendation.relevance_score
    content = (
        f"**Original title:** {paper.title}\n"
        f"**Authors:** {authors}\n"
        f"**Published:** {paper.published.date().isoformat()}\n"
        f"**Priority:** {recommendation.relevance_score}/3 {score_icons}\n"
        f"**Matched terms:** {matched_terms}\n\n"
        f"**Summary**\n{summary}\n\n"
        f"**Why it matters**\n{recommendation.reason}\n\n"
        f"**Key relevance:** {relevance}"
    )
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "blue" if recommendation.relevance_score == 2 else "red",
            "title": {"tag": "plain_text", "content": _truncate(display_title, 120)},
        },
        "elements": [
            {"tag": "div", "text": {"tag": "lark_md", "content": content}},
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "Open arXiv"},
                        "url": paper.abstract_url,
                        "type": "primary",
                    },
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "Open PDF"},
                        "url": paper.pdf_url,
                        "type": "default",
                    },
                ],
            },
        ],
    }


@dataclass(slots=True)
class FeishuClient:
    webhook_url: str
    signing_secret: str = ""
    timeout: float = 30.0

    def send(self, recommendation: Recommendation, match: MatchResult) -> None:
        payload: dict[str, Any] = {
            "msg_type": "interactive",
            "card": build_card(recommendation, match),
        }
        if self.signing_secret:
            timestamp = int(time.time())
            payload["timestamp"] = str(timestamp)
            payload["sign"] = build_signature(self.signing_secret, timestamp)

        response = httpx.post(self.webhook_url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        body = response.json()
        code = body.get("code", body.get("StatusCode", 0))
        if code not in {0, "0", None}:
            message = body.get("msg", body.get("StatusMessage", "unknown Feishu error"))
            raise RuntimeError(f"Feishu rejected the message: {code} {message}")

