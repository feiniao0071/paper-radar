from __future__ import annotations

import base64
import hashlib
import hmac
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from paper_radar.models import MatchResult, Recommendation

BEIJING_TIMEZONE = ZoneInfo("Asia/Shanghai")


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


def build_digest_card(
    recommendations: list[Recommendation],
    matches: dict[str, MatchResult],
    *,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    if not recommendations:
        raise ValueError("Cannot build an empty paper digest")

    digest_date = (generated_at or datetime.now(UTC)).astimezone(BEIJING_TIMEZONE).date()
    high_priority_count = sum(item.relevance_score == 3 for item in recommendations)
    if high_priority_count:
        priority_note = f"其中 {high_priority_count} 篇建议优先阅读。"
    else:
        priority_note = "本期均为值得关注的相关工作。"

    elements: list[dict[str, Any]] = [
        {
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": (
                    f"本期筛选出 **{len(recommendations)}** 篇与二维材料、量子器件和"
                    f"纳米加工相关的新论文，{priority_note}"
                ),
            },
        }
    ]

    for index, recommendation in enumerate(recommendations, start=1):
        paper = recommendation.paper
        match = matches[paper.paper_id]
        display_title = _truncate(recommendation.title_zh or paper.title, 100)
        summary = recommendation.summary_zh or paper.abstract
        relevance = _truncate(", ".join(recommendation.key_relevance[:6]), 180)
        if not relevance:
            relevance = _truncate(", ".join(match.matched_terms[:6]), 180)
        categories = _truncate(", ".join(paper.categories[:3]), 100)
        authors = _truncate(", ".join(paper.authors), 160)
        content = (
            f"**{index}. {display_title}**\n"
            f"[arXiv]({paper.abstract_url}) · [PDF]({paper.pdf_url}) · "
            f"{paper.published.date().isoformat()} · 优先级 {recommendation.relevance_score}/3\n"
            f"**英文题目：** {_truncate(paper.title, 220)}\n"
            f"**作者：** {authors}\n"
            f"**分类：** {categories}\n\n"
            f"**做什么：** {_truncate(summary, 320)}\n\n"
            f"**和我们组的关系：** {_truncate(recommendation.reason, 200)}\n\n"
            f"**关键词：** {relevance}"
        )
        elements.append({"tag": "hr"})
        elements.append({"tag": "div", "text": {"tag": "lark_md", "content": content}})

    elements.extend(
        [
            {"tag": "hr"},
            {
                "tag": "note",
                "elements": [
                    {
                        "tag": "plain_text",
                        "content": "仅推送新增且达到推荐阈值的论文；没有新增时保持安静。",
                    }
                ],
            },
        ]
    )
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "red" if high_priority_count else "blue",
            "title": {
                "tag": "plain_text",
                "content": f"二维材料论文速递 | {digest_date.isoformat()}",
            },
        },
        "elements": elements,
    }


@dataclass(slots=True)
class FeishuClient:
    webhook_url: str
    signing_secret: str = ""
    timeout: float = 30.0

    def _send_card(self, card: dict[str, Any]) -> None:
        payload: dict[str, Any] = {
            "msg_type": "interactive",
            "card": card,
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

    def send(self, recommendation: Recommendation, match: MatchResult) -> None:
        self._send_card(build_card(recommendation, match))

    def send_digest(
        self,
        recommendations: list[Recommendation],
        matches: dict[str, MatchResult],
    ) -> None:
        self._send_card(build_digest_card(recommendations, matches))
