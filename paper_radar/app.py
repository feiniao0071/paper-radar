from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_radar.config import RadarConfig, load_config
from paper_radar.feishu import FeishuClient, build_deep_read_card, build_digest_card
from paper_radar.matcher import match_paper
from paper_radar.models import DeepRead, MatchResult, Paper, Recommendation
from paper_radar.recommender import AIRecommender, fallback_recommendation
from paper_radar.sources import fetch_all_papers
from paper_radar.state import StateStore

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find relevant research papers and send them to Feishu"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "keywords.yml",
        help="Keyword configuration file",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=PROJECT_ROOT / "config" / "recommender_prompt.txt",
        help="AI recommendation prompt",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=PROJECT_ROOT / "state" / "seen.json",
        help="Persistent deduplication state",
    )
    parser.add_argument(
        "--deep-read-prompt",
        type=Path,
        default=PROJECT_ROOT / "config" / "deep_read_prompt.txt",
        help="AI prompt used for the optional Top 1 PDF deep read",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not send or update state")
    parser.add_argument("--no-ai", action="store_true", help="Disable optional AI evaluation")
    parser.add_argument(
        "--resend-latest",
        action="store_true",
        help="Ignore deduplication for this run and leave existing state unchanged",
    )
    parser.add_argument("--lookback-days", type=int, help="Override the lookback window")
    parser.add_argument("--max-results", type=int, help="Override the arXiv result limit")
    parser.add_argument("--max-papers", type=int, help="Override the delivery limit")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def _rank_matches(
    papers: list[Paper],
    matches: dict[str, MatchResult],
    state: StateStore,
) -> list[Paper]:
    return sorted(
        papers,
        key=lambda paper: (
            state.status(paper.paper_id) == "deferred",
            paper.published,
            matches[paper.paper_id].score,
        ),
        reverse=True,
    )


def _evaluate(
    papers: list[Paper],
    matches: dict[str, MatchResult],
    *,
    recommender: AIRecommender | None,
) -> tuple[list[Recommendation], tuple[str, ...]]:
    if recommender is None:
        LOGGER.info("AI evaluation is disabled; using deterministic keyword ranking")
        return (
            [fallback_recommendation(paper, matches[paper.paper_id]) for paper in papers],
            ("AI 评估未启用，本期使用关键词兜底排序。",),
        )

    try:
        recommendations = recommender.evaluate(papers, matches)
        LOGGER.info("AI evaluated %d paper(s)", len(recommendations))
        return recommendations, ()
    except Exception as error:
        LOGGER.exception("AI evaluation failed; using deterministic keyword ranking")
        return (
            [fallback_recommendation(paper, matches[paper.paper_id]) for paper in papers],
            (f"AI 评估失败（{type(error).__name__}），本期使用关键词兜底排序。",),
        )


def _calibrate_priorities(
    recommendations: list[Recommendation],
    matches: dict[str, MatchResult],
    *,
    max_high_priority: int,
) -> list[Recommendation]:
    recommendations.sort(
        key=lambda item: (
            item.priority_score,
            item.paper.published,
            matches[item.paper.paper_id].score,
        ),
        reverse=True,
    )
    calibrated = []
    high_priority_count = 0
    for recommendation in recommendations:
        if recommendation.relevance_score == 3:
            if high_priority_count >= max_high_priority:
                recommendation = replace(
                    recommendation,
                    relevance_score=2,
                    reading_action="速读",
                )
            else:
                high_priority_count += 1
        calibrated.append(recommendation)
    return calibrated


def _select_for_delivery(
    recommendations: list[Recommendation],
    matches: dict[str, MatchResult],
    *,
    minimum_relevance: int,
    max_papers: int,
    max_non_preferred_papers: int,
) -> tuple[list[Recommendation], set[str]]:
    eligible = [
        item for item in recommendations if item.relevance_score >= minimum_relevance
    ]
    selected: list[Recommendation] = []
    quota_skipped: set[str] = set()
    non_preferred_count = 0
    for recommendation in eligible:
        if len(selected) >= max_papers:
            break
        paper_id = recommendation.paper.paper_id
        is_preferred = bool(matches[paper_id].preferred_terms)
        if not is_preferred and non_preferred_count >= max_non_preferred_papers:
            quota_skipped.add(paper_id)
            continue
        selected.append(recommendation)
        if not is_preferred:
            non_preferred_count += 1
    return selected, quota_skipped


def _select_deep_read_candidate(
    recommendations: list[Recommendation],
    *,
    enabled: bool,
    minimum_priority_score: int,
) -> Recommendation | None:
    if not enabled:
        return None
    return next(
        (
            item
            for item in recommendations
            if item.used_ai
            and item.relevance_score == 3
            and item.priority_score >= minimum_priority_score
            and item.method_value_score >= 4
            and item.evidence_score >= 3
            and bool(item.paper.pdf_url)
            and item.paper.pdf_url != item.paper.abstract_url
        ),
        None,
    )


def _generate_deep_read(
    candidate: Recommendation | None,
    recommender: AIRecommender | None,
    *,
    profile_name: str,
    prompt_path: Path,
) -> DeepRead | None:
    if candidate is None or recommender is None:
        return None
    try:
        deep_read = recommender.generate_deep_read(
            candidate.paper,
            profile_name=profile_name,
            prompt_path=prompt_path,
        )
        LOGGER.info("Generated a PDF deep read for %s", candidate.paper.paper_id)
        return deep_read
    except Exception:
        LOGGER.exception(
            "Optional PDF deep read failed for %s; continuing with the digest",
            candidate.paper.paper_id,
        )
        return None


def _print_preview(
    recommendations: list[Recommendation],
    matches: dict[str, MatchResult],
    notices: tuple[str, ...],
    *,
    deep_read: DeepRead | None,
    digest_title: str,
    digest_intro: str,
) -> None:
    digest = (
        build_digest_card(
            recommendations,
            matches,
            notices=notices,
            digest_title=digest_title,
            digest_intro=digest_intro,
        )
        if recommendations
        else None
    )
    preview: Any = digest
    if deep_read is not None:
        preview = {
            "digest": digest,
            "deep_read": build_deep_read_card(deep_read),
        }
    output = json.dumps(preview, ensure_ascii=False, indent=2) + "\n"
    if hasattr(sys.stdout, "buffer"):
        sys.stdout.buffer.write(output.encode("utf-8"))
    else:
        print(output, end="")


def _feishu_client() -> FeishuClient | None:
    webhook_url = os.getenv("FEISHU_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return None
    return FeishuClient(
        webhook_url=webhook_url,
        signing_secret=os.getenv("FEISHU_SIGNING_SECRET", "").strip(),
    )


def _send_alert(title: str, message: str, *, fatal: bool = False) -> None:
    client = _feishu_client()
    if client is None:
        LOGGER.warning("Cannot send Feishu alert because FEISHU_WEBHOOK_URL is missing")
        return
    try:
        client.send_alert(title, message, fatal=fatal)
    except Exception:
        LOGGER.exception("Failed to send the Feishu runtime alert")


def run(args: argparse.Namespace) -> int:
    config: RadarConfig = load_config(args.config)
    if args.lookback_days is not None or args.max_results is not None:
        config = replace(
            config,
            arxiv=replace(
                config.arxiv,
                lookback_days=args.lookback_days or config.arxiv.lookback_days,
                max_results=args.max_results or config.arxiv.max_results,
            ),
            crossref=replace(
                config.crossref,
                lookback_days=args.lookback_days or config.crossref.lookback_days,
            ),
        )

    state = StateStore(args.state)
    fetch_result = fetch_all_papers(config)
    papers = fetch_result.papers
    notices = list(fetch_result.warnings)
    matches: dict[str, MatchResult] = {}
    matched_papers: list[Paper] = []
    for paper in papers:
        match = match_paper(paper, config.matching)
        if match is not None and (
            args.resend_latest or state.should_consider(paper.paper_id)
        ):
            matches[paper.paper_id] = match
            matched_papers.append(paper)

    if args.resend_latest:
        LOGGER.info(
            "%d paper(s) matched the %s rules with deduplication bypassed",
            len(matched_papers),
            config.profile.name,
        )
    else:
        LOGGER.info(
            "%d pending paper(s) matched the %s rules",
            len(matched_papers),
            config.profile.name,
        )
    if not matched_papers:
        if notices and not args.dry_run:
            _send_alert(
                f"{config.profile.name}雷达降级运行",
                "\n".join(f"- {item}" for item in notices),
            )
        if state.migrated and not args.dry_run and not args.resend_latest:
            state.save()
        return 0

    ranked = _rank_matches(matched_papers, matches, state)
    candidates = ranked[: config.run.ai_candidate_limit]
    recommender = None if args.no_ai else AIRecommender.from_environment(args.prompt)
    recommendations, evaluation_notices = _evaluate(
        candidates,
        matches,
        recommender=recommender,
    )
    notices.extend(evaluation_notices)
    recommendations = _calibrate_priorities(
        recommendations,
        matches,
        max_high_priority=config.run.max_high_priority_per_run,
    )
    max_papers = args.max_papers or config.run.max_papers_per_run
    selected, quota_skipped_ids = _select_for_delivery(
        recommendations,
        matches,
        minimum_relevance=config.run.minimum_ai_relevance,
        max_papers=max_papers,
        max_non_preferred_papers=config.run.max_non_preferred_papers,
    )
    deep_read_candidate = _select_deep_read_candidate(
        selected,
        enabled=config.run.deep_read_enabled,
        minimum_priority_score=config.run.deep_read_min_priority_score,
    )
    deep_read = _generate_deep_read(
        deep_read_candidate,
        recommender,
        profile_name=config.profile.name,
        prompt_path=args.deep_read_prompt,
    )

    client = _feishu_client()
    effective_dry_run = args.dry_run or client is None
    if client is None and not args.dry_run:
        LOGGER.warning("FEISHU_WEBHOOK_URL is missing; switching to dry-run mode")
    if effective_dry_run:
        _print_preview(
            selected,
            matches,
            tuple(notices),
            deep_read=deep_read,
            digest_title=config.profile.digest_title,
            digest_intro=config.profile.digest_intro,
        )
        return 0

    sent_ids: set[str] = set()
    failures = 0
    if selected:
        try:
            client.send_digest(
                selected,
                matches,
                notices=tuple(notices),
                digest_title=config.profile.digest_title,
                digest_intro=config.profile.digest_intro,
            )
            sent_ids.update(item.paper.paper_id for item in selected)
            LOGGER.info("Sent a Feishu digest containing %d paper(s)", len(selected))
            if deep_read is not None:
                try:
                    client.send_deep_read(deep_read)
                    LOGGER.info("Sent the optional Top 1 paper deep read")
                except Exception:
                    LOGGER.exception(
                        "Failed to send the optional paper deep read; digest was sent"
                    )
        except Exception:
            failures = 1
            LOGGER.exception("Failed to send the Feishu paper digest")
    else:
        LOGGER.info("No papers met the delivery threshold; no digest was sent")
        if notices:
            _send_alert(
                f"{config.profile.name}雷达降级运行",
                "\n".join(f"- {item}" for item in notices),
            )

    if args.resend_latest:
        LOGGER.info("Resend mode left the deduplication state unchanged")
        return 1 if failures else 0

    now = datetime.now(UTC)
    selected_ids = {item.paper.paper_id for item in selected}
    recommendations_by_id = {item.paper.paper_id: item for item in recommendations}
    for paper in matched_papers:
        if paper.paper_id in sent_ids:
            state.mark(paper, "sent", now=now)
        elif paper.paper_id in selected_ids:
            continue
        elif (
            paper.paper_id in recommendations_by_id
            and (
                recommendations_by_id[paper.paper_id].relevance_score
                < config.run.minimum_ai_relevance
                or paper.paper_id in quota_skipped_ids
            )
        ):
            state.mark(paper, "skipped", now=now)
        else:
            state.mark(paper, "deferred", now=now)
    state.prune(config.run.state_retention_days, now=now)
    state.save()
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return run(args)
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted")
        return 130
    except Exception as error:
        LOGGER.exception("Paper Radar failed")
        if not args.dry_run:
            _send_alert(
                "论文雷达运行失败",
                f"任务未完成：{type(error).__name__}。请检查 GitHub Actions 日志。",
                fatal=True,
            )
        return 1


if __name__ == "__main__":
    sys.exit(main())
