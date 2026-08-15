from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from paper_radar.config import RadarConfig, load_config
from paper_radar.feishu import FeishuClient, build_digest_card
from paper_radar.matcher import match_paper
from paper_radar.models import MatchResult, Paper, Recommendation
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
    prompt_path: Path,
    no_ai: bool,
) -> tuple[list[Recommendation], tuple[str, ...]]:
    recommender = None if no_ai else AIRecommender.from_environment(prompt_path)
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


def _print_preview(
    recommendations: list[Recommendation],
    matches: dict[str, MatchResult],
    notices: tuple[str, ...],
    *,
    digest_title: str,
    digest_intro: str,
) -> None:
    preview = (
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
    recommendations, evaluation_notices = _evaluate(
        candidates,
        matches,
        prompt_path=args.prompt,
        no_ai=args.no_ai,
    )
    notices.extend(evaluation_notices)
    recommendations = _calibrate_priorities(
        recommendations,
        matches,
        max_high_priority=config.run.max_high_priority_per_run,
    )
    max_papers = args.max_papers or config.run.max_papers_per_run
    selected = [
        recommendation
        for recommendation in recommendations
        if recommendation.relevance_score >= config.run.minimum_ai_relevance
    ][:max_papers]

    client = _feishu_client()
    effective_dry_run = args.dry_run or client is None
    if client is None and not args.dry_run:
        LOGGER.warning("FEISHU_WEBHOOK_URL is missing; switching to dry-run mode")
    if effective_dry_run:
        _print_preview(
            selected,
            matches,
            tuple(notices),
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
            and recommendations_by_id[paper.paper_id].relevance_score
            < config.run.minimum_ai_relevance
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
