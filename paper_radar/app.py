from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from paper_radar.arxiv import fetch_recent_papers
from paper_radar.config import RadarConfig, load_config
from paper_radar.feishu import FeishuClient, build_digest_card
from paper_radar.matcher import match_paper
from paper_radar.models import MatchResult, Paper, Recommendation
from paper_radar.recommender import AIRecommender, fallback_recommendation
from paper_radar.state import StateStore

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find relevant arXiv papers and send them to Feishu"
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
) -> list[Paper]:
    return sorted(
        papers,
        key=lambda paper: (paper.published, matches[paper.paper_id].score),
        reverse=True,
    )


def _evaluate(
    papers: list[Paper],
    matches: dict[str, MatchResult],
    *,
    prompt_path: Path,
    no_ai: bool,
) -> list[Recommendation]:
    recommender = None if no_ai else AIRecommender.from_environment(prompt_path)
    if recommender is None:
        LOGGER.info("AI evaluation is disabled; using deterministic keyword ranking")
        return [fallback_recommendation(paper, matches[paper.paper_id]) for paper in papers]

    try:
        recommendations = recommender.evaluate(papers, matches)
        LOGGER.info("AI evaluated %d paper(s)", len(recommendations))
        return recommendations
    except Exception:
        LOGGER.exception("AI evaluation failed; using deterministic keyword ranking")
        return [fallback_recommendation(paper, matches[paper.paper_id]) for paper in papers]


def _print_preview(recommendations: list[Recommendation], matches: dict[str, MatchResult]) -> None:
    preview = build_digest_card(recommendations, matches) if recommendations else None
    print(json.dumps(preview, ensure_ascii=False, indent=2))


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
        )

    state = StateStore(args.state)
    papers = fetch_recent_papers(config.arxiv)
    matches: dict[str, MatchResult] = {}
    matched_papers: list[Paper] = []
    for paper in papers:
        match = match_paper(paper, config.matching)
        if match is not None and not state.contains(paper.paper_id):
            matches[paper.paper_id] = match
            matched_papers.append(paper)

    LOGGER.info(
        "%d unseen paper(s) matched the 2D-material rules",
        len(matched_papers),
    )
    if not matched_papers:
        return 0

    ranked = _rank_matches(matched_papers, matches)
    candidates = ranked[: config.run.ai_candidate_limit]
    recommendations = _evaluate(
        candidates,
        matches,
        prompt_path=args.prompt,
        no_ai=args.no_ai,
    )
    recommendations.sort(
        key=lambda item: (
            item.relevance_score,
            item.paper.published,
            matches[item.paper.paper_id].score,
        ),
        reverse=True,
    )
    max_papers = args.max_papers or config.run.max_papers_per_run
    selected = [
        recommendation
        for recommendation in recommendations
        if recommendation.relevance_score >= config.run.minimum_ai_relevance
    ][:max_papers]

    webhook_url = os.getenv("FEISHU_WEBHOOK_URL", "").strip()
    effective_dry_run = args.dry_run or not webhook_url
    if not webhook_url and not args.dry_run:
        LOGGER.warning("FEISHU_WEBHOOK_URL is missing; switching to dry-run mode")
    if effective_dry_run:
        _print_preview(selected, matches)
        return 0

    client = FeishuClient(
        webhook_url=webhook_url,
        signing_secret=os.getenv("FEISHU_SIGNING_SECRET", "").strip(),
    )
    sent_ids: set[str] = set()
    failures = 0
    if selected:
        try:
            client.send_digest(selected, matches)
            sent_ids.update(item.paper.paper_id for item in selected)
            LOGGER.info("Sent a Feishu digest containing %d paper(s)", len(selected))
        except Exception:
            failures = 1
            LOGGER.exception("Failed to send the Feishu paper digest")
    else:
        LOGGER.info("No papers met the delivery threshold; no digest was sent")

    now = datetime.now(UTC)
    selected_ids = {item.paper.paper_id for item in selected}
    for paper in matched_papers:
        if paper.paper_id in sent_ids:
            state.mark(paper, "sent", now=now)
        elif paper.paper_id not in selected_ids:
            state.mark(paper, "skipped", now=now)
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
    except Exception:
        LOGGER.exception("Paper Radar failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
