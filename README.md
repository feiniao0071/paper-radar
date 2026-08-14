# Paper Radar

Paper Radar watches arXiv for new work related to two-dimensional materials,
ranks matching papers, optionally generates a structured AI recommendation,
and sends the best results to a Feishu group bot.

The project is designed for GitHub Actions. It does not require a continuously
running Windows, WSL, or Docker host.

## Pipeline

1. Query the official arXiv Atom API in rate-limited high-signal term batches.
2. Require at least one core term and score supporting research terms locally.
3. Remove papers already recorded in `state/seen.json`.
4. Optionally evaluate the strongest candidates with the Responses API.
5. Send one Chinese daily digest containing up to ten recommended papers.
6. Commit the updated deduplication state back to the repository.

AI evaluation is optional. If it is disabled or fails, deterministic keyword
ranking still produces useful results and the delivery pipeline continues.

## Repository configuration

The initial research profile was built from the supplied interest and prompt
files. Edit these files to tune it:

- `config/keywords.yml`: arXiv query terms, required core terms, supporting
  terms, exclusions, thresholds, and run limits.
- `config/recommender_prompt.txt`: laboratory interests and AI scoring rules.

Broad phrases such as `DFT`, `sensor`, and `band structure` are supporting
terms. They cannot select a paper unless the title or abstract also contains a
core 2D-material term. This prevents unrelated general-purpose papers from
dominating the feed.

## GitHub and Feishu setup

Create a Feishu custom bot in the target group and enable signature
verification. Then open:

`Repository Settings > Secrets and variables > Actions`

Add these repository secrets:

| Name | Required | Purpose |
| --- | --- | --- |
| `FEISHU_WEBHOOK_URL` | Yes for delivery | Feishu custom-bot webhook |
| `FEISHU_SIGNING_SECRET` | Recommended | Feishu webhook signature secret |
| `LLM_API_KEY` | No | Enables AI recommendation and Chinese summaries |
| `LLM_BASE_URL` | No | Responses-compatible API base URL |

Optional repository variables:

| Name | Default | Purpose |
| --- | --- | --- |
| `LLM_MODEL` | `gpt-5.6-sol` | Responses API model |
| `LLM_REASONING_EFFORT` | `low` | Evaluation reasoning effort |
| `ARXIV_CONTACT` | empty | Contact text appended to the arXiv User-Agent |

Never commit webhook URLs, signing secrets, or API keys to the repository.

## First run

1. Open the repository's **Actions** tab.
2. Select **Paper Radar** and choose **Run workflow**.
3. Leave `dry_run` enabled for the first run.
4. Inspect the JSON preview in the workflow log.
5. Add the Feishu secrets, run again with `dry_run` disabled, and confirm the
   digest card in the group.

For a manual delivery test after the day's papers have already been processed,
enable `resend_latest`. It ignores deduplication for that run and leaves the
saved state unchanged. Keep it disabled for scheduled and normal manual runs.

The scheduled workflow runs once per day at approximately 19:00 Beijing time.
GitHub cron uses UTC and can start a few minutes late during busy periods.

## Optional AI evaluation

The evaluator calls only the Responses API and requests schema-constrained JSON
for paper ID, score, reason, key relevance, Chinese title, and Chinese summary.
The implementation follows the OpenAI Structured Outputs guidance:

<https://developers.openai.com/api/docs/guides/structured-outputs>

For the previously configured Responses-compatible relay, use:

```text
LLM_BASE_URL=https://bench.physcai.com/openai
LLM_MODEL=gpt-5.6-sol
```

Store the URL as a GitHub secret and the model as a GitHub variable. Do not add
the local Codex `auth.json` file to this repository.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m paper_radar --dry-run --no-ai --max-results 20
ruff check .
pytest -q
```

If `FEISHU_WEBHOOK_URL` is missing, the command automatically switches to
dry-run mode and does not modify the deduplication state.

## State behavior

`state/seen.json` is deliberately small and reviewable. Successfully delivered
papers are marked `sent`; lower-ranked matches beyond the delivery limit are
marked `skipped`. Failed deliveries are not marked, so the next scheduled run
can retry them. Records older than the configured retention window are removed.

GitHub Actions uses a concurrency group so two runs cannot update this state at
the same time.
