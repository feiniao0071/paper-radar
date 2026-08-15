# Paper Radar

Paper Radar watches arXiv and Crossref for new work on two-dimensional quantum
materials, enriches available metadata through Semantic Scholar, ranks matching
papers, and sends the best results to a Feishu group bot.

The project is designed for GitHub Actions. It does not require a continuously
running Windows, WSL, or Docker host.

## Pipeline

1. Query the official arXiv Atom API and Crossref Works API with polite rate limits.
2. Merge duplicate preprints and journal records by DOI and normalized title.
3. Add venue, publication type, and citation metadata from Semantic Scholar when available.
4. Require at least one core term and score supporting research terms locally.
5. Reconsider papers marked `deferred` before newly discovered papers.
6. Evaluate up to twenty candidates with a strict Responses API JSON schema.
7. Send one Chinese daily digest containing up to ten recommended papers.
8. Commit the updated delivery state back to the repository.

AI evaluation is optional. If it is disabled or fails, deterministic keyword
ranking keeps the pipeline running and the digest visibly reports the degraded
mode. Source and fatal runtime failures are also reported to Feishu.

## Repository configuration

The initial research profile was built from the supplied interest and prompt
files. Edit these files to tune it:

- `config/keywords.yml`: arXiv and Crossref query terms, required core terms,
  supporting terms, exclusions, source settings, thresholds, and run limits.
- `config/recommender_prompt.txt`: laboratory interests and AI scoring rules.

Broad phrases such as `DFT`, `band structure`, and `CVD` are supporting
terms. A paper must independently match both a core 2D-material term and a
quantum-focus term such as quantum transport, superconductivity, topology,
correlation, magnetism, spin/valley physics, excitons, or moire physics.
Generic catalysis, energy storage, sensing, and AI/materials-informatics work is
excluded so it can be handled by a separate Quantum AI Materials radar.

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
| `SEMANTIC_SCHOLAR_API_KEY` | No | Raises Semantic Scholar rate limits if available |

Optional repository variables:

| Name | Default | Purpose |
| --- | --- | --- |
| `LLM_MODEL` | `gpt-5.6-sol` | Responses API model |
| `LLM_REASONING_EFFORT` | `low` | Evaluation reasoning effort |
| `ARXIV_CONTACT` | empty | Contact email sent to arXiv and Crossref |

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

The evaluator calls the Responses API and requests schema-constrained JSON for
group fit, novelty, method value, abstract evidence, study type, concrete
quality signals, Chinese title, summary, and recommendation reason. The final
priority is computed in application code rather than accepted directly from the
model. At most three papers per digest can retain the `3/3` label.

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
papers are marked `sent`; papers explicitly rated below the threshold are marked
`skipped`; papers that only missed a candidate or delivery limit are marked
`deferred` and receive priority in the next run. Failed deliveries remain
eligible for retry. Version-one state files are migrated once so historical
`skipped` records are reconsidered as `deferred`. Records older than the
configured retention window are removed.

GitHub Actions uses a concurrency group so two runs cannot update this state at
the same time.
