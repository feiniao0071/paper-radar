# Paper Radar

Paper Radar runs two independent research feeds from one GitHub repository:

- **2D Quantum Materials** watches quantum phenomena in genuinely two-dimensional
  materials.
- **Quantum AI Materials** watches AI for materials and quantum science,
  scientific agents, autonomous laboratories, and quantum machine learning with
  a concrete scientific application.

Both feeds query arXiv and Crossref, enrich available metadata through Semantic
Scholar, rank matching papers, and send a Chinese digest to a dedicated Feishu
group bot. GitHub Actions hosts the scheduled jobs, so Windows, WSL, and Docker
do not need to remain online.

## Pipeline

1. Query the official arXiv Atom API and Crossref Works API with polite rate limits.
2. Merge duplicate preprints and journal records by DOI and normalized title.
3. Add venue, publication type, and citation metadata from Semantic Scholar when available.
4. Require both a profile-specific core term and a profile-specific focus term.
5. Reconsider papers marked `deferred` before newly discovered papers.
6. Evaluate up to twenty candidates with a strict Responses API JSON schema.
7. Send one concise Chinese daily digest containing up to ten recommended papers.
8. When a PDF-backed paper clears the strict Top 1 threshold, append one structured
   deep read covering its route, findings, advances, limitations, and lab takeaways.
9. Commit that profile's independent delivery state back to the repository.

AI evaluation is optional. If it is disabled or fails, deterministic keyword
ranking keeps the pipeline running and the digest reports the degraded mode.
Source and fatal runtime failures are also reported to the corresponding Feishu
group.

## Radar profiles

| Profile | Keywords | AI prompt | State | GitHub workflow |
| --- | --- | --- | --- | --- |
| 2D Quantum Materials | `config/keywords.yml` | `config/recommender_prompt.txt` | `state/seen.json` | `2D Quantum Materials Radar` |
| Quantum AI Materials | `config/quantum_ai_keywords.yml` | `config/quantum_ai_recommender_prompt.txt` | `state/quantum_ai_seen.json` | `Quantum AI Materials Radar` |

Both profiles share `config/deep_read_prompt.txt` for the optional PDF-grounded
Top 1 paper deep read.

The 2D profile requires a 2D-material term plus a quantum-physics term such as
quantum transport, superconductivity, topology, correlation, magnetism,
spin/valley physics, excitons, or moire physics. It explicitly excludes AI and
materials-informatics papers.

The Quantum AI profile requires an AI-method term plus a materials, chemistry,
condensed-matter, or quantum-science term. Generic business, medical, finance,
traffic, and consumer AI is excluded. A plain materials paper without a
substantive AI method is also rejected, so the two feeds do not duplicate one
another.

Each YAML profile also controls the Feishu title and introduction through its
`profile` section. This keeps presentation, matching, AI review, and state
isolated while sharing the source and delivery code.

## GitHub and Feishu setup

Create one Feishu custom bot in each target group and enable signature
verification. Then open:

`Repository Settings > Secrets and variables > Actions`

Add the following delivery secrets. The existing 2D values stay unchanged; the
Quantum AI values must come from the new group's bot.

| Secret | Required | Purpose |
| --- | --- | --- |
| `FEISHU_WEBHOOK_URL` | Yes for 2D delivery | 2D group's custom-bot webhook |
| `FEISHU_SIGNING_SECRET` | Recommended | 2D group's signature secret |
| `QUANTUM_AI_FEISHU_WEBHOOK_URL` | Yes for Quantum AI delivery | Quantum AI group's custom-bot webhook |
| `QUANTUM_AI_FEISHU_SIGNING_SECRET` | Recommended | Quantum AI group's signature secret |
| `LLM_API_KEY` | No | Shared AI recommendation and Chinese summaries |
| `LLM_BASE_URL` | No | Shared Responses-compatible API base URL |
| `SEMANTIC_SCHOLAR_API_KEY` | No | Raises Semantic Scholar rate limits if available |

Optional repository variables are shared by both feeds:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_MODEL` | `gpt-5.6-sol` | Responses API model |
| `LLM_REASONING_EFFORT` | `low` | Evaluation reasoning effort |
| `ARXIV_CONTACT` | empty | Contact email sent to arXiv and Crossref |

Never commit webhook URLs, signing secrets, or API keys to the repository.

## First Quantum AI run

1. Open the repository's **Actions** tab.
2. Select **Quantum AI Materials Radar** and choose **Run workflow**.
3. Keep `dry_run` enabled, `no_ai` disabled, and enable `resend_latest` for a
   complete preview that ignores the empty or existing state.
4. Inspect the JSON preview in the workflow log.
5. Add the two `QUANTUM_AI_FEISHU_*` secrets from the new Feishu group.
6. Run again with `dry_run` disabled and confirm the digest card in that group.

`resend_latest` leaves state unchanged. Keep it disabled for scheduled and
normal manual runs.

The 2D workflow runs daily at approximately 19:00 Beijing time. The Quantum AI
workflow runs at approximately 19:10. The ten-minute offset and shared
concurrency group prevent simultaneous state commits. GitHub cron can start a
few minutes late during busy periods.

## Optional AI evaluation

The evaluator calls the Responses API and requests schema-constrained JSON for
group fit, novelty, method value, abstract evidence, study type, concrete
quality signals, Chinese title, summary, and recommendation reason. The final
priority is computed in application code rather than accepted directly from the
model. At most three papers per digest can retain the `3/3` label.

The optional deep read is stricter: it requires a successful AI evaluation,
`3/3` priority, a composite score of at least 82, method value of at least 4/5,
abstract evidence of at least 3/5, and a distinct PDF URL. Only the highest
qualifying paper is processed. PDF or relay failures skip the deep read without
blocking the daily digest. Author context is restricted to affiliations and
contribution information stated in the PDF; the model is not asked to invent
biographies or recent publication lists.

Daily Feishu digests use one Card JSON 2.0 collapsible panel for the entire
delivery. The card defaults to one compact summary row; expanding it once
reveals every recommendation in the same concise format: recommendation rank,
linked original title, source/date, “做什么”, and “和我们组的关系”. Internal scores,
authors, and keyword diagnostics stay out of the group-facing digest. Individual
papers are not nested in separate panels, and a day with no new qualifying paper
remains silent.

The implementation follows the official OpenAI Structured Outputs guidance:

<https://developers.openai.com/api/docs/guides/structured-outputs>

PDF deep reads use Responses API file inputs:

<https://developers.openai.com/api/docs/guides/file-inputs>

For the configured Responses-compatible relay, use:

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

# 2D Quantum Materials preview
python -m paper_radar --dry-run --no-ai --max-results 20

# Quantum AI Materials preview
python -m paper_radar \
  --config config/quantum_ai_keywords.yml \
  --prompt config/quantum_ai_recommender_prompt.txt \
  --state state/quantum_ai_seen.json \
  --dry-run --no-ai --max-results 20

ruff check .
pytest -q
```

If the selected workflow's `FEISHU_WEBHOOK_URL` is missing, the command
automatically switches to dry-run mode and does not modify state.

## State behavior

Each radar has its own small, reviewable JSON state. Successfully delivered
papers are marked `sent`; papers explicitly rated below the threshold are
marked `skipped`; papers that only missed a candidate or delivery limit are
marked `deferred` and receive priority in the next run. Failed deliveries
remain eligible for retry. Records older than the configured retention window
are removed.

Both GitHub Actions workflows share one concurrency group so only one can update
repository state at a time.
