# AGENT Context for LLMRouter Safety Pipeline

## Current Goal

Build the pipeline for a safety / policy-violation golden dataset and the downstream router training flow.

This repo is a clone of a paper codebase, so the work should stay focused:

- use the existing router framework where it helps,
- avoid expanding the project with unrelated paper artifacts,
- do not run real model benchmarks until API tokens and model names are confirmed.

## What The User Actually Wants

The user is not asking for a full repo-wide refactor.

They want:

1. A pipeline that can synthesize safety queries.
2. A way to benchmark `local / medium / high` tiers later.
3. A golden dataset with labels for routing.
4. A compact repo context file so the next agent does not need to re-read the whole codebase.

## Important Status

- We are still at the pipeline-building stage.
- No real API-token-backed benchmark should be executed yet.
- The exact model names for `local`, `medium`, and `high` are still pending.
- Mentor feedback on label policy is still pending.
- The dataset labeling scheme is still being aligned with the safety router goal.

## Source Of Truth

Use these files first:

- `plan_golden dataset.md`
- `plan_pipeline_router.md`
- `task.csv`
- `policy.csv`

These define the intended safety-router / golden-dataset workflow.

## Safety Pipeline Shape

The intended flow is:

1. Normalize policies into a consistent schema.
2. Generate synthetic queries using 3 models (ALL RUN):
   - `MiniMax-M2.7` (MiniMax API - PRIMARY)
   - `DeepSeek-V4-Pro` (Alibaba API)
   - `qwen3-next-80b-a3b-thinking` (Alibaba API)
3. Query groups:
   - `single-policy`
   - `multi-policy`
   - `no-policy`
4. Assign a designed complexity:
   - `low`
   - `medium`
   - `high`
5. Run 2 response generation models:
   - `local`: Qwen3-4B (Colab GPU)
   - `high`: Gemini (API)
6. Use judge + human review to produce final routing labels:
   - `local`
   - `high`

**Dataset size:** 330 queries × 3 query-models × 2 response-models × 2 languages = **3,960 responses**

## Current Repository Entry Points

The safety-specific code lives here:

- `safety/router/`
- `safety/dataset/`
- `safety/tasks/safety_aware_policy.py`
- `safety/tasks/task_prompts/task_safety_aware_policy.yaml`
- `scripts/run_safety_evaluation.py`

These are the files to extend if the pipeline changes.

## Router / Task Notes

The safety router is the project-specific piece that matters for this task.

It is intended to:

- parse custom policies,
- estimate `route`, `risk`, `difficulty`, and `violation_status`,
- produce dataset labels for the golden set,
- serve as the core benchmark router later.

The task registration exists so `generate_task_query("safety_aware_policy", ...)` can work without extra wiring.

## Files That Are Probably Not Core To This Task

Do not spend time expanding these unless the user explicitly asks:

- multi-round router families,
- personalized router families,
- graph-heavy router families,
- agentic router families,
- notebooks,
- paper draft artifacts,
- demo scripts unrelated to safety dataset generation.

These can be treated as archive candidates if the repo needs to be trimmed.

## Repo Cleanup Guidance

If the next agent is asked to reduce project bloat, prefer:

- keep `llmrouter/data`, `llmrouter/utils`, and the safety custom router/task files,
- archive, do not delete blindly, the paper-specific router families not used for safety routing,
- keep generated outputs under `artifacts/safety_golden/`,
- avoid touching unrelated user changes already present in the worktree.

## Environment Notes

The local environment may be incomplete:

- `torch` may be absent,
- `PyYAML` may be absent,
- API tokens are not yet configured.

So the pipeline should remain import-safe and should not require live API access just to understand the structure.

## Operational Rules For The Next Agent

- Do not run the real benchmark until the user provides the model names and API tokens.
- Do not replace the user’s label strategy without checking the mentor feedback.
- Do not revert unrelated uncommitted changes.
- Prefer small, additive changes over large refactors.
- If a new output is generated, keep it inside `artifacts/safety_golden/`.

## Practical Next Steps

When the user is ready, the next agent should:

1. Confirm the exact `local / medium / high` model names.
2. Confirm label policy from mentor feedback.
3. Wire the real API calls into the safety pipeline.
4. Generate the golden dataset.
5. Export train/dev/test JSONL and a manifest.
6. Train the downstream router on the resulting labels.
