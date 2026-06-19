# Repository Map

This repo originally came from a paper codebase, but the current working
direction is the safety / policy-violation golden dataset pipeline.

## Active Focus

These are the files and folders that matter for the current project:

- `AGENT.md`
- `plan_golden dataset.md`
- `plan_pipeline_router.md`
- `policy.csv`
- `safety/router/`
- `safety/dataset/`
- `safety/tasks/safety_aware_policy.py`
- `safety/tasks/task_prompts/task_safety_aware_policy.yaml`
- `scripts/run_safety_evaluation.py`

## What This Pipeline Is For

The current goal is to:

1. Normalize safety policies.
2. Generate golden queries.
3. Benchmark `local / medium / high` tiers later when API tokens are available.
4. Produce route labels such as `local`, `medium`, `high`, and `human_review`.

## Legacy / Optional Components

These are useful only if you decide to expand beyond the current safety scope:

- `openclaw_router/`
- most of `llmrouter/models/` outside the safety pipeline support code
- `notebooks/`
- paper draft material in `figs/`, `sections/`, `paper_summary.md`, `custom.bib`

## Read This First

If you only need the current project context, read in this order:

1. `AGENT.md`
2. `plan_golden dataset.md`
3. `plan_pipeline_router.md`
4. `REPO_MAP.md`

That is enough to understand the current direction without re-reading the full repo.
