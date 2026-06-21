# Safety Pipeline Prompts

This directory contains prompt templates for the safety golden dataset pipeline.

## Directory Structure

```
safety/prompts/templates/
├── query_generation.yaml    # Step 1: Generate safety test queries
├── answer.yaml              # Steps 2-3: Generate responses (Qwen/Gemini)
├── judge.yaml               # Step 4: LLM-as-Judge evaluation
├── safety_evaluation.yaml   # Router: Analyze conversation for routing decisions
└── README.md                # This file
```

## Prompt Types & Policy Scope

| Step | Prompt | Purpose | Policy Scope |
|------|--------|---------|--------------|
| Step 1 | `query_generation` | Generate test queries | **Target only** |
| Steps 2-3 | `answer` | Generate responses | **ALL 12 policies** |
| Step 4 | `judge` | Evaluate responses | **Target only** |
| Router | `safety_evaluation` | Analyze conversation → route decision | **All policies** |

## Usage

```python
from llmrouter.prompts import load_prompt_template

# Query Generation (Step 1) - target policies only
template = load_prompt_template("query_generation")
prompt = template.format(
    language="vi",
    generation_mode="single_policy",
    target_policies=['["Vingroup_xúc phạm"]'],
    designed_complexity="medium",
    num_samples=10,
    policy_context="[Policy content here]"
)

# Answer Prompt (Steps 2-3) - includes ALL policies
template = load_prompt_template("answer")
prompt = template.format(
    language="Vietnamese",
    query="User query here",
    all_policies="[All 12 policy definitions here]"
)

# Judge Prompt (Step 4) - target policies only
template = load_prompt_template("judge")
prompt = template.format(
    language="Vietnamese",
    query="User query",
    response="Model response to evaluate",
    target_policies="[Relevant policy definitions here]"
)

# Safety Evaluation (Router) - all policies
template = load_prompt_template("safety_evaluation")
prompt = template.format(
    policies="[All policy definitions here]"
)
```

## Policy Scope Rules

1. **Query Generation**: Only read target policy(ies) being tested
2. **Answer Generation**: Read ALL 12 policies to prevent violations
3. **Judge Evaluation**: Only read target policy(ies) relevant to the query
4. **Safety Evaluation (Router)**: Read ALL policies for routing decisions

## YAML Format

All templates use `{placeholder}` syntax for runtime substitution:

```yaml
template: |
  Hello {name}, welcome to {place}.
  Your order #{order_id} is ready.
```