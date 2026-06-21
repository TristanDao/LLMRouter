# Safety Pipeline Prompts

This directory contains all prompt templates for the safety golden dataset pipeline, organized into a centralized location.

## Directory Structure

```
safety/prompts/
├── __init__.py                # Prompt manager utilities
├── manager.py                 # Prompt manager for loading templates
├── _registry.yaml             # Registry of all prompt templates
├── templates/                 # YAML prompt templates
│   ├── query_generation.yaml  # Generate safety test queries
│   ├── judge_eng.yaml         # Judge evaluation prompt (English)
│   ├── judge_vni.yaml         # Judge evaluation prompt (Vietnamese)
│   ├── model_evaluation.yaml  # Model evaluation prompt
│   ├── safety_evaluation.yaml # Safety-aware policy evaluation
│   └── task_safety_query_generation.yaml  # Task-specific query generation
└── README.md                  # This file
```

## Usage

```python
from llmrouter.prompts import load_prompt_template

# Load a template by name (searches safety/prompts/templates/ and llmrouter/prompts/)
template = load_prompt_template("safety_evaluation")
prompt = template.format(policies=policies, history=history, target=target)

# Load query generation template
template = load_prompt_template("task_safety_query_generation")
prompt = template.format(
    language="vi",
    policy_context=context,
    generation_mode="single_policy",
    target_policies=["Vingroup_xúc phạm"],
    designed_complexity="medium",
    num_samples=10
)
```

## Template Categories

### Query Generation (`query_generation.yaml`, `task_safety_query_generation.yaml`)
Prompts for generating safety test queries:
- **query_generation.yaml**: Generic query generation with $variable syntax
- **task_safety_query_generation.yaml**: Task-specific query generation with {placeholder} syntax

Generation modes:
- **single_policy**: Generate queries testing a single target policy
- **multi_policy**: Generate queries relating to multiple policies
- **no_policy**: Generate non-violation general queries

### Evaluation (`judge_eng.yaml`, `judge_vni.yaml`, `model_evaluation.yaml`)
Prompts for evaluating model responses:
- **judge_eng.yaml**: Judge evaluation for English responses
- **judge_vni.yaml**: Judge evaluation for Vietnamese responses
- **model_evaluation.yaml**: Model evaluation for safety-aware policy routing

### Safety Evaluation (`safety_evaluation.yaml`)
Analyzes conversation history against corporate policies to determine:
- Policy violations
- Risk score (0.0 → 1.0)
- Difficulty/uncertainty score (0.0 → 1.0)
- Routing decision (LOCAL vs HIGH)
- Mapped policy conflicts

## YAML Format

Templates use two syntaxes:

### 1. `{placeholder}` syntax (Python str.format)
```yaml
template: |
  Hello {name}, welcome to {place}.
  Your order #{order_id} is ready.
```

### 2. `$variable` syntax (string.Template safe substitution)
```yaml
template: |
  $system_instruction

  Input: $input_data
  Mode: $mode
```

## Loading Templates

The `load_prompt_template()` function automatically searches:
1. `llmrouter/prompts/` (legacy location)
2. `safety/prompts/templates/` (current consolidated location)

You can load by template name only (recommended):
```python
template = load_prompt_template("safety_evaluation")
```

## Registry

The `_registry.yaml` file tracks all available templates:
```yaml
prompts:
  query_generation:
    template: templates/query_generation.yaml
  safety_evaluation:
    template: templates/safety_evaluation.yaml
  model_evaluation:
    template: templates/model_evaluation.yaml
  # ... etc
```