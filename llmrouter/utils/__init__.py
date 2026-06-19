"""
Utils package for LLMRouter scripts.

This module is intentionally import-safe: it avoids eagerly importing helpers
that pull in optional heavy dependencies such as `torch`.
"""

try:  # pragma: no cover
    from .prompting import (
        format_mc_prompt,
        format_gsm8k_prompt,
        format_math_prompt,
        format_commonsense_qa_prompt,
        format_mbpp_prompt,
        format_humaneval_prompt,
        generate_task_query,
        register_prompt,
        register_task_metric,
        PROMPT_REGISTRY,
        TASK_METRIC_REGISTRY,
    )
except Exception:  # pragma: no cover
    format_mc_prompt = None
    format_gsm8k_prompt = None
    format_math_prompt = None
    format_commonsense_qa_prompt = None
    format_mbpp_prompt = None
    format_humaneval_prompt = None
    generate_task_query = None
    register_prompt = None
    register_task_metric = None
    PROMPT_REGISTRY = {}
    TASK_METRIC_REGISTRY = {}

try:  # pragma: no cover
    from .api_calling import call_api
except Exception:  # pragma: no cover
    call_api = None

from .constants import TASK_DESCRIPTIONS, TASK_CATEGORIES, API_KEYS, HF_TOKEN, CASE_NUM

__all__ = [
    "format_mc_prompt",
    "format_gsm8k_prompt",
    "format_math_prompt",
    "format_commonsense_qa_prompt",
    "format_mbpp_prompt",
    "format_humaneval_prompt",
    "generate_task_query",
    "register_prompt",
    "register_task_metric",
    "PROMPT_REGISTRY",
    "TASK_METRIC_REGISTRY",
    "call_api",
    "TASK_DESCRIPTIONS",
    "TASK_CATEGORIES",
    "API_KEYS",
    "HF_TOKEN",
    "CASE_NUM",
]
