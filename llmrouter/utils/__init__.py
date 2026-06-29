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

try:  # pragma: no cover
    from .data_loader import load_csv, load_jsonl, jsonl_to_csv, load_pt
except Exception:  # pragma: no cover
    load_csv = None
    load_jsonl = None
    jsonl_to_csv = None
    load_pt = None

try:  # pragma: no cover
    from .model_loader import save_model, load_model
except Exception:  # pragma: no cover
    save_model = None
    load_model = None

try:  # pragma: no cover
    from .evaluation import calculate_task_performance
except Exception:  # pragma: no cover
    calculate_task_performance = None

try:  # pragma: no cover
    from .embeddings import get_longformer_embedding
except Exception:  # pragma: no cover
    get_longformer_embedding = None

try:  # pragma: no cover
    from .setup import setup_environment
except Exception:  # pragma: no cover
    setup_environment = None

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
    "load_csv",
    "load_jsonl",
    "jsonl_to_csv",
    "load_pt",
    "get_longformer_embedding",
    "load_model",
    "save_model",
    "calculate_task_performance",
    "setup_environment",
    "TASK_DESCRIPTIONS",
    "TASK_CATEGORIES",
    "API_KEYS",
    "HF_TOKEN",
    "CASE_NUM",
]
