"""Canonical safety task registry."""

try:  # pragma: no cover
    from .safety_aware_policy import format_safety_aware_policy_prompt, safety_aware_metric
except Exception:  # pragma: no cover
    format_safety_aware_policy_prompt = None
    safety_aware_metric = None

__all__ = ["format_safety_aware_policy_prompt", "safety_aware_metric"]
