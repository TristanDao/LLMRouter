"""
Safety Prompts Package

Centralized prompt management for safety pipeline.
Use SafetyPromptManager to load and render prompts.

Example:
    from safety.prompts.manager import get_prompt_manager, render_query_generation_prompt

    pm = get_prompt_manager()
    prompts = pm.list_prompts()

    prompt = render_query_generation_prompt(
        generation_mode="single_policy",
        target_policies=["P01"],
        designed_complexity="low",
        num_samples=5,
    )
"""

from safety.prompts.manager import (
    get_prompt_manager,
    render_query_generation_prompt,
    render_judge_prompt,
    SafetyPromptManager,
)

__all__ = [
    "get_prompt_manager",
    "render_query_generation_prompt",
    "render_judge_prompt",
    "SafetyPromptManager",
]