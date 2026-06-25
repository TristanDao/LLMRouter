"""
Safety Prompt Manager - Centralized prompt management for safety pipeline.

All prompts are defined in _registry.yaml and rendered via this module.
Uses string.Template for safe substitution (avoids conflict with JSON {}).

Directory structure:
    safety/prompts/
        _registry.yaml        # Central registry
        templates/
            query_generation.yaml
            model_evaluation.yaml
            judge_eng.yaml
            judge_vni.yaml
        manager.py            # This file
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional

import yaml


_PROJECT_ROOT = Path(__file__).parent.parent.parent
_PROMPTS_DIR = Path(__file__).parent
_REGISTRY_PATH = _PROMPTS_DIR / "_registry.yaml"
_TEMPLATES_DIR = _PROMPTS_DIR / "templates"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class SafetyPromptManager:
    """
    Central manager for all safety pipeline prompts.

    Usage:
        pm = SafetyPromptManager()
        prompt = pm.get("query_generation", generation_mode="single_policy", ...)
    """

    _instance: Optional["SafetyPromptManager"] = None

    def __new__(cls) -> "SafetyPromptManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        self._registry = _load_yaml(_REGISTRY_PATH)
        self._templates: Dict[str, str] = {}
        self._templates_dir = _TEMPLATES_DIR
        self._load_all_templates()

    def _load_all_templates(self) -> None:
        templates_dir = self._templates_dir
        if templates_dir.exists():
            for f in templates_dir.glob("*.yaml"):
                key = f.stem
                data = _load_yaml(f)
                if "template" in data:
                    self._templates[key] = data["template"]

    def get(self, prompt_name: str, **kwargs: Any) -> str:
        """
        Get a rendered prompt by name with keyword substitution.

        Uses string.Template for safe substitution - uses $var or ${var}
        instead of {var} to avoid conflict with JSON in prompt content.

        Args:
            prompt_name: Name of the prompt (e.g., "query_generation")
            **kwargs: Key-value pairs for template substitution

        Returns:
            Rendered prompt string
        """
        if prompt_name not in self._templates:
            available = list(self._templates.keys())
            raise KeyError(
                f"Prompt '{prompt_name}' not found. Available: {available}"
            )

        template_str = self._templates[prompt_name]
        template = Template(template_str)

        safe_kwargs = {k: self._ensure_string(v) for k, v in kwargs.items()}
        return template.safe_substitute(safe_kwargs)

    def get_raw(self, prompt_name: str) -> str:
        """Get raw template without rendering."""
        return self._templates.get(prompt_name, "")

    def list_prompts(self) -> List[str]:
        """List all available prompt names."""
        return list(self._templates.keys())

    def reload(self) -> None:
        """Reload all templates from disk."""
        self._templates.clear()
        self._load_all_templates()

    @staticmethod
    def _ensure_string(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, (list, dict)):
            import json
            return json.dumps(value)
        return str(value)


def get_prompt_manager() -> SafetyPromptManager:
    return SafetyPromptManager()


def render_query_generation_prompt(
    generation_mode: str,
    target_policies: List[str],
    designed_complexity: str,
    num_samples: int,
    language: str = "vi",
    policy_context: str = "",
) -> str:
    """Render query generation prompt with given parameters."""
    pm = get_prompt_manager()
    target_policies_json = json.dumps(target_policies)
    prompt_name = "query_generation_en" if language == "eng" else "query_generation_vi"
    return pm.get(
        prompt_name,
        generation_mode=generation_mode,
        target_policies=target_policies_json,
        designed_complexity=designed_complexity,
        num_samples=num_samples,
        language=language,
        policy_context=policy_context or "No additional policy context.",
    )


def render_judge_prompt(
    language: str,
    query: str,
    responses: Dict[str, str],
    policy_context: str,
) -> str:
    """Render judge prompt with given parameters."""
    pm = get_prompt_manager()
    prompt_name = f"judge_{language}"
    responses_str = "\n".join([f"- {model}: {resp}" for model, resp in responses.items()])
    return pm.get(
        prompt_name,
        query=query,
        responses=responses_str,
        policy_context=policy_context,
    )


import json
