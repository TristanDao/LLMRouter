"""
Prompt template loader for LLMRouter.

Provides a centralized way to load YAML prompt templates from subdirectories.
Templates are cached in memory after first load.
"""

import os
import re
from pathlib import Path
from typing import Optional, Dict
import yaml


# Cache for loaded templates
_TEMPLATE_CACHE: Dict[str, str] = {}

# Base directories for prompts (searched in order)
_PROMPTS_DIRS = [
    Path(__file__).parent.resolve(),
    Path(__file__).parent.parent.parent / "safety" / "prompts" / "templates",
]


def load_prompt_template(template_name: str) -> str:
    """
    Load a prompt template by name from the prompts directories.

    Searches recursively through subdirectories for YAML files matching
    the template name (without .yaml extension).

    Args:
        template_name: Name of the template to load (e.g., 'task_mc',
                      'safety_evaluation', 'agent_prompt')

    Returns:
        The raw template string from the YAML file

    Raises:
        FileNotFoundError: If no matching template is found
    """
    if template_name in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[template_name]

    # Normalize template name
    search_name = template_name
    if search_name.endswith(".yaml"):
        search_name = search_name[:-5]

    # Search for YAML file in all prompts directories
    for prompts_dir in _PROMPTS_DIRS:
        if not prompts_dir.exists():
            continue
        for yaml_file in prompts_dir.rglob("*.yaml"):
            if yaml_file.stem == search_name:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                if isinstance(data, dict) and "template" in data:
                    template_str = data["template"]
                elif isinstance(data, str):
                    template_str = data
                else:
                    template_str = str(data)

                _TEMPLATE_CACHE[template_name] = template_str
                return template_str

    raise FileNotFoundError(f"Prompt template '{template_name}' not found in {_PROMPTS_DIRS}")


def clear_cache():
    """Clear the template cache. Useful for testing."""
    _TEMPLATE_CACHE.clear()