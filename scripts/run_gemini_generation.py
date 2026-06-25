#!/usr/bin/env python3
"""
scripts/run_gemini_generation.py - Generate responses using Gemini API

This script reads queries from a JSONL file and generates responses
using gemini-3.1-flash-lite API.

Usage:
    # Generate Vietnamese responses
    python scripts/run_gemini_generation.py \\
        --input artifacts/safety_queries/queries_vni.jsonl \\
        --output artifacts/safety_queries/queries_gemini_vni.jsonl

    # Generate English responses
    python scripts/run_gemini_generation.py \\
        --input artifacts/safety_queries/queries_eng.jsonl \\
        --output artifacts/safety_queries/queries_gemini_eng.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safety.common.env import load_dotenv_file
load_dotenv_file(PROJECT_ROOT / ".env")

from llmrouter.prompts import load_prompt_template
from safety.dataset.pipeline import SafetyGoldenDatasetBuilder

_POLICIES_CACHE = None


def load_policies():
    """Load policies from policy.csv."""
    global _POLICIES_CACHE
    if _POLICIES_CACHE is None:
        builder = SafetyGoldenDatasetBuilder(
            router_config_path=str(PROJECT_ROOT / "configs/safety/router.yaml"),
            policy_path=str(PROJECT_ROOT / "policy.csv"),
            seed=42,
        )
        _POLICIES_CACHE = builder.load_policies()
    return _POLICIES_CACHE


def build_all_policies_text(policies, policy_ids=None):
    """Build policies text for answer prompt.

    Args:
        policies: List of policy dicts
        policy_ids: If provided, only include policies matching these IDs. If None, includes all.
    """
    policy_parts = []
    for p in policies:
        if policy_ids is None or str(p.policy_id) in [str(pid) for pid in policy_ids]:
            policy_parts.append(f"- [{p.policy_id}] {p.policy_name}: {p.definition}")
    return "\n".join(policy_parts)


def build_answer_prompt(query: str, policies, policy_ids, language: str = "vi") -> str:
    """Build system prompt using answer.yaml template.

    Args:
        query: The user query
        policies: List of all policy dicts
        policy_ids: Only include policies matching these IDs
        language: 'vi' or 'eng'
    """
    answer_template = load_prompt_template("answer")
    target_policies = build_all_policies_text(policies, policy_ids)
    lang_display = "Vietnamese" if language == "vi" else "English"

    system_prompt = answer_template.format(
        language=lang_display,
        all_policies=target_policies,
        query=query,
    )
    return system_prompt


def call_gemini(
    query: str,
    model_name: str = "gemini-3.1-flash-lite",
    system_prompt: Optional[str] = None,
    language: str = "vi",
    max_tokens: int = 1024,
    temperature: float = 0.01,
    timeout: int = 60,
    max_retries: int = 5,
    base_delay: float = 4.0,
) -> Dict[str, Any]:
    """Call Gemini API with a single query and retry on rate limit errors.

    Args:
        query: The user query
        model_name: Gemini model name
        system_prompt: System prompt for the model
        language: 'vi' or 'eng'
        max_tokens: Max tokens in response
        temperature: Sampling temperature
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries on rate limit errors
        base_delay: Base delay between retries in seconds (for 15 RPM = 4s)
    """
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return {"error": "GEMINI_API_KEY not set", "response": ""}

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt,
    )

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            start_time = time.time()
            response = model.generate_content(
                query,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                }
            )
            elapsed = time.time() - start_time

            return {
                "response": response.text.strip() if hasattr(response, 'text') else str(response),
                "response_time": elapsed,
                "error": None,
            }
        except ResourceExhausted as e:
            last_error = str(e)[:200]
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                retry_delay = _extract_retry_delay(e) or delay
                print(f"  [Rate limit] Attempt {attempt + 1}/{max_retries + 1} failed: {last_error}")
                print(f"  [Rate limit] Retrying in {retry_delay:.1f}s...")
                time.sleep(retry_delay)
            else:
                print(f"  [Rate limit] All {max_retries + 1} attempts failed. Giving up.")
        except Exception as e:
            return {
                "response": "",
                "error": str(e)[:200],
                "response_time": 0,
            }

    return {
        "response": "",
        "error": f"[RateLimitExceeded] {last_error}",
        "response_time": 0,
    }


def _extract_retry_delay(error: Exception) -> Optional[float]:
    """Extract retry delay from ResourceExhausted error if available."""
    try:
        if hasattr(error, 'retry_info') and error.retry_info:
            return getattr(error.retry_info, 'retry_delay', None)
        if hasattr(error, 'metadata') and error.metadata:
            delay = error.metadata.get('retry_delay')
            if delay:
                return float(delay.rstrip('s'))
    except (ValueError, AttributeError):
        pass
    return None


def process_queries(
    input_path: str,
    output_path: str,
    model_name: str = "gemini-3.1-flash-lite",
    system_prompt: Optional[str] = None,
    delay_between_calls: float = 4.0,
) -> int:
    """Process queries and generate responses via Gemini API.

    Note: Default delay is 4.0s to respect 15 RPM limit.
    """
    from tqdm import tqdm

    policies = load_policies()

    with open(input_path, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f]

    print(f"Loaded {len(queries)} queries from {input_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    processed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    qid = obj.get("query_id")
                    if qid:
                        processed_ids.add(qid)
                except json.JSONDecodeError:
                    continue
        if processed_ids:
            queries = [q for q in queries if q.get("query_id") not in processed_ids]
            print(f"Resume mode: skipping {len(processed_ids)} already processed, {len(queries)} remaining")

    results = []
    for i, query_obj in enumerate(tqdm(queries, desc="Generating")):
        query_text = query_obj.get("query", "")
        language = query_obj.get("language", "vi")
        policy_ids = query_obj.get("policy_ids", [])

        if system_prompt is None:
            sys_prompt = build_answer_prompt(query_text, policies, policy_ids, language)
        else:
            sys_prompt = system_prompt

        result = call_gemini(
            query=query_text,
            model_name=model_name,
            system_prompt=sys_prompt,
            language=language,
        )

        record = {
            **query_obj,
            "model_name": model_name,
            "model_response": result.get("response", ""),
            "response_time": result.get("response_time", 0),
        }
        if result.get("error"):
            record["error"] = result["error"]

        results.append(record)

        if (i + 1) % 10 == 0:
            with open(output_path, "a", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            results = []

        if delay_between_calls > 0 and i < len(queries) - 1:
            time.sleep(delay_between_calls)

    if results:
        with open(output_path, "a", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(queries)} responses to {output_path}")
    return len(queries)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate responses using Gemini API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input JSONL file with queries"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output JSONL file for responses"
    )
    parser.add_argument(
        "--model-name", "-m",
        default=os.getenv("GEMINI_GENERATION_NAME", "gemini-3.1-flash-lite"),
        help="Gemini model name (default: GEMINI_GENERATION_NAME from .env)"
    )
    parser.add_argument(
        "--system-prompt", "-s",
        default=None,
        help="System prompt for the model (if None, uses answer.yaml with policies)"
    )
    parser.add_argument(
        "--delay", "-d",
        type=float, default=4.0,
        help="Delay between API calls in seconds (default: 4.0 for 15 RPM limit)"
    )
    args = parser.parse_args()

    count = process_queries(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model_name,
        system_prompt=args.system_prompt,
        delay_between_calls=args.delay,
    )

    print(f"Done. Processed {count} queries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
