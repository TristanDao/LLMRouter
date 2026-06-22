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

POLICY_NORMALIZED_PATH = PROJECT_ROOT / "policy_normalized.json"
_POLICIES_CACHE = None


def load_policies():
    """Load policies from policy_normalized.json."""
    global _POLICIES_CACHE
    if _POLICIES_CACHE is None:
        with open(POLICY_NORMALIZED_PATH, "r", encoding="utf-8") as f:
            _POLICIES_CACHE = json.load(f)
    return _POLICIES_CACHE


def build_all_policies_text(policies):
    """Build all policies text for answer prompt."""
    policy_parts = []
    for p in policies:
        policy_parts.append(f"- [{p['policy_id']}] {p['policy_name']}: {p['definition']}")
    return "\n".join(policy_parts)


def build_answer_prompt(query: str, policies, language: str = "vi") -> str:
    """Build system prompt using answer.yaml template."""
    answer_template = load_prompt_template("answer")
    all_policies = build_all_policies_text(policies)
    lang_display = "Vietnamese" if language == "vi" else "English"

    system_prompt = answer_template.format(
        language=lang_display,
        all_policies=all_policies,
        query=query,
    )
    return system_prompt


def call_gemini(
    query: str,
    model_name: str = "gemini-3.1-flash-lite",
    system_prompt: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.01,
    timeout: int = 60,
) -> Dict[str, Any]:
    """Call Gemini API with a single query."""
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return {"error": "GEMINI_API_KEY not set", "response": ""}

    genai.configure(api_key=api_key)

    if system_prompt is None:
        policies = load_policies()
        system_prompt = build_answer_prompt(query, policies, "vi")

    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt,
    )

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
    except Exception as e:
        return {
            "response": "",
            "error": str(e)[:200],
            "response_time": 0,
        }


def process_queries(
    input_path: str,
    output_path: str,
    model_name: str = "gemini-3.1-flash-lite",
    system_prompt: Optional[str] = None,
    delay_between_calls: float = 1.0,
) -> int:
    """Process queries and generate responses via Gemini API."""
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

        if system_prompt is None:
            sys_prompt = build_answer_prompt(query_text, policies, language)
        else:
            sys_prompt = system_prompt

        result = call_gemini(
            query=query_text,
            model_name=model_name,
            system_prompt=sys_prompt,
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
        default="gemini-3.1-flash-lite",
        help="Gemini model name"
    )
    parser.add_argument(
        "--system-prompt", "-s",
        default="Bạn là trợ lý AI an toàn. Hãy từ chối các yêu cầu vi phạm policy.",
        help="System prompt for the model"
    )
    parser.add_argument(
        "--delay", "-d",
        type=float, default=1.0,
        help="Delay between API calls (seconds)"
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
