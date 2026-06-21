#!/usr/bin/env python3
"""
scripts/run_local_generation.py - Generate responses using Qwen3-4B on Colab

This script is designed to run on Google Colab with GPU support.
It reads queries from a JSONL file and generates responses using
Qwen/Qwen3-4B-Instruct-2507 via HuggingFace Transformers.

Usage on Colab:
    # Clone repo
    !git clone https://github.com/your-repo/LLMRouter.git
    %cd LLMRouter

    # Install dependencies
    !pip install transformers torch accelerate

    # Upload queries.jsonl (or git pull if already pushed)

    # Run generation
    !python scripts/run_local_generation.py \\
        --input artifacts/safety_queries/queries_vni.jsonl \\
        --output artifacts/safety_queries/queries_with_local_vni.jsonl

    # Download output
    # from google.colab import files
    # files.download("artifacts/safety_queries/queries_with_local_vni.jsonl")
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


def build_all_policies_text(policies: List[Dict]) -> str:
    """Build all policies text for answer prompt."""
    policy_parts = []
    for p in policies:
        policy_parts.append(f"- [{p['policy_id']}] {p['policy_name']}: {p['definition']}")
    return "\n".join(policy_parts)


def build_answer_prompt(query: str, policies: List[Dict], language: str = "vi") -> tuple:
    """Build system and user prompts using answer.yaml template."""
    answer_template = load_prompt_template("answer")
    all_policies = build_all_policies_text(policies)
    lang_display = "Vietnamese" if language == "vi" else "English"

    system_prompt = answer_template.format(
        language=lang_display,
        all_policies=all_policies,
        query=query,
    )
    return system_prompt, query


def load_model(model_name: str = "Qwen/Qwen3-4B-Instruct-2507"):
    """Load Qwen3-4B model with HuggingFace Transformers."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        raise ImportError(
            "transformers, torch, and accelerate are required. "
            "Install with: pip install transformers torch accelerate"
        )

    print(f"Loading model: {model_name}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    model.eval()

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    query: str,
    system_prompt: str = "Bạn là trợ lý AI an toàn. Hãy từ chối các yêu cầu vi phạm policy.",
    max_new_tokens: int = 512,
    temperature: float = 0.01,
) -> str:
    """Generate a single response from the model."""
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def process_queries(
    input_path: str,
    output_path: str,
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    batch_size: int = 8,
    system_prompt: Optional[str] = None,
    include_all_policies: bool = False,
) -> int:
    """Process queries and generate responses."""
    from tqdm import tqdm

    policies = load_policies()

    model, tokenizer = load_model(model_name)

    with open(input_path, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f]

    print(f"Loaded {len(queries)} queries from {input_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    processed_ids = set()
    resume_mode = False
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if "id" in obj:
                    processed_ids.add(obj["id"])
        if processed_ids:
            resume_mode = True
            queries = [q for q in queries if q.get("id") not in processed_ids]
            print(f"Resume mode: skipping {len(processed_ids)} already processed queries, {len(queries)} remaining")

    results = []
    for i, query_obj in enumerate(tqdm(queries, desc="Generating")):
        query_text = query_obj.get("query", "")
        language = query_obj.get("language", "vi")

        if system_prompt is None:
            sys_prompt, user_query = build_answer_prompt(query_text, policies, language)
        else:
            sys_prompt, user_query = system_prompt, query_text

        start_time = time.time()
        try:
            response = generate_response(
                model, tokenizer, user_query, sys_prompt
            )
            elapsed = time.time() - start_time

            result = {
                **query_obj,
                "model_name": model_name,
                "model_response": response,
                "response_time": elapsed,
            }
        except Exception as e:
            elapsed = time.time() - start_time
            result = {
                **query_obj,
                "model_name": model_name,
                "model_response": f"Error: {str(e)[:200]}",
                "response_time": elapsed,
                "error": str(e),
            }

        results.append(result)

        if (i + 1) % batch_size == 0:
            with open(output_path, "a", encoding="utf-8") as f:
                for r in results[-batch_size:]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if results:
        with open(output_path, "a", encoding="utf-8") as f:
            remaining = len(results) % batch_size
            if remaining:
                for r in results[-remaining:]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} new responses to {output_path}")
    return len(results)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate responses using Qwen3-4B on Colab",
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
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Model name on HuggingFace"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int, default=8,
        help="Batch size for progress reporting"
    )
    parser.add_argument(
        "--system-prompt", "-s",
        default=None,
        help="System prompt for the model"
    )
    parser.add_argument(
        "--include-all-policies",
        action="store_true",
        help="Include all 12 policies in system prompt (default: only relevant policies per query)"
    )
    args = parser.parse_args()

    count = process_queries(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model_name,
        batch_size=args.batch_size,
        system_prompt=args.system_prompt,
        include_all_policies=args.include_all_policies,
    )

    print(f"Done. Processed {count} queries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
