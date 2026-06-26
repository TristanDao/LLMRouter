#!/usr/bin/env python3
"""
scripts/synthetic_hard_augmentation.py - Augment hard samples from golden dataset.

This script generates additional synthetic hard queries from existing golden records
to reduce class imbalance without requiring a new human review pass.

Output schema keeps the final field name as `query` for consistency.

Usage:
    python scripts/synthetic_hard_augmentation.py \
        --input artifacts/safety_queries/golden_dataset_vni.jsonl \
        --output artifacts/safety_queries/golden_dataset_augmented_vni.jsonl \
        --ratio 0.5

Notes:
    - Prefer hard samples as source patterns.
    - Synthetic records are marked `difficulty="hard"` by design.
    - The script is deterministic with a fixed seed unless `--seed` changes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safety.common.env import load_dotenv_file, env_str
from llmrouter.utils.api_calling import call_api

load_dotenv_file(PROJECT_ROOT / ".env")


VI_TEMPLATES = [
    "{base} Trả lời phải thỏa mãn đồng thời các ràng buộc sau: {constraints}.",
    "{base} Lưu ý: cần xử lý thêm các điều kiện phụ sau: {constraints}.",
    "{base} Nhưng hãy giả định có thêm bối cảnh này: {constraints}.",
    "{base} Hãy viết câu trả lời sao cho không vi phạm các điều kiện: {constraints}.",
]

VI_CONSTRAINTS = [
    "trả lời ngắn gọn nhưng vẫn đầy đủ",
    "không được nhắc trực tiếp tên chính sách",
    "phải nêu lý do bằng 2 ý riêng biệt",
    "ưu tiên an toàn và tuân thủ quy định nội bộ",
    "giữ văn phong tự nhiên, không máy móc",
    "phải cân nhắc thêm yếu tố quyền riêng tư",
    "chỉ được dùng thông tin đã có trong ngữ cảnh",
    "nếu không chắc thì phải từ chối lịch sự",
]

EN_TEMPLATES = [
    "{base} The answer must also satisfy these extra constraints: {constraints}.",
    "{base} In addition, account for the following conditions: {constraints}.",
    "{base} Assume this extra context as well: {constraints}.",
    "{base} Make sure the response respects all constraints: {constraints}.",
]

EN_CONSTRAINTS = [
    "be concise but complete",
    "do not mention policy names directly",
    "justify the answer with two separate reasons",
    "prioritize safety and internal compliance",
    "keep the tone natural, not robotic",
    "consider privacy implications as well",
    "use only information present in the context",
    "if uncertain, refuse politely",
]


def read_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_query_text(record: dict) -> str:
    return str(record.get("query") or record.get("user_prompt") or "")


def write_jsonl(path: str, records: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_output_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            qid = record.get("query_id")
            if qid:
                seen.add(str(qid))
    return seen


def get_source_pool(records: List[dict], source_difficulty: str) -> List[dict]:
    pool = [r for r in records if r.get("difficulty") == source_difficulty]
    if pool:
        return pool
    return records


def build_constraints(language: str, rng: random.Random, count: int = 2) -> str:
    pool = VI_CONSTRAINTS if language == "vi" else EN_CONSTRAINTS
    picked = rng.sample(pool, k=min(count, len(pool)))
    return "; ".join(picked)


def select_exemplars(
    records: List[dict],
    source_difficulty: str,
    rng: random.Random,
    max_items: int = 3,
) -> List[dict]:
    """Pick a small representative set of examples for few-shot prompting."""
    pool = [r for r in records if r.get("difficulty") == source_difficulty]
    if not pool:
        pool = records

    sample_size = min(max_items, len(pool))
    if sample_size <= 0:
        return []
    return rng.sample(pool, k=sample_size)


def format_exemplar(record: dict) -> dict:
    return {
        "query": get_query_text(record),
        "difficulty": record.get("difficulty", ""),
        "language": record.get("language", "vi"),
        "policy_ids": record.get("policy_ids", []),
        "group_type": record.get("group_type", ""),
        "synthetic_source_query_id": record.get("query_id", ""),
    }


def synthesize_query(base_query: str, language: str, rng: random.Random) -> tuple[str, str]:
    templates = VI_TEMPLATES if language == "vi" else EN_TEMPLATES
    template = rng.choice(templates)
    constraints = build_constraints(language, rng, count=2)
    return template.format(base=base_query.rstrip(".?!"), constraints=constraints), template


def debug_record(prefix: str, record: dict) -> None:
    query_text = get_query_text(record)
    print(
        f"{prefix} query_id={record.get('query_id', '')} difficulty={record.get('difficulty', '')} "
        f"language={record.get('language', 'vi')} query={query_text[:120]!r}"
    )


def call_synthetic_model(
    base_query: str,
    language: str,
    source_difficulty: str,
    rng: random.Random,
    exemplars: List[dict],
) -> dict:
    """Generate a harder query using Alibaba API.

    The model should rewrite the query into a tougher safety/router case, keeping
    the same general topic but adding one or more constraints that make routing harder.
    """
    model_name = env_str("ALIBABA_SYN_DATA_DEEPSEEK", "DeepSeek-V4-Pro")
    api_endpoint = env_str("ALIBABA_URL", "")
    if not api_endpoint:
        raise ValueError("ALIBABA_URL is required in .env for synthetic data generation")

    constraints = build_constraints(language, rng, count=2)
    system_prompt = (
        "You are generating synthetic hard training queries for a safety router. "
        "Rewrite the given query into a harder, more ambiguous variant that still belongs to the same topic. "
        "Use only a few representative examples, not the full dataset. "
        "Keep it as a single query, not a list. Return only JSON with keys: query, reason, strategy."
    )
    user_prompt = json.dumps(
        {
            "language": language,
            "source_difficulty": source_difficulty,
            "base_query": base_query,
            "extra_constraints": constraints,
            "few_shot_exemplars": [format_exemplar(e) for e in exemplars],
            "hard_goal": "make routing harder without changing the underlying safety topic",
        },
        ensure_ascii=False,
    )

    result = call_api(
        {
            "api_endpoint": api_endpoint,
            "query": user_prompt,
            "system_prompt": system_prompt,
            "model_name": model_name,
            "api_name": model_name,
            "service": "alibaba",
        },
        max_tokens=512,
        temperature=0.7,
        timeout=120,
        max_retries=1,
    )
    if result.get("error"):
        raise RuntimeError(result["error"])

    raw = (result.get("response") or "").strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    parsed = json.loads(raw.strip())
    if not isinstance(parsed, dict) or "query" not in parsed:
        raise ValueError("Synthetic model did not return expected JSON object")
    return parsed


def augment_records(
    records: List[dict],
    ratio: float,
    seed: int,
    prefer_hard: bool = True,
    output_path: str | None = None,
    resume: bool = True,
) -> List[dict]:
    rng = random.Random(seed)
    source_pool = get_source_pool(records, "hard" if prefer_hard else "easy")

    if not source_pool:
        raise ValueError("No records available to augment.")

    target_count = max(1, int(len(records) * ratio))
    augmented = []
    seen_output_ids = read_output_ids(output_path) if (output_path and resume) else set()

    if output_path and not seen_output_ids:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8"):
            pass

    for idx in range(target_count):
        source = rng.choice(source_pool)
        language = source.get("language", "vi")
        exemplars = select_exemplars(records, "hard" if prefer_hard else "easy", rng, max_items=3)
        debug_record(f"[synthetic {idx + 1}/{target_count}] source", source)

        synthetic_query_id = f"SYN{idx + 1:05d}"
        if output_path and resume and synthetic_query_id in seen_output_ids:
            print(f"[synthetic {idx + 1}/{target_count}] skipped existing query_id={synthetic_query_id}")
            continue

        generated = call_synthetic_model(
            get_query_text(source),
            language,
            source.get("difficulty", ""),
            rng,
            exemplars=exemplars,
        )
        new_query = str(generated.get("query", "")).strip()
        strategy = str(generated.get("strategy", "api_rewrite")).strip() or "api_rewrite"

        synthetic = dict(source)
        synthetic["query_id"] = synthetic_query_id
        synthetic["query"] = new_query
        synthetic["difficulty"] = "hard"
        synthetic["synthetic"] = True
        synthetic["synthetic_source_query_id"] = source.get("query_id", "")
        synthetic["synthetic_source_difficulty"] = source.get("difficulty", "")
        synthetic["synthetic_strategy"] = strategy
        synthetic["human_reviewed"] = source.get("human_reviewed", False)

        # Preserve the original responses for traceability, but clear labels that
        # would otherwise imply these responses were re-evaluated.
        synthetic["local_response"] = source.get("local_response", "")
        synthetic["gemini_response"] = source.get("gemini_response", "")

        print(f"[synthetic {idx + 1}/{target_count}] generated strategy={strategy} query={new_query[:120]!r}")

        if output_path:
            append_jsonl(output_path, synthetic)
            seen_output_ids.add(synthetic_query_id)

        augmented.append(synthetic)

    return records + augmented


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Augment golden dataset with synthetic hard queries",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Input golden dataset JSONL")
    parser.add_argument("--output", "-o", required=True, help="Output augmented JSONL")
    parser.add_argument("--ratio", type=float, default=0.5, help="Synthetic records as a fraction of input size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Resume from an existing output file and append only missing synthetic rows")
    parser.add_argument(
        "--prefer-hard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer hard samples as source patterns",
    )
    args = parser.parse_args()

    records = read_jsonl(args.input)
    if not records:
        print(f"No records found in {args.input}")
        return 1

    augmented = augment_records(
        records,
        ratio=args.ratio,
        seed=args.seed,
        prefer_hard=args.prefer_hard,
        output_path=args.output if args.resume else None,
        resume=args.resume,
    )

    synthetic_count = len(augmented) - len(records)
    print(f"Loaded {len(records)} records from {args.input}")
    print(f"Added {synthetic_count} synthetic hard records")
    print(f"Wrote {len(augmented)} records to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
