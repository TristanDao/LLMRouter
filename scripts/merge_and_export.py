#!/usr/bin/env python3
"""
scripts/merge_and_export.py - Merge responses and export golden dataset

This script merges local (Qwen3-4B) and high (Gemini) responses,
then exports the final golden dataset.

Usage:
    python scripts/merge_and_export.py \\
        --local artifacts/safety_queries/queries_with_local_vni.jsonl \\
        --gemini artifacts/safety_queries/queries_gemini_vni.jsonl \\
        --output artifacts/safety_queries/golden_dataset_vni.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _extract_provider_prefix(path: str) -> str:
    """Extract provider name from file path for query_id rebasing.

    e.g., '.../answers_deepseek_local_vni.jsonl' -> 'deepseek'
          '.../answers_qwen_gemini_vni.jsonl' -> 'qwen'
          '.../answers_minimax_local_vni.jsonl' -> 'minimax'
    """
    basename = os.path.basename(path)
    for prefix in ("deepseek", "minimax", "qwen", "gemini"):
        if f"_{prefix}_" in basename or basename.startswith(f"answers_{prefix}"):
            return prefix
    return "unknown"


def merge_responses(
    input_paths: List[str],
    output_path: str,
    rebase_query_id: bool = True,
) -> int:
    """Merge multiple response files into a single file by query_id.

    Args:
        input_paths: List of input JSONL files to merge.
        output_path: Path to output merged JSONL file.
        rebase_query_id: If True, prepend provider prefix to query_id so that
            the same query from different providers gets separate records.
            e.g., "q_001" from DeepSeek becomes "deepseek/q_001".
    """
    all_records = {}
    source_models = set()

    for input_path in input_paths:
        if not os.path.exists(input_path):
            print(f"Warning: {input_path} not found, skipping")
            continue

        provider_prefix = _extract_provider_prefix(input_path)

        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                query_id = record.get("query_id")
                if query_id:
                    if rebase_query_id:
                        composite_id = f"{provider_prefix}/{query_id}"
                    else:
                        composite_id = query_id

                    if composite_id not in all_records:
                        all_records[composite_id] = {"_provider": provider_prefix}
                    model_name = record.get("model_name", "unknown")
                    source_models.add(model_name)
                    all_records[composite_id][model_name] = record

    print(f"Loaded {len(all_records)} query_ids from {len(input_paths)} files")
    print(f"Models found: {source_models}")

    merged = []
    for query_id in sorted(all_records.keys()):
        records_by_model = all_records[query_id]

        actual_records = {k: v for k, v in records_by_model.items() if k != "_provider"}
        first_record = next(iter(actual_records.values()))

        merged_record = {
            "query_id": query_id,
            "original_query_id": first_record.get("query_id", ""),
            "provider": all_records[query_id].get("_provider", "unknown"),
            "query": first_record.get("query", ""),
            "policy_ids": first_record.get("policy_ids", []),
            "policy_names": first_record.get("policy_names", []),
            "designed_complexity": first_record.get("designed_complexity", ""),
            "policy_match_type": first_record.get("policy_match_type", ""),
            "group_type": first_record.get("group_type", ""),
            "language": first_record.get("language", "vi"),
            "metadata": {
                "expected_behavior": first_record.get("metadata", {}).get("expected_behavior", ""),
                "reason": first_record.get("metadata", {}).get("reason", ""),
            },
            "responses": {},
        }

        for model_name, record in records_by_model.items():
            if model_name == "_provider":
                continue
            merged_record["responses"][model_name] = {
                "response": record.get("model_response", ""),
                "response_time": record.get("response_time", 0),
                "error": record.get("error"),
            }

            if record.get("judge_result"):
                merged_record["judge_result"] = record.get("judge_result")

        merged.append(merged_record)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in merged:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Merged {len(merged)} records to {output_path}")
    return len(merged)


def export_golden_dataset(
    input_path: str,
    output_path: str,
    sample_for_human_review: float = 0.15,
) -> int:
    """Export golden dataset with route labels."""
    with open(input_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    print(f"Loaded {len(records)} records from {input_path}")

    golden_records = []
    for record in records:
        responses = record.get("responses", {})
        local_response = responses.get("Qwen/Qwen3-4B-Instruct-2507", {}).get("response", "") if responses else ""
        gemini_response = responses.get("gemini-3.1-flash-lite", {}).get("response", "") if responses else ""
        judge_result = record.get("judge_result", {})

        consensus = judge_result.get("consensus") if judge_result else None

        if consensus == "pass" or judge_result.get("is_pass") is True:
            route_label = "local"
        elif consensus == "fail" or judge_result.get("is_pass") is False:
            route_label = "high"
        else:
            route_label = "human_review"

        golden_record = {
            "user_prompt": record.get("query", ""),
            "label": route_label,
            "query_id": record.get("query_id", ""),
            "policy_ids": record.get("policy_ids", []),
            "designed_complexity": record.get("designed_complexity", ""),
            "group_type": record.get("group_type", ""),
            "language": record.get("language", "vi"),
            "metadata": {
                "local_response": local_response[:500] if local_response else "",
                "gemini_response": gemini_response[:500] if gemini_response else "",
                "judge_consensus": consensus,
            },
        }
        golden_records.append(golden_record)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in golden_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Exported {len(golden_records)} golden records to {output_path}")

    stats = {
        "total": len(golden_records),
        "by_label": {},
    }
    for r in golden_records:
        label = r["label"]
        stats["by_label"][label] = stats["by_label"].get(label, 0) + 1
    print(f"Stats: {json.dumps(stats, ensure_ascii=False, indent=2)}")

    uncertain_records = [r for r in golden_records if r["label"] == "human_review"]
    if uncertain_records:
        review_path = output_path.replace(".jsonl", "_needs_review.jsonl")
        with open(review_path, "w", encoding="utf-8") as f:
            for r in uncertain_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Records needing human review: {len(uncertain_records)} -> {review_path}")

    return len(golden_records)


def split_dataset(
    input_path: str,
    output_dir: str,
    train_ratio: float = 0.7,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, int]:
    """Split dataset into train/dev/test."""
    with open(input_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    import random
    random.seed(42)
    random.shuffle(records)

    total = len(records)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)

    splits = {
        "train": records[:train_end],
        "dev": records[train_end:dev_end],
        "test": records[dev_end:],
    }

    os.makedirs(output_dir, exist_ok=True)
    counts = {}
    for split_name, split_records in splits.items():
        output_path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for record in split_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        counts[split_name] = len(split_records)
        print(f"Saved {len(split_records)} records to {output_path}")

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge responses and export golden dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    merge_parser = subparsers.add_parser("merge", help="Merge multiple response files")
    merge_parser.add_argument("--inputs", "-i", nargs="+", required=True,
                              help="Input JSONL files (can specify multiple)")
    merge_parser.add_argument("--output", "-o", required=True, help="Output merged JSONL")
    merge_parser.add_argument("--rebase-query-id", action="store_true", default=True,
                              help="Prepend provider prefix to query_id (default: True)")
    merge_parser.add_argument("--no-rebase-query-id", dest="rebase_query_id", action="store_false",
                              help="Disable query_id rebasing (keep original query_id)")

    export_parser = subparsers.add_parser("export", help="Export golden dataset with labels")
    export_parser.add_argument("--input", "-i", required=True, help="Merged responses JSONL")
    export_parser.add_argument("--output", "-o", required=True, help="Output golden dataset JSONL")

    split_parser = subparsers.add_parser("split", help="Split dataset into train/dev/test")
    split_parser.add_argument("--input", "-i", required=True, help="Golden dataset JSONL")
    split_parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    split_parser.add_argument("--train-ratio", type=float, default=0.7)
    split_parser.add_argument("--dev-ratio", type=float, default=0.15)
    split_parser.add_argument("--test-ratio", type=float, default=0.15)

    args = parser.parse_args()

    if args.command == "merge":
        count = merge_responses(args.inputs, args.output, rebase_query_id=args.rebase_query_id)
        print(f"Done. Merged {count} records.")
        return 0
    elif args.command == "export":
        count = export_golden_dataset(args.input, args.output)
        print(f"Done. Exported {count} records.")
        return 0
    elif args.command == "split":
        counts = split_dataset(
            args.input, args.output_dir,
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
            test_ratio=args.test_ratio,
        )
        print(f"Done. Split: {counts}")
        return 0
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
