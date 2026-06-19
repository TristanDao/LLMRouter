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
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def merge_responses(
    local_path: str,
    gemini_path: str,
    output_path: str,
) -> int:
    """Merge local and gemini responses into a single file."""
    local_records = {}
    with open(local_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            query_id = record.get("query_id")
            if query_id:
                local_records[query_id] = record

    gemini_records = {}
    with open(gemini_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            query_id = record.get("query_id")
            if query_id:
                gemini_records[query_id] = record

    print(f"Loaded {len(local_records)} local records")
    print(f"Loaded {len(gemini_records)} gemini records")

    merged = []
    all_query_ids = set(local_records.keys()) | set(gemini_records.keys())

    for query_id in sorted(all_query_ids):
        local = local_records.get(query_id, {})
        gemini = gemini_records.get(query_id, {})

        record = {
            "query_id": query_id,
            "query": local.get("query") or gemini.get("query", ""),
            "policy_ids": local.get("policy_ids") or gemini.get("policy_ids", []),
            "policy_names": local.get("policy_names") or gemini.get("policy_names", []),
            "designed_complexity": local.get("designed_complexity") or gemini.get("designed_complexity", ""),
            "policy_match_type": local.get("policy_match_type") or gemini.get("policy_match_type", ""),
            "group_type": local.get("group_type") or gemini.get("group_type", ""),
            "language": local.get("language") or gemini.get("language", "vi"),
            "local_model": local.get("model_name", ""),
            "local_response": local.get("model_response", ""),
            "gemini_model": gemini.get("model_name", ""),
            "gemini_response": gemini.get("model_response", ""),
        }

        if local.get("response_time"):
            record["local_response_time"] = local.get("response_time", 0)
        if gemini.get("response_time"):
            record["gemini_response_time"] = gemini.get("response_time", 0)

        if local.get("judge_result"):
            record["judge_result"] = local.get("judge_result")
        elif gemini.get("judge_result"):
            record["judge_result"] = gemini.get("judge_result")

        merged.append(record)

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
        local_response = record.get("local_response", "")
        gemini_response = record.get("gemini_response", "")
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

    merge_parser = subparsers.add_parser("merge", help="Merge local and gemini responses")
    merge_parser.add_argument("--local", "-l", required=True, help="Local model responses JSONL")
    merge_parser.add_argument("--gemini", "-g", required=True, help="Gemini model responses JSONL")
    merge_parser.add_argument("--output", "-o", required=True, help="Output merged JSONL")

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
        count = merge_responses(args.local, args.gemini, args.output)
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
