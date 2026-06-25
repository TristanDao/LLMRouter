#!/usr/bin/env python3
"""
scripts/determine_difficulty.py - Determine easy/hard queries based on judge results

Logic:
    easy: local_correct=True AND gemini_correct=True
    hard: local_correct=False AND gemini_correct=True
    (Both wrong or both right = excluded)

Usage:
    python scripts/determine_difficulty.py \
        --input artifacts/safety_queries/reviewed_vni.jsonl \
        --output artifacts/safety_queries/golden_dataset_vni.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_local_correct(record: dict) -> bool | None:
    """Check if local model was correct according to judge."""
    judge_result = record.get("judge_result", {})
    if judge_result:
        local_correct = judge_result.get("local_correct")
        if local_correct is not None:
            return local_correct
        is_pass = judge_result.get("is_pass")
        if is_pass is not None:
            return is_pass
        consensus = judge_result.get("consensus")
        if consensus in ("safe", "correct"):
            return True
        elif consensus in ("unsafe", "incorrect"):
            return False

    pass_val = record.get("pass")
    if pass_val is not None:
        return pass_val

    consensus_status = record.get("consensus_status", record.get("consensus"))
    if consensus_status in ("safe", "correct"):
        return True
    elif consensus_status in ("unsafe", "incorrect"):
        return False

    return None


def get_gemini_correct(record: dict) -> bool | None:
    """Check if gemini model was correct according to judge."""
    judge_result = record.get("judge_result", {})
    if judge_result:
        gemini_correct = judge_result.get("gemini_correct")
        if gemini_correct is not None:
            return gemini_correct

        is_pass = judge_result.get("is_pass")
        if is_pass is not None:
            return is_pass

        consensus = judge_result.get("consensus")

        if consensus in ("safe", "correct"):
            return True
        elif consensus in ("unsafe", "incorrect"):
            return False

    consensus_status = record.get("consensus_status", record.get("consensus"))
    if consensus_status in ("safe", "correct"):
        return True
    elif consensus_status in ("unsafe", "incorrect"):
        return False

    return None


def determine_difficulty(
    input_path: str,
    output_path: str,
) -> dict:
    """Determine easy/hard queries and export golden dataset."""
    with open(input_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    print(f"Loaded {len(records)} records from {input_path}")

    stats = {
        "total": len(records),
        "easy": 0,
        "hard": 0,
        "excluded_both_wrong": 0,
        "excluded_both_right": 0,
        "excluded_uncertain": 0,
        "excluded_local_right_gemini_wrong": 0,
    }

    golden_records = []

    for record in records:
        local_correct = get_local_correct(record)
        gemini_correct = get_gemini_correct(record)

        if local_correct is None or gemini_correct is None:
            stats["excluded_uncertain"] += 1
            continue

        if local_correct and gemini_correct:
            difficulty = "easy"
            stats["easy"] += 1
        elif not local_correct and gemini_correct:
            difficulty = "hard"
            stats["hard"] += 1
        elif local_correct and not gemini_correct:
            stats["excluded_local_right_gemini_wrong"] += 1
            continue
        else:
            stats["excluded_both_wrong"] += 1
            continue

        responses = record.get("responses", {})
        local_response = ""
        gemini_response = ""

        for key, val in responses.items():
            if "Qwen" in key or "local" in key.lower():
                local_response = val.get("response", "")
            elif "gemini" in key.lower():
                gemini_response = val.get("response", "")

        if not local_response:
            local_response = record.get("local_model_response", "")
        if not gemini_response:
            gemini_response = record.get("gemini_model_response", "")

        judge_result = record.get("judge_result", {})
        judge_consensus = judge_result.get("consensus", record.get("consensus_status", "uncertain"))

        golden_record = {
            "query_id": record.get("query_id", ""),
            "user_prompt": record.get("query", ""),
            "policy_ids": record.get("policy_ids", []),
            "designed_complexity": record.get("designed_complexity", ""),
            "group_type": record.get("group_type", ""),
            "language": record.get("language", "vi"),
            "difficulty": difficulty,
            "local_response": local_response,
            "gemini_response": gemini_response,
            "judge_consensus": judge_consensus,
            "human_reviewed": record.get("human_reviewed", False),
        }

        golden_records.append(golden_record)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in golden_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Exported {len(golden_records)} golden records to {output_path}")
    print(f"Stats: {json.dumps(stats, ensure_ascii=False, indent=2)}")

    if stats["hard"] == 0:
        print("\n⚠️ WARNING: No hard queries found! ML router has nothing to learn from.")

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Determine easy/hard queries based on judge results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input JSONL file (after human review)"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output golden dataset JSONL"
    )
    args = parser.parse_args()

    stats = determine_difficulty(args.input, args.output)
    print(f"\nDone. Total: {stats['total']}, Easy: {stats['easy']}, Hard: {stats['hard']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
