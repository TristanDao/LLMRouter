#!/usr/bin/env python3
"""
scripts/generate_queries.py - Generate safety queries using LLM

Calls pipeline.py::SafetyGoldenDatasetBuilder.build_queries_with_llm()

Usage:
    # Generate Vietnamese queries with specific model
    python scripts/generate_queries.py --language vi --model minimax --output artifacts/safety_queries/queries_minimax_vni.jsonl
    python scripts/generate_queries.py --language vi --model deepseek --output artifacts/safety_queries/queries_deepseek_vni.jsonl
    python scripts/generate_queries.py --language vi --model qwen --output artifacts/safety_queries/queries_qwen_vni.jsonl

    # Generate English queries with specific model
    python scripts/generate_queries.py --language eng --model minimax --output artifacts/safety_queries/queries_minimax_eng.jsonl
    python scripts/generate_queries.py --language eng --model deepseek --output artifacts/safety_queries/queries_deepseek_eng.jsonl
    python scripts/generate_queries.py --language eng --model qwen --output artifacts/safety_queries/queries_qwen_eng.jsonl

    # Resume: if output file exists, automatically continues from last query
    # Force restart (overwrite existing file after backup):
    python scripts/generate_queries.py --language vi --model minimax --output artifacts/safety_queries/queries_minimax_vni.jsonl --force-restart
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safety.dataset.pipeline import SafetyGoldenDatasetBuilder

SEMANTIC_MULTI_POLICY_GROUPS = [
    [1, 4, 5],   # Group 1: Vingroup violations
    [1, 3, 5],   # Group 2: Vingroup reputation attack
    [1, 4, 7],   # Group 3: Vingroup confidential + privacy
    [3, 2],      # Group 4: Vingroup political debate
    [3, 2, 6],   # Group 5: Vingroup + politics + sovereignty
    [2, 6],      # Group 6: Politics + sovereignty
    [4, 7],      # Group 7: Privacy + internal
    [8, 9],      # Group 8: Self-harm + dangerous behaviors
    [8, 10],     # Group 9: Self-harm + domestic violence
    [9, 10],     # Group 10: Dangerous + violence
    [11, 12],    # Group 11: Child abuse + gambling
    [1, 5, 7],   # Group 12: Vingroup multi-aspect
    [2, 3, 6],   # Group 13: Political + sovereignty
    [4, 5, 7],   # Group 14: Vingroup privacy + rumors + internal
    [9, 10, 11], # Group 15: Violence cluster
]


def parse_existing_queries(filepath: str) -> dict:
    """Parse existing queries file and return stats."""
    stats = {
        "total": 0,
        "by_group_type": defaultdict(int),
        "by_complexity": defaultdict(int),
        "by_group_and_complexity": defaultdict(int),
        "last_query_id": None,
        "query_ids": set(),
    }

    if not os.path.exists(filepath):
        return stats

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                stats["total"] += 1
                stats["by_group_type"][record.get("group_type", "unknown")] += 1
                stats["by_complexity"][record.get("designed_complexity", "unknown")] += 1

                key = f"{record.get('group_type', '')}_{record.get('designed_complexity', '')}"
                stats["by_group_and_complexity"][key] += 1

                query_id = record.get("query_id", "")
                if query_id.startswith("Q"):
                    stats["query_ids"].add(query_id)
                    if stats["last_query_id"] is None or query_id > stats["last_query_id"]:
                        stats["last_query_id"] = query_id
            except json.JSONDecodeError:
                continue

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate safety queries using LLM")
    parser.add_argument("--language", "-l", default="vi", choices=["vi", "eng"],
                        help="Language for queries (vi= Vietnamese, eng= English)")
    parser.add_argument("--model", "-m", required=True,
                        choices=["minimax", "deepseek", "qwen"],
                        help="Query generation model to use")
    parser.add_argument("--output", "-o", required=True,
                        help="Output JSONL file path")
    parser.add_argument("--single-per-policy", type=int, default=5,
                        help="Number of samples per policy per complexity")
    parser.add_argument("--multi-groups", type=int, default=15,
                        help="Number of multi-policy groups")
    parser.add_argument("--multi-per-group", type=int, default=3,
                        help="Number of samples per multi-policy group per complexity")
    parser.add_argument("--no-policy-per-complexity", type=int, default=20,
                        help="Number of no-policy samples per complexity")
    parser.add_argument("--force-restart", action="store_true",
                        help="Delete existing file and start fresh (after backup)")
    args = parser.parse_args()

    print(f"Initializing SafetyGoldenDatasetBuilder...")
    builder = SafetyGoldenDatasetBuilder(
        router_config_path=str(PROJECT_ROOT / "configs/safety/router.yaml"),
        policy_path=str(PROJECT_ROOT / "policy.csv"),
        seed=42,
        output_dir=str(PROJECT_ROOT / "artifacts" / "safety_queries"),
    )

    policies = builder.load_policies()
    print(f"Loaded {len(policies)} policies")

    print(f"Generating queries with LLM...")
    print(f"  model: {args.model}")
    print(f"  single_per_policy: {args.single_per_policy}")
    print(f"  multi_groups: {args.multi_groups}")
    print(f"  multi_per_group: {args.multi_per_group}")
    print(f"  no_policy_per_complexity: {args.no_policy_per_complexity}")
    print(f"  language: {args.language}")
    print(f"  output: {args.output}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    stats = parse_existing_queries(args.output)

    if args.force_restart and stats["total"] > 0:
        backup_path = args.output + f".backup_before_restart_{stats['last_query_id']}"
        print(f"Force restart: backing up existing {stats['total']} queries to {backup_path}")
        os.rename(args.output, backup_path)
        stats = {"total": 0, "by_group_type": defaultdict(int), "by_complexity": defaultdict(int),
                 "by_group_and_complexity": defaultdict(int), "last_query_id": None, "query_ids": set()}

    single_policy_count = stats["by_group_type"].get("single_policy", 0)
    multi_policy_count = stats["by_group_type"].get("multi_policy", 0)
    no_policy_count = stats["by_group_type"].get("no_policy", 0)

    single_policy_batches = len(policies) * 3
    multi_policy_batches = args.multi_groups * 2
    no_policy_batches = 3

    section_offsets = {
        "single_policy": {
            "batches": single_policy_count // args.single_per_policy,
            "items": single_policy_count % args.single_per_policy
        },
        "multi_policy": {
            "batches": multi_policy_count // args.multi_per_group,
            "items": multi_policy_count % args.multi_per_group
        },
        "no_policy": {
            "batches": no_policy_count // args.no_policy_per_complexity,
            "items": no_policy_count % args.no_policy_per_complexity
        }
    }

    total_queries = single_policy_count + multi_policy_count + no_policy_count
    start_idx = total_queries

    print(f"\nExisting file analysis:")
    print(f"  Total queries: {stats['total']}")
    print(f"  Last query_id: {stats['last_query_id']}")
    print(f"  By group_type: {dict(stats['by_group_type'])}")
    print(f"  By complexity: {dict(stats['by_complexity'])}")
    print(f"\nSection breakdown:")
    print(f"  single_policy: {single_policy_count} queries = {section_offsets['single_policy']['batches']} batches + {section_offsets['single_policy']['items']} items")
    print(f"  multi_policy: {multi_policy_count} queries = {section_offsets['multi_policy']['batches']} batches + {section_offsets['multi_policy']['items']} items")
    print(f"  no_policy: {no_policy_count} queries = {section_offsets['no_policy']['batches']} batches + {section_offsets['no_policy']['items']} items")
    print(f"\nGeneration will continue from query_id: Q{(start_idx + 1):04d}")
    print(f"  section_offsets: {section_offsets}")

    f = open(args.output, "a", encoding="utf-8")

    total_count = 0
    new_stats = {
        "by_mode": {},
        "by_complexity": {},
    }

    def streaming_callback(q):
        """Called after each query is generated - writes immediately to file"""
        nonlocal total_count, start_idx
        total_count += 1
        start_idx += 1

        record = {
            "query_id": f"Q{start_idx:04d}",
            "query": q.query,
            "policy_ids": q.policy_ids,
            "policy_names": q.policy_names,
            "designed_complexity": q.designed_complexity,
            "policy_match_type": q.policy_match_type,
            "group_type": q.group_type,
            "language": args.language,
            "query_model": args.model,
            "metadata": q.metadata,
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

        new_stats["by_mode"][q.group_type] = new_stats["by_mode"].get(q.group_type, 0) + 1
        new_stats["by_complexity"][q.designed_complexity] = new_stats["by_complexity"].get(q.designed_complexity, 0) + 1

        if total_count % 10 == 0:
            print(f"  Progress: {total_count} new queries written (query_id: Q{start_idx:04d})...")

    try:
        generation_offset = total_queries
        builder.build_queries_with_llm(
            single_per_policy=args.single_per_policy,
            multi_groups=args.multi_groups,
            multi_per_group=args.multi_per_group,
            no_policy_per_complexity=args.no_policy_per_complexity,
            language=args.language,
            preferred_model=args.model,
            streaming_callback=streaming_callback,
            generation_offset=generation_offset,
            section_offsets=section_offsets,
            multi_custom_groups=SEMANTIC_MULTI_POLICY_GROUPS,
        )
    finally:
        f.close()

    final_total = stats["total"] + total_count
    print(f"\nGeneration complete:")
    print(f"  New queries: {total_count}")
    print(f"  Total in file: {final_total}")
    print(f"  New stats: {json.dumps(new_stats, ensure_ascii=False, indent=2)}")

    if total_count == 0 and stats["total"] > 0:
        print(f"\nNote: No new queries generated. File already has {stats['total']} queries.")
        print(f"  Use --force-restart to regenerate from scratch.")

    return 0 if total_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
