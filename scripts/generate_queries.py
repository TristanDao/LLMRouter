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

Note: Uses append mode - if output file exists, will continue from last query_id.
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

from safety.dataset.pipeline import SafetyGoldenDatasetBuilder


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

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    # Check if file exists and has content - get last query_id to continue from
    start_idx = 0
    if os.path.exists(args.output) and os.path.getsize(args.output) > 0:
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        query_id = record.get("query_id", "")
                        if query_id.startswith("Q"):
                            num = int(query_id[1:])
                            if num > start_idx:
                                start_idx = num
                    except:
                        pass
        print(f"  Found existing file with {start_idx} queries - will continue from Q{start_idx+1:04d}")
    
    # Open file in append mode
    f = open(args.output, "a", encoding="utf-8")
    
    # Track stats
    total_count = 0
    stats = {
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
        
        # Update stats
        stats["by_mode"][q.group_type] = stats["by_mode"].get(q.group_type, 0) + 1
        stats["by_complexity"][q.designed_complexity] = stats["by_complexity"].get(q.designed_complexity, 0) + 1
        
        # Print progress every 10 queries
        if total_count % 10 == 0:
            print(f"  Progress: {total_count} queries written (total: {start_idx})...")
    
    try:
        # Generate queries with streaming callback
        builder.build_queries_with_llm(
            single_per_policy=args.single_per_policy,
            multi_groups=args.multi_groups,
            multi_per_group=args.multi_per_group,
            no_policy_per_complexity=args.no_policy_per_complexity,
            language=args.language,
            preferred_model=args.model,
            streaming_callback=streaming_callback,
        )
    finally:
        f.close()
    
    print(f"Generated {total_count} new queries (total: {start_idx} in file)")
    print(f"Stats: {json.dumps(stats, ensure_ascii=False, indent=2)}")

    return 0 if total_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
