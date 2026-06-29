#!/usr/bin/env python3
"""
scripts/convert_to_routing_ds.py - Convert golden dataset splits to MFRouter routing format.

Two-phase operation:

  Phase 1 (local -- no GPU/torch needed):
      Convert JSONL splits → routing_data + query_data JSONL files.
      Also writes a text file listing all unique query texts for Phase 2.

  Phase 2 (Colab -- needs torch + httpx):
      Reads the query text file, calls Alibaba text-embedding-v3 API,
      saves query_embeddings.pt, and backfills embedding_id into routing data.

Usage:
    # Phase 1: Convert (local, no torch)
    python scripts/convert_to_routing_ds.py \
        --input-dir artifacts/golden/splits_vni \
        --output-dir artifacts/routing \
        --skip-embeddings

    # Phase 2: Embeddings (Colab)
    python scripts/convert_to_routing_ds.py \
        --output-dir artifacts/routing \
        --embeddings-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safety.common.env import load_dotenv_file, env_str

load_dotenv_file(PROJECT_ROOT / ".env")

MODEL_NAMES = ["local", "gemini"]
_QUERY_TEXTS_FILE = "unique_query_texts.txt"


def _maybe_import_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def _maybe_import_httpx():
    try:
        import httpx
        return httpx
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def read_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, records: List[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_lines(path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Record helpers
# ---------------------------------------------------------------------------

def get_query_text(record: dict) -> str:
    return str(record.get("query") or record.get("user_prompt") or "")


def should_include(record: dict) -> bool:
    difficulty = record.get("difficulty", "")
    if difficulty not in ("easy", "hard"):
        return False
    consensus = record.get("judge_consensus", "")
    if consensus in ("fail",):
        return False
    return True


def compute_performance(difficulty: str, model_name: str) -> float:
    if difficulty == "easy":
        return 1.0
    elif difficulty == "hard":
        return 0.0 if model_name == "local" else 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Build routing data
# ---------------------------------------------------------------------------

def build_routing_records(records: List[dict]) -> List[dict]:
    routing = []
    for rec in records:
        if not should_include(rec):
            continue
        query_text = get_query_text(rec)
        query_id = rec.get("query_id", "")
        difficulty = rec.get("difficulty", "")
        for model_name in MODEL_NAMES:
            row = {
                "query": query_text,
                "query_id": query_id,
                "model_name": model_name,
                "performance": compute_performance(difficulty, model_name),
                "embedding_id": -1,  # filled in Phase 2
            }
            routing.append(row)
    return routing


def build_query_records(records: List[dict]) -> List[dict]:
    queries = []
    seen = set()
    for rec in records:
        if not should_include(rec):
            continue
        qid = rec.get("query_id", "")
        if qid in seen:
            continue
        seen.add(qid)
        queries.append({
            "query": get_query_text(rec),
            "query_id": qid,
            "task": "safety_routing",
        })
    return queries


def assign_embedding_ids(
    routing_records: List[dict],
    query_texts: List[str],
) -> None:
    query_to_eid = {}
    for eid, text in enumerate(query_texts):
        query_to_eid[text] = eid

    for row in routing_records:
        qt = row["query"]
        row["embedding_id"] = query_to_eid.get(qt, -1)


# ---------------------------------------------------------------------------
# Phase 1: Convert splits → JSONL (local, no torch)
# ---------------------------------------------------------------------------

def convert_splits(input_dir: str, output_dir: str) -> None:
    splits = ["train", "test", "dev"]

    all_routing = {}
    all_queries = {}

    for split in splits:
        input_path = os.path.join(input_dir, f"{split}.jsonl")
        if not os.path.exists(input_path):
            print(f"Skip {split}: file not found at {input_path}")
            continue

        records = read_jsonl(input_path)
        print(f"Read {len(records)} records from {input_path}")

        routing = build_routing_records(records)
        queries = build_query_records(records)

        filtered_count = len(queries)
        dropped = len(records) - filtered_count
        print(f"  {filtered_count} queries kept, {dropped} dropped (filtered)")

        all_routing[split] = routing
        all_queries[split] = queries

    # Collect unique query texts across all splits
    unique_query_texts = []
    seen_texts = set()
    for split, queries in all_queries.items():
        for q in queries:
            txt = q["query"]
            if txt not in seen_texts:
                seen_texts.add(txt)
                unique_query_texts.append(txt)
    print(f"Total unique queries across all splits: {len(unique_query_texts)}")

    # Write routing + query JSONL files (embedding_id = -1 as placeholder)
    os.makedirs(output_dir, exist_ok=True)
    for split, routing in all_routing.items():
        routing_path = os.path.join(output_dir, f"routing_data_{split}.jsonl")
        query_path = os.path.join(output_dir, f"query_data_{split}.jsonl")
        write_jsonl(routing_path, routing)
        write_jsonl(query_path, all_queries[split])
        print(f"Wrote {len(routing)} routing rows to {routing_path}")
        print(f"Wrote {len(all_queries[split])} query rows to {query_path}")

    # Write query texts file for Phase 2
    texts_path = os.path.join(output_dir, _QUERY_TEXTS_FILE)
    write_lines(texts_path, unique_query_texts)
    print(f"Wrote {len(unique_query_texts)} query texts to {texts_path}")

    print("\nPhase 1 complete. Run Phase 2 on Colab:")
    print(f"  python scripts/convert_to_routing_ds.py --output-dir {output_dir} --embeddings-only")


# ---------------------------------------------------------------------------
# Phase 2: Generate embeddings (Colab, needs torch + httpx)
# ---------------------------------------------------------------------------

def embed_alibaba(
    texts: List[str],
    model_name: str = "text-embedding-v3",
    batch_size: int = 32,
) -> List[Optional[np.ndarray]]:
    httpx = _maybe_import_httpx()
    if httpx is None:
        raise ImportError("httpx is required for Alibaba embedding API")

    api_endpoint = env_str("ALIBABA_URL", "")
    if not api_endpoint:
        raise ValueError("ALIBABA_URL is required in .env")
    api_key = env_str("ALIBABA_API_KEY", "")
    if not api_key:
        raise ValueError("ALIBABA_API_KEY is required in .env")

    embedding_url = api_endpoint.rstrip("/") + "/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    results: List[Optional[np.ndarray]] = [None] * len(texts)

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        payload = {
            "model": model_name,
            "input": batch,
            "encoding_format": "float",
        }
        success = False
        for attempt in range(3):
            try:
                response = httpx.post(
                    embedding_url,
                    json=payload,
                    headers=headers,
                    timeout=60.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    for idx, item in enumerate(data.get("data", [])):
                        emb = np.array(item["embedding"], dtype=np.float32)
                        results[i + idx] = emb
                    success = True
                    break
                else:
                    print(f"  Embedding API error (attempt {attempt+1}): "
                          f"status={response.status_code} body={response.text[:200]}")
                    import time
                    time.sleep(2.0)
            except Exception as e:
                print(f"  Embedding API exception (attempt {attempt+1}): {e}")
                import time
                time.sleep(2.0)

        if not success:
            print(f"  Failed to embed batch {i // batch_size + 1} "
                  f"after 3 attempts")

    return results


def run_embeddings_only(output_dir: str) -> None:
    torch = _maybe_import_torch()
    if torch is None:
        print("ERROR: torch is required for Phase 2. Run this on Colab with GPU.")
        print("  pip install torch")
        return 1

    texts_path = os.path.join(output_dir, _QUERY_TEXTS_FILE)
    if not os.path.exists(texts_path):
        print(f"ERROR: {texts_path} not found. Run Phase 1 first.")
        return 1

    query_texts = read_lines(texts_path)
    print(f"Read {len(query_texts)} unique query texts from {texts_path}")

    model_name = env_str("ALIBABA_EMBEDDING", "text-embedding-v3")
    print(f"Embedding model: {model_name}")

    emb_list = embed_alibaba(query_texts, model_name=model_name)

    # Build embedding dict and save .pt
    embedding_dict = {}
    fail_count = 0
    dim = 1024
    for idx, emb in enumerate(emb_list):
        if emb is not None:
            embedding_dict[idx] = torch.tensor(emb, dtype=torch.float32)
            dim = emb.shape[0]
        else:
            print(f"  WARNING: embedding {idx} failed, using zeros")
            embedding_dict[idx] = torch.zeros(dim, dtype=torch.float32)
            fail_count += 1

    pt_path = os.path.join(output_dir, "query_embeddings.pt")
    torch.save(embedding_dict, pt_path)
    print(f"Saved {len(embedding_dict)} embeddings to {pt_path} (dim={dim}, failed={fail_count})")

    # Backfill embedding_id into routing JSONL files
    splits = ["train", "test", "dev"]
    for split in splits:
        routing_path = os.path.join(output_dir, f"routing_data_{split}.jsonl")
        if not os.path.exists(routing_path):
            continue

        routing = read_jsonl(routing_path)
        query_to_eid = {}
        for eid, text in enumerate(query_texts):
            query_to_eid[text] = eid

        for row in routing:
            qt = row.get("query", "")
            row["embedding_id"] = query_to_eid.get(qt, -1)

        routing_rows = [r for r in routing if r["embedding_id"] >= 0]
        write_jsonl(routing_path, routing_rows)
        print(f"Backfilled {len(routing_rows)} rows in {routing_path}")

    # Update query JSONL with embedding_id
    for split in splits:
        query_path = os.path.join(output_dir, f"query_data_{split}.jsonl")
        if not os.path.exists(query_path):
            continue
        queries = read_jsonl(query_path)
        query_to_eid = {}
        for eid, text in enumerate(query_texts):
            query_to_eid[text] = eid
        for q in queries:
            qt = q.get("query", "")
            q["embedding_id"] = query_to_eid.get(qt, -1)
        write_jsonl(query_path, queries)
        print(f"Updated embedding_id in {query_path}")

    print(f"\nPhase 2 complete. Embedding dim: {dim}")
    print(f"Ensure mfrouter.yaml text_dim = {dim}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert golden dataset to MFRouter routing format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", "-i", default=None,
                        help="Directory with train/test/dev.jsonl (Phase 1)")
    parser.add_argument("--output-dir", "-o", required=True,
                        help="Output directory for routing data and embeddings")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Phase 1 only: convert JSONL, skip embedding generation")
    parser.add_argument("--embeddings-only", action="store_true",
                        help="Phase 2 only: generate embeddings from existing text file")
    parser.add_argument("--embedding-model", default=env_str("ALIBABA_EMBEDDING", "text-embedding-v3"),
                        help="Embedding model name")
    args = parser.parse_args()

    if args.embeddings_only:
        return run_embeddings_only(args.output_dir)

    if args.input_dir is None:
        parser.error("--input-dir is required for Phase 1")

    convert_splits(args.input_dir, args.output_dir)

    if not args.skip_embeddings:
        return run_embeddings_only(args.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
