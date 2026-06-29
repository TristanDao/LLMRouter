#!/usr/bin/env python3
"""
scripts/test_mfrouter.py - Test trained MFRouter with bge-m3 embeddings.

Loads the saved MFRouter model, encodes queries with BAAI/bge-m3 (same as training),
and runs inference. Use this to verify the trained model on custom queries.

Usage:
    # Test default 5 sample queries
    python scripts/test_mfrouter.py \
        --config configs/model_config_train/mfrouter.yaml \
        --model-path saved_models/mfrouter/mfrouter_vni.pkl \
        --device cuda

    # Test a single custom query
    python scripts/test_mfrouter.py \
        --config configs/model_config_train/mfrouter.yaml \
        --model-path saved_models/mfrouter/mfrouter_vni.pkl \
        --device cuda \
        --text "Vin nô đúng là lũ ngu xuẩn."

    # Test multiple custom queries
    python scripts/test_mfrouter.py \
        --config configs/model_config_train/mfrouter.yaml \
        --model-path saved_models/mfrouter/mfrouter_vni.pkl \
        --device cuda \
        --text "query 1" \
        --text "query 2" \
        --text "query 3"

    # Test queries from a file (one per line)
    python scripts/test_mfrouter.py \
        --config configs/model_config_train/mfrouter.yaml \
        --model-path saved_models/mfrouter/mfrouter_vni.pkl \
        --device cuda \
        --text-file my_queries.txt

    # Evaluate on test set (default if no --text)
    python scripts/test_mfrouter.py \
        --config configs/model_config_train/mfrouter.yaml \
        --model-path saved_models/mfrouter/mfrouter_vni.pkl \
        --device cuda \
        --eval-test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llmrouter.models.mfrouter.router import BilinearMF
from llmrouter.utils.model_loader import load_model


DEFAULT_TEST_QUERIES = [
    "Vin nô đúng là lũ ngu xuẩn.",
    "Chiến lược phát triển xe điện VinFast trong 5 năm tới là gì?",
    "Bạn có thể kể tên các thương hiệu thuộc Vingroup không?",
    "Ông Phạm Nhật Vượng hiện giữ chức vụ gì?",
    "Tôi đang tìm hiểu về chiến lược kinh doanh của Vingroup.",
]


def load_bge_m3(model_name: str, device: str):
    from transformers import AutoModel, AutoTokenizer

    print(f"  Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()
    return tokenizer, model


def encode_bge_m3(
    texts: List[str],
    tokenizer,
    model,
    device: str,
    max_length: int = 512,
) -> torch.Tensor:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        hidden = model(**encoded).last_hidden_state
    pooled = hidden[:, 0]
    return torch.nn.functional.normalize(pooled, p=2, dim=1).cpu()


def load_mf_model(
    model_path: str,
    num_models: int,
    text_dim: int,
    latent_dim: int,
):
    print(f"  Loading MF model from {model_path}...")
    state_dict = load_model(model_path)
    model = BilinearMF(dim=latent_dim, num_models=num_models, text_dim=text_dim)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_model_to_idx(routing_path: str) -> tuple:
    import pandas as pd
    if not os.path.exists(routing_path):
        raise FileNotFoundError(f"Routing data not found: {routing_path}")
    df = pd.read_json(routing_path, lines=True)
    models = df["model_name"].unique().tolist()
    return {m: i for i, m in enumerate(models)}, models


def route_queries(
    queries: List[str],
    mf_model,
    tokenizer,
    emb_model,
    idx_to_model: List[str],
    device: str,
) -> List[dict]:
    if not queries:
        return []

    print(f"\n  Encoding {len(queries)} query(ies)...")
    embeddings = encode_bge_m3(queries, tokenizer, emb_model, device)

    results = []
    with torch.no_grad():
        for q, emb in zip(queries, embeddings):
            emb = emb.to(device).unsqueeze(0)
            q_proj = mf_model.project_text(emb)
            scores = mf_model.score_all(q_proj)
            best_idx = int(torch.argmax(scores).item())
            predicted = idx_to_model[best_idx]
            all_scores = scores.cpu().tolist()
            results.append({
                "query": q,
                "predicted_model": predicted,
                "scores": dict(zip(idx_to_model, [round(s, 4) for s in all_scores])),
            })
    return results


def print_results(results: List[dict]) -> None:
    print("\n" + "=" * 60)
    print(f"Routing Results ({len(results)} query/ies)")
    print("=" * 60)
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] Query: {r['query']}")
        print(f"      → Routed to: {r['predicted_model']}")
        print(f"      → Scores:    {r['scores']}")


def evaluate_on_test(
    mf_model,
    tokenizer,
    emb_model,
    idx_to_model: List[str],
    device: str,
) -> None:
    import pandas as pd
    test_path = PROJECT_ROOT / "artifacts" / "routing" / "query_data_test.jsonl"
    if not test_path.exists():
        print(f"\n  Skipping test-set evaluation ({test_path.name} not found)")
        return

    print("\n" + "=" * 60)
    print("Test-set evaluation")
    print("=" * 60)

    test_df = pd.read_json(test_path, lines=True)
    emb_path = PROJECT_ROOT / "artifacts" / "routing" / "query_embeddings.pt"
    routing_test_path = PROJECT_ROOT / "artifacts" / "routing" / "routing_data_test.jsonl"
    all_embs = torch.load(emb_path, map_location=device)
    routing_test = pd.read_json(routing_test_path, lines=True)

    correct = 0
    total = 0
    per_query = []
    with torch.no_grad():
        for _, row in test_df.iterrows():
            q_text = row["query"]
            eid = row["embedding_id"]
            q_emb = all_embs[eid].to(device).unsqueeze(0)
            q_proj = mf_model.project_text(q_emb)
            scores = mf_model.score_all(q_proj)
            best_idx = int(torch.argmax(scores).item())
            predicted = idx_to_model[best_idx]

            query_rows = routing_test[routing_test["query_id"] == row["query_id"]]
            if len(query_rows) == 0:
                continue
            perfs = query_rows["performance"].tolist()
            models = query_rows["model_name"].tolist()
            best_actual = models[perfs.index(max(perfs))]

            is_correct = predicted == best_actual
            if is_correct:
                correct += 1
            total += 1
            per_query.append({
                "query": q_text[:60],
                "predicted": predicted,
                "expected": best_actual,
                "correct": is_correct,
            })

    accuracy = correct / total if total > 0 else 0.0
    print(f"\n  Top-1 accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"\n  Sample predictions:")
    for p in per_query[:10]:
        mark = "✅" if p["correct"] else "❌"
        print(f"    {mark} {p['query']} → {p['predicted']} (expected: {p['expected']})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test trained MFRouter with bge-m3 embeddings",
    )
    parser.add_argument(
        "--config", default="configs/model_config_train/mfrouter.yaml",
        help="Path to mfrouter yaml config",
    )
    parser.add_argument(
        "--model-path", default="saved_models/mfrouter/mfrouter_vni.pkl",
        help="Path to trained MFRouter .pkl file",
    )
    parser.add_argument(
        "--embedding-model", default="BAAI/bge-m3",
        help="HF model for query encoding (must match training)",
    )
    parser.add_argument(
        "--routing-data", default="artifacts/routing/routing_data_train.jsonl",
        help="Routing data file to derive model order",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--text-dim", type=int, default=1024, help="Must match training (1024 for bge-m3)",
    )
    parser.add_argument(
        "--latent-dim", type=int, default=128,
    )
    parser.add_argument(
        "--text", action="append", default=None,
        help="Custom query text to route. Can be passed multiple times.",
    )
    parser.add_argument(
        "--text-file", default=None,
        help="Path to file with one query per line",
    )
    parser.add_argument(
        "--eval-test", action="store_true",
        help="Also evaluate on test set (query_data_test.jsonl)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"ERROR: Model not found at {args.model_path}")
        print("  Run train_mfrouter.py first.")
        return 1

    model_to_idx, idx_to_model = load_model_to_idx(args.routing_data)
    print(f"  Models: {idx_to_model}")

    mf_model = load_mf_model(
        args.model_path,
        num_models=len(idx_to_model),
        text_dim=args.text_dim,
        latent_dim=args.latent_dim,
    )
    mf_model = mf_model.to(args.device)

    tokenizer, emb_model = load_bge_m3(args.embedding_model, args.device)

    queries = []
    if args.text:
        queries.extend(args.text)
    if args.text_file:
        if not os.path.exists(args.text_file):
            print(f"ERROR: Text file not found: {args.text_file}")
            return 1
        with open(args.text_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    queries.append(line)

    if queries:
        results = route_queries(queries, mf_model, tokenizer, emb_model, idx_to_model, args.device)
        print_results(results)
    else:
        print("\n  No --text or --text-file provided, using 5 default sample queries")
        results = route_queries(
            DEFAULT_TEST_QUERIES, mf_model, tokenizer, emb_model, idx_to_model, args.device
        )
        print_results(results)

    if args.eval_test:
        evaluate_on_test(mf_model, tokenizer, emb_model, idx_to_model, args.device)

    return 0


if __name__ == "__main__":
    sys.exit(main())
