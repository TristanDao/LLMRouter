#!/usr/bin/env python3
"""
scripts/train_mfrouter.py - Train MFRouter for safety routing (Colab-friendly).

Loads routing data + embeddings from artifacts/routing/, trains BilinearMF,
saves model to saved_models/mfrouter/.

Usage:
    python scripts/train_mfrouter.py \
        --config configs/model_config_train/mfrouter.yaml \
        --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llmrouter.models.mfrouter import MFRouter
from llmrouter.models.mfrouter.trainer import MFRouterTrainer


def patch_init_for_broken_imports() -> None:
    """Wrap broken imports so they don't break MFRouter load."""
    utils_init = PROJECT_ROOT / "llmrouter" / "utils" / "__init__.py"
    if utils_init.exists():
        content = utils_init.read_text()
        if "load_model" not in content and "save_model" not in content:
            content = content.replace(
                "from .setup import setup_environment",
                "try:\n    from .model_loader import save_model, load_model\n"
                "except Exception:\n    save_model = None\n    load_model = None\n\n"
                "try:\n    from .evaluation import calculate_task_performance\n"
                "except Exception:\n    calculate_task_performance = None\n\n"
                "try:\n    from .embeddings import get_longformer_embedding\n"
                "except Exception:\n    get_longformer_embedding = None\n\n"
                "from .setup import setup_environment",
            )
            content = content.replace(
                '"load_pt",\n    "setup_environment",',
                '"load_pt",\n    "get_longformer_embedding",\n    "load_model",\n    "save_model",\n    "calculate_task_performance",\n    "setup_environment",',
            )
            utils_init.write_text(content)
            print(f"  Patched {utils_init.name} to export load_model, save_model, calculate_task_performance, get_longformer_embedding")

    models_init = PROJECT_ROOT / "llmrouter" / "models" / "__init__.py"
    if not models_init.exists():
        return
    content = models_init.read_text()
    old = "from .smallest_llm import SmallestLLM\nfrom .largest_llm import LargestLLM\n"
    if old not in content:
        return
    new = (
        "try:\n    from .smallest_llm import SmallestLLM\n"
        "except Exception:\n    SmallestLLM = None\n"
        "try:\n    from .largest_llm import LargestLLM\n"
        "except Exception:\n    LargestLLM = None\n"
    )
    models_init.write_text(content.replace(old, new))
    print(f"  Patched {models_init.name} for broken smallest_llm import")


def evaluate_on_test(
    trainer: MFRouterTrainer,
    router: MFRouter,
    device: str,
) -> dict:
    """Compute routing accuracy on test set."""
    test_df = router.routing_data_test
    if test_df is None or len(test_df) == 0:
        return {"accuracy": None, "n": 0}

    model = trainer.model
    model.eval()
    model = model.to(device)

    grouped = test_df.groupby("query")
    correct = 0
    total = 0
    with torch.no_grad():
        for query, group in grouped:
            eids = [int(eid) for eid in group["embedding_id"]]
            perfs = group["performance"].tolist()
            actual_models = group["model_name"].tolist()
            best_idx = perfs.index(max(perfs))
            best_actual = actual_models[best_idx]

            emb = router.query_embedding_data[eids[0]].to(device).unsqueeze(0)
            q_proj = model.project_text(emb)
            scores = model.score_all(q_proj)
            predicted_idx = int(torch.argmax(scores).item())
            predicted = router.idx_to_model[predicted_idx]

            if predicted == best_actual:
                correct += 1
            total += 1

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "n": total,
        "correct": correct,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train MFRouter for safety routing",
    )
    parser.add_argument(
        "--config", default="configs/model_config_train/mfrouter.yaml",
        help="Path to mfrouter yaml config",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--no-eval", action="store_true", help="Skip test-set evaluation after training",
    )
    args = parser.parse_args()

    patch_init_for_broken_imports()

    print("=" * 60)
    print("MFRouter Training")
    print("=" * 60)
    print(f"  Config: {args.config}")
    print(f"  Device: {args.device}")

    router = MFRouter(yaml_path=args.config)
    print(f"\n  Models: {router.idx_to_model}")
    print(f"  Training pairs: {len(router.pairs)}")
    print(f"  Embedding dim: {router.text_dim}")
    print(f"  Latent dim: {router.dim}")

    n_test = len(router.routing_data_test) if router.routing_data_test is not None else 0
    print(f"  Test rows: {n_test}")

    trainer = MFRouterTrainer(router=router, device=args.device)
    print(f"\n  Save path: {trainer.save_model_path}")
    print(f"  Epochs: {trainer.epochs}")
    print(f"  Batch size: {trainer.batch_size}")
    print(f"  Learning rate: {trainer.lr}")
    print()

    trainer.train()

    if not args.no_eval and router.routing_data_test is not None:
        print("\n" + "=" * 60)
        print("Test-set evaluation")
        print("=" * 60)
        metrics = evaluate_on_test(trainer, router, args.device)
        if metrics["n"] > 0:
            print(f"  Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['n']})")
        else:
            print("  No test data")

    print(f"\n  Model saved to: {trainer.save_model_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
