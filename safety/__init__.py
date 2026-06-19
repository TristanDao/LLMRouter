"""Canonical safety namespace for the safety / golden dataset pipeline."""

try:  # pragma: no cover
    from .router import SafetyRouter
except Exception:  # pragma: no cover
    SafetyRouter = None

try:  # pragma: no cover
    from .router import SafetyRouterTrainer
except Exception:  # pragma: no cover
    SafetyRouterTrainer = None

try:  # pragma: no cover
    from .dataset import SafetyGoldenDatasetBuilder, summarize_records
except Exception:  # pragma: no cover
    SafetyGoldenDatasetBuilder = None
    summarize_records = None

__all__ = [
    "SafetyRouter",
    "SafetyRouterTrainer",
    "SafetyGoldenDatasetBuilder",
    "summarize_records",
]
