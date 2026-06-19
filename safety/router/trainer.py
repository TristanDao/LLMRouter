try:  # pragma: no cover
    from safety_routers.trainer import SafetyRouterTrainer
except Exception:  # pragma: no cover
    SafetyRouterTrainer = None

__all__ = ["SafetyRouterTrainer"]
