"""
nemreg - lightweight regression + plotting helpers

Keep __init__ thin to avoid circular imports and import-time side effects.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "Dataset",
    "DatasetCollection",
    "Model",
    "FitResult",
    "fit",
    "plot",
    "Session",
    "models",
]


def __getattr__(name: str) -> Any:
    # core types
    if name == "Dataset":
        from nemreg.core.dataset import Dataset
        return Dataset
    if name == "DatasetCollection":
        from nemreg.core.dataset import DatasetCollection
        return DatasetCollection
    if name == "Model":
        from nemreg.core.model import Model
        return Model
    if name == "FitResult":
        from nemreg.core.result import FitResult
        return FitResult

    # high-level functions
    if name == "fit":
        from nemreg.core.fit import fit
        return fit
    if name == "plot":
        from nemreg.plotting.dispatcher import plot
        return plot

    # session
    if name == "Session":
        from nemreg.core.session import Session
        return Session

    # models namespace (lazy)
    if name == "models":
        import nemreg.models as models
        return models

    raise AttributeError(f"module 'nemreg' has no attribute {name!r}")