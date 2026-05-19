"""
nemreg public API (stable entry point)

All user-facing operations route through Session to keep one canonical pipeline:
User -> Session -> Dataset -> Model -> Fitter -> FitResult -> User
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Any, Dict

from nemreg.core.session import Session
from nemreg.core.dataset import Dataset, DatasetCollection
from nemreg.core.model import Model
from nemreg.core.result import FitResult

from nemreg.fitters.levenberg_marquardt import LMOptions

from Test.nemreg.src.nemreg.plotting.dispatcher import plot

# expose models namespace: nemreg.api.models.gaussian(), etc.
from nemreg import models


def new_session(*, options: Optional[LMOptions] = None) -> Session:
    """
    Create a new Session with optional default LMOptions.
    """
    s = Session()
    if options is not None:
        s.default_lm_options = options
    return s


def fit(
    dataset: Dataset,
    model: Model,
    *,
    p0: Optional[Sequence[float]] = None,
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    options: Optional[LMOptions] = None,
    store: bool = False,
) -> FitResult:
    """
    Convenience one-shot fit that still routes through Session.

    store=False by default to keep it "stateless"; set store=True if you want history.
    """
    s = new_session(options=options)
    s.add_dataset(dataset, set_active=True)
    s.use_model(model)
    return s.fit(p0=p0, bounds=bounds, options=options, store=store)


def fit_named(
    session: Session,
    dataset_name: str,
    model: Optional[Model] = None,
    *,
    p0: Optional[Sequence[float]] = None,
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    options: Optional[LMOptions] = None,
    store: bool = True,
) -> FitResult:
    """
    Fit a named dataset inside an existing Session.
    """
    session.use_dataset(dataset_name)
    if model is not None:
        session.use_model(model)
    return session.fit(p0=p0, bounds=bounds, options=options, store=store)


def fit_collection(
    collection: DatasetCollection,
    model: Model,
    *,
    p0: Optional[Sequence[float]] = None,
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    options: Optional[LMOptions] = None,
    store: bool = False,
) -> Dict[str, FitResult]:
    """
    One-shot fit for a DatasetCollection, routing through Session.
    Returns {curve_name: FitResult}
    """
    s = new_session(options=options)
    s.add_collection(collection, set_active=True)
    s.use_model(model)
    return s.fit_collection(p0=p0, bounds=bounds, options=options, store=store)