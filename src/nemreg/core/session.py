# nemreg/core/session.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Dict, Any, List, Union

import numpy as np

from nemreg.core.dataset import Dataset, DatasetCollection
from nemreg.core.model import Model
from nemreg.core.result import FitResult
from nemreg.fitters.levenberg_marquardt import lm_fit, LMOptions


@dataclass
class Session:
    """
    High-level user entry point.

    Flow:
      User -> Session -> Dataset -> Model -> Fitter -> FitResult -> User

    Session stores:
      - datasets (named)
      - last dataset/model/result
      - history of fits
    """

    datasets: Dict[str, Union[Dataset, DatasetCollection]] = field(default_factory=dict)
    history: List[FitResult] = field(default_factory=list)

    active_dataset: Optional[str] = None
    active_model: Optional[Model] = None
    last_result: Optional[FitResult] = None

    # default fitter options
    default_lm_options: LMOptions = field(default_factory=LMOptions)

    # -----------------------
    # Dataset management
    # -----------------------

    def add_dataset(self, dataset: Dataset, *, name: Optional[str] = None, set_active: bool = True) -> str:
        ds_name = name or dataset.name
        if not ds_name:
            raise ValueError("Session.add_dataset(): dataset name is empty")
        self.datasets[ds_name] = dataset
        if set_active:
            self.active_dataset = ds_name
        return ds_name

    def add_collection(self, collection: DatasetCollection, *, name: Optional[str] = None, set_active: bool = False) -> str:
        col_name = name or collection.name
        if not col_name:
            raise ValueError("Session.add_collection(): collection name is empty")
        self.datasets[col_name] = collection
        if set_active:
            self.active_dataset = col_name
        return col_name

    def get_dataset(self, name: Optional[str] = None) -> Union[Dataset, DatasetCollection]:
        ds_name = name or self.active_dataset
        if ds_name is None:
            raise ValueError("Session.get_dataset(): no active dataset set")
        if ds_name not in self.datasets:
            raise KeyError(f"Session.get_dataset(): unknown dataset '{ds_name}'")
        return self.datasets[ds_name]

    def use_dataset(self, name: str) -> "Session":
        if name not in self.datasets:
            raise KeyError(f"Session.use_dataset(): unknown dataset '{name}'")
        self.active_dataset = name
        # reset last result when switching active dataset
        self.last_result = None
        return self

    def list_datasets(self) -> List[str]:
        return list(self.datasets.keys())

    # -----------------------
    # Model management
    # -----------------------

    def use_model(self, model: Model) -> "Session":
        self.active_model = model
        self.last_result = None
        return self

    # -----------------------
    # Fit
    # -----------------------

    def fit(
        self,
        *,
        dataset: Optional[str] = None,
        model: Optional[Model] = None,
        p0: Optional[Sequence[float]] = None,
        bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        options: Optional[LMOptions] = None,
        store: bool = True,
    ) -> FitResult:
        """
        Fit active (or provided) dataset with active (or provided) model using LM.

        Parameters
        ----------
        dataset : optional dataset name (uses active if None)
        model   : optional Model (uses active if None)
        p0      : initial guess overrides model.guess
        bounds  : optional (lower, upper) bounds (clamped candidate parameters)
        options : optional LMOptions (defaults to session default)
        store   : if True, store in history and last_result

        Returns
        -------
        FitResult
        """
        ds = self.get_dataset(dataset)

        if isinstance(ds, DatasetCollection):
            raise TypeError(
                "Session.fit(): active dataset is a DatasetCollection. "
                "Use Session.fit_collection() or select a single Dataset."
            )

        m = model or self.active_model
        if m is None:
            raise ValueError("Session.fit(): no model provided and no active model set")

        opt = options or self.default_lm_options

        res = lm_fit(ds, m, p0=p0, bounds=bounds, options=opt)

        if store:
            self.last_result = res
            self.history.append(res)
            self.active_model = m

        return res

    def fit_collection(
        self,
        collection_name: Optional[str] = None,
        *,
        model: Optional[Model] = None,
        p0: Optional[Sequence[float]] = None,
        bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        options: Optional[LMOptions] = None,
        store: bool = True,
    ) -> Dict[str, FitResult]:
        """
        Fit each Dataset in a DatasetCollection with the same model.
        Returns dict: dataset_name -> FitResult
        """
        ds = self.get_dataset(collection_name)
        if not isinstance(ds, DatasetCollection):
            raise TypeError("Session.fit_collection(): selected item is not a DatasetCollection")

        m = model or self.active_model
        if m is None:
            raise ValueError("Session.fit_collection(): no model provided and no active model set")

        opt = options or self.default_lm_options

        results: Dict[str, FitResult] = {}
        for d in ds.datasets:
            res = lm_fit(d, m, p0=p0, bounds=bounds, options=opt)
            results[d.name] = res
            if store:
                self.history.append(res)

        if store and results:
            # last_result becomes the last fit in the loop
            self.last_result = list(results.values())[-1]
            self.active_model = m

        return results

    # -----------------------
    # Convenience helpers
    # -----------------------

    def summary(self) -> str:
        parts = []
        parts.append(f"Session: {len(self.datasets)} dataset(s), {len(self.history)} fit(s)")
        if self.active_dataset is not None:
            parts.append(f"Active dataset: {self.active_dataset}")
        if self.active_model is not None:
            parts.append(f"Active model: {getattr(self.active_model, 'name', 'model')}")
        if self.last_result is not None:
            parts.append("Last result:")
            parts.append(self.last_result.summary())
        return "\n".join(parts)

    def clear_history(self) -> None:
        self.history.clear()
        self.last_result = None

    def last(self) -> FitResult:
        if self.last_result is None:
            raise ValueError("Session.last(): no previous fit result")
        return self.last_result