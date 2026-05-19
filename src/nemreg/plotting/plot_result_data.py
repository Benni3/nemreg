from __future__ import annotations

from typing import Optional, Any
import numpy as np
import matplotlib.pyplot as plt

from nemreg.core.dataset import Dataset
from nemreg.core.result import FitResult


def _axis_label(label: str, unit: str) -> str:
    unit = (unit or "").strip()
    return f"{label} [{unit}]" if unit else label


def _params_one_line(result: FitResult, digits: int = 4) -> str:
    parts = []
    for name, val in zip(result.param_names, result.popt):
        parts.append(f"{name}={val:.{digits}g}")
    return ", ".join(parts)


def plot_result_data(
    dataset: Dataset,
    result: FitResult,
    *,
    ax: Optional[plt.Axes] = None,
    n_grid: int = 400,
    show_errors: bool = True,
    grid: bool = True,
    legend: bool = True,
    data_marker: str = "o",
    fit_linewidth: float = 2.0,
    title: Optional[str] = None,
    **kwargs: Any,
):
    """
    Plot dataset + fitted curve.

    Uses result.predict(...) via the stored model.
    - If dataset is 1D (d==1): draws a smooth curve over a dense x-grid.
    - If dataset is multivariable (d>1): overlays fitted yhat at original points
      and draws a line vs x[:,0] (sorted) for display.
    """

    kwargs.pop("legend", None)
    kwargs.pop("grid", None)
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    X = np.asarray(dataset.x, float)
    y = np.asarray(dataset.y, float).ravel()

    if X.ndim != 2:
        raise ValueError(f"plot_result_data(): dataset.x must be (n,d). Got {X.shape}")

    n, d = X.shape
    if y.size != n:
        raise ValueError("plot_result_data(): y length must match x rows")

    # display axis
    x_disp = X[:, 0]

    # errorbars
    xerr_disp = None
    yerr = None
    if show_errors:
        if getattr(dataset, "xerr", None) is not None:
            Xe = np.asarray(dataset.xerr, float)
            if Xe.shape == X.shape:
                xerr_disp = Xe[:, 0]
        if getattr(dataset, "yerr", None) is not None:
            yerr = np.asarray(dataset.yerr, float).ravel()

    # plot data
    if show_errors and (xerr_disp is not None or yerr is not None):
        ax.errorbar(x_disp, y, xerr=xerr_disp, yerr=yerr, fmt=data_marker, linestyle="None", **kwargs)
    else:
        ax.plot(x_disp, y, marker=data_marker, linestyle="None", **kwargs)

    # plot fit
    if d == 1:
        xg = np.linspace(np.min(x_disp), np.max(x_disp), int(n_grid))
        yg = result.predict(xg)
        ax.plot(xg, yg, linewidth=fit_linewidth)
    else:
        # for multivariable, we can only show yhat at the observed points
        yhat = np.asarray(result.yhat, float).ravel()
        if yhat.size != n:
            # safe fallback: recompute at dataset x
            yhat = result.predict_from_dataset_x(dataset)

        order = np.argsort(x_disp)
        ax.plot(x_disp[order], yhat[order], linewidth=fit_linewidth)

    # labels
    ax.set_xlabel(_axis_label(dataset.xlabel, dataset.xunit))
    ax.set_ylabel(_axis_label(dataset.ylabel, dataset.yunit))

    # title
    if title is None:
        r2v = result.stats.get("r2", float("nan"))
        title = f"{dataset.name} | {result.model_name} | R²={r2v:.4g}\n{_params_one_line(result)}"
        if d > 1:
            title += f"\n(showing x[:,0] of d={d})"
    ax.set_title(title)

    if grid:
        ax.grid(True)
    if legend:
        ax.legend()

    return fig, ax