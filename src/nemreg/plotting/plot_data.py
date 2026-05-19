from __future__ import annotations

from typing import Optional, Any
import numpy as np
import matplotlib.pyplot as plt

from nemreg.core.dataset import Dataset


def _axis_label(label: str, unit: str) -> str:
    unit = (unit or "").strip()
    return f"{label} [{unit}]" if unit else label


def plot_data(
    dataset: Dataset,
    *,
    ax: Optional[plt.Axes] = None,
    show_errors: bool = True,
    grid: bool = True,
    legend: bool = True,
    marker: str = "o",
    linestyle: str = "None",
    title: Optional[str] = None,
    **kwargs: Any,
):
    """
    Plot dataset only (optional error bars).

    If dataset has multiple x features (d>1), plots y vs x[:,0] for display.
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
        raise ValueError(f"plot_data(): dataset.x must be (n,d). Got {X.shape}")

    n, d = X.shape
    if y.size != n:
        raise ValueError("plot_data(): y length must match x rows")

    x = X[:, 0]  # display axis

    xerr = None
    yerr = None
    if show_errors:
        if getattr(dataset, "xerr", None) is not None:
            Xe = np.asarray(dataset.xerr, float)
            if Xe.shape == X.shape:
                xerr = Xe[:, 0]
        if getattr(dataset, "yerr", None) is not None:
            yerr = np.asarray(dataset.yerr, float).ravel()

    if show_errors and (xerr is not None or yerr is not None):
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=marker, linestyle=linestyle, **kwargs)
    else:
        ax.plot(x, y, marker=marker, linestyle=linestyle, **kwargs)

    ax.set_xlabel(_axis_label(dataset.xlabel, dataset.xunit))
    ax.set_ylabel(_axis_label(dataset.ylabel, dataset.yunit))

    if title is None:
        title = f"{dataset.name} (data)" if d == 1 else f"{dataset.name} (data; showing x[:,0] of d={d})"
    ax.set_title(title)

    if grid:
        ax.grid(True)
    if legend:
        ax.legend()

    return fig, ax