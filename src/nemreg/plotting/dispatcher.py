# nemreg/plotting/dispatcher.py
from __future__ import annotations

from typing import Any, Literal, Optional
import matplotlib.pyplot as plt

from nemreg.core.dataset import Dataset
from nemreg.core.result import FitResult

from nemreg.plotting.plot_data import plot_data
from nemreg.plotting.plot_result_data import plot_result_data
from nemreg.plotting.plot_result import plot_result

try:
    from nemreg.plotting.multivariable_plot import multivariable_plot
except Exception:
    multivariable_plot = None

PlotMode = Literal["data", "result_data", "result", "multivariable"]


def plot(
    *,
    mode: PlotMode = "data",
    dataset: Optional[Dataset] = None,
    result: Optional[FitResult] = None,
    ax: Optional[plt.Axes] = None,
    grid: bool = True,
    legend: bool = True,
    debug_kwargs: bool = False,
    **kwargs: Any,
):
    """
    Dispatcher for plotting.
    """

    # Allow user to pass grid/legend in kwargs too; dispatcher param wins unless user explicitly set it in kwargs.
    grid = bool(kwargs.pop("grid", grid))
    legend = bool(kwargs.pop("legend", legend))

    # Very common option for plot_data
    show_errors = kwargs.pop("show_errors", True)

    if debug_kwargs:
        print(f"[nr.plot] mode={mode}")
        print(f"[nr.plot] grid={grid} legend={legend} show_errors={show_errors}")
        print(f"[nr.plot] kwargs keys={sorted(kwargs.keys())}")
        print(f"[nr.plot] kwargs={kwargs}")

    if mode == "data":
        if dataset is None:
            raise ValueError("plot(mode='data') requires dataset=")
        return plot_data(
            dataset,
            ax=ax,
            grid=grid,
            legend=legend,
            show_errors=show_errors,
            **kwargs,
        )

    if mode == "result_data":
        if dataset is None or result is None:
            raise ValueError("plot(mode='result_data') requires dataset= and result=")
        return plot_result_data(
            dataset,
            result,
            ax=ax,
            grid=grid,
            legend=legend,
            **kwargs,
        )

    if mode == "result":
        if result is None:
            raise ValueError("plot(mode='result') requires result=")

        if "plot_mode" in kwargs and "mode" not in kwargs:
            kwargs["mode"] = kwargs.pop("plot_mode")

        return plot_result(
            result,
            dataset=dataset,
            ax=ax,
            grid=grid,
            legend=legend,
            **kwargs,
        )

    if mode == "multivariable":
        if multivariable_plot is None:
            raise ImportError("multivariable_plot is not available (import failed)")
        if dataset is None:
            raise ValueError("plot(mode='multivariable') requires dataset=")

        if "plot_mode" in kwargs and "mode" not in kwargs:
            kwargs["mode"] = kwargs.pop("plot_mode")

        return multivariable_plot(
            dataset,
            result=result,
            ax=ax,
            grid=grid,
            legend=legend,
            **kwargs,
        )

    raise ValueError(f"Unknown plot mode: {mode!r}")