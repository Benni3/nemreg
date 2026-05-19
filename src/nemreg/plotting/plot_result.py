from __future__ import annotations

from typing import Optional, Any, Literal
import numpy as np
import matplotlib.pyplot as plt

from nemreg.core.dataset import Dataset
from nemreg.core.result import FitResult


def _params_one_line(result: FitResult, digits: int = 4) -> str:
    parts = []
    for name, val in zip(result.param_names, result.popt):
        parts.append(f"{name}={val:.{digits}g}")
    return ", ".join(parts)


def plot_result(
    result: FitResult,
    *,
    ax: Optional[plt.Axes] = None,
    dataset: Optional[Dataset] = None,
    mode: Literal["fit", "residuals", "both"] = "fit",
    n_grid: int = 400,
    grid: bool = True,
    legend: bool = True,
    title: Optional[str] = None,
    **kwargs: Any,
):
    """
    Plot ONLY the result.

    mode:
      - "fit": fitted curve (smooth if dataset is 1D; else yhat vs x[:,0] at observed points)
      - "residuals": residuals vs index
      - "both": residuals vs x[:,0] if dataset provided, else vs index
    """


    kwargs.pop("legend", None)
    kwargs.pop("grid", None)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    r2v = result.stats.get("r2", float("nan"))
    rmsev = result.stats.get("rmse", float("nan"))

    if mode == "fit":
        if dataset is None:
            # fallback: yhat vs index
            yhat = np.asarray(result.yhat, float).ravel()
            ax.plot(np.arange(yhat.size), yhat, **kwargs)
            ax.set_xlabel("index")
            ax.set_ylabel("y_hat")
            if title is None:
                title = f"{result.model_name} (fit) | R²={r2v:.4g}\n{_params_one_line(result)}"
            ax.set_title(title)
        else:
            X = np.asarray(dataset.x, float)
            if X.ndim != 2:
                raise ValueError("plot_result(): dataset.x must be (n,d)")
            x_disp = X[:, 0]
            n, d = X.shape

            if d == 1:
                xg = np.linspace(np.min(x_disp), np.max(x_disp), int(n_grid))
                yg = result.predict(xg)
                ax.plot(xg, yg, **kwargs)
            else:
                # observed-point curve
                yhat = np.asarray(result.yhat, float).ravel()
                if yhat.size != n:
                    yhat = result.predict_from_dataset_x(dataset)
                order = np.argsort(x_disp)
                ax.plot(x_disp[order], yhat[order], **kwargs)

            ax.set_xlabel(dataset.xlabel)
            ax.set_ylabel(dataset.ylabel)

            if title is None:
                title = f"{result.model_name} (fit) | R²={r2v:.4g}\n{_params_one_line(result)}"
                if d > 1:
                    title += f"\n(showing x[:,0] of d={d})"
            ax.set_title(title)

    elif mode == "residuals":
        resid = np.asarray(result.residuals, float).ravel()
        ax.axhline(0.0)
        ax.plot(np.arange(resid.size), resid, **kwargs)
        ax.set_xlabel("index")
        ax.set_ylabel("residual (y - yhat)")
        if title is None:
            title = f"{result.model_name} residuals\nRMSE={rmsev:.4g}"
        ax.set_title(title)

    elif mode == "both":
        resid = np.asarray(result.residuals, float).ravel()
        ax.axhline(0.0)
        if dataset is not None:
            X = np.asarray(dataset.x, float)
            if X.ndim != 2:
                raise ValueError("plot_result(): dataset.x must be (n,d)")
            x_disp = X[:, 0]
            order = np.argsort(x_disp)
            ax.plot(x_disp[order], resid[order], **kwargs)
            ax.set_xlabel(dataset.xlabel)
        else:
            ax.plot(np.arange(resid.size), resid, **kwargs)
            ax.set_xlabel("index")

        ax.set_ylabel("residual (y - yhat)")
        if title is None:
            title = f"{result.model_name} residuals | R²={r2v:.4g}\n{_params_one_line(result)}"
        ax.set_title(title)

    else:
        raise ValueError("plot_result(): mode must be 'fit', 'residuals', or 'both'")

    if grid:
        ax.grid(True)
    if legend:
        ax.legend()

    return fig, ax