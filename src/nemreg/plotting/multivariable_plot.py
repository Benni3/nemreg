from __future__ import annotations

from typing import Optional, Any, Literal, Tuple
import numpy as np
import matplotlib.pyplot as plt

from nemreg.core.dataset import Dataset
from nemreg.core.result import FitResult


def multivariable_plot(
    dataset: Dataset,
    result: Optional[FitResult] = None,
    *,
    ax: Optional[plt.Axes] = None,
    kind: Literal["scatter", "surface", "both"] = "scatter",
    features: Tuple[int, int] = (0, 1),
    grid_size: int = 60,
    grid: bool = True,
    legend: bool = True,
    title: Optional[str] = None,
    alpha: float = 0.9,
    show_fit_points: bool = True,
    surface_alpha: float = 0.35,
    **kwargs: Any,
):
    """
    Plot multivariable data for d=2 or d=3 features.

    Dataset.x: (n, d), d in {2,3}
    Dataset.y: (n,)

    kind:
      - "scatter": data scatter (+ optional predicted points if result given)
      - "surface": requires result; plot fitted surface (+ data scatter)
      - "both": scatter + surface

    features=(i,j):
      which two features to plot on x/y axes if d==3.
      The remaining feature is fixed at its mean for surface plotting.
    """

    kwargs.pop("legend", None)
    kwargs.pop("grid", None)

    X = np.asarray(dataset.x, dtype=float)
    y = np.asarray(dataset.y, dtype=float).ravel()

    if X.ndim != 2:
        raise ValueError(f"multivariable_plot(): dataset.x must be (n,d). Got {X.shape}")

    n, d = X.shape
    if d < 2 or d > 3:
        raise ValueError(f"multivariable_plot(): requires d=2 or d=3. Got d={d}")
    if y.size != n:
        raise ValueError("multivariable_plot(): y length must match x rows")

    i, j = features
    if not (0 <= i < d and 0 <= j < d) or i == j:
        raise ValueError(f"multivariable_plot(): features must be two distinct indices in [0,{d-1}]")

    x1 = X[:, i]
    x2 = X[:, j]

    # Create 3D axis
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    # Always show data scatter
    ax.scatter(x1, x2, y, alpha=alpha, label="data", **kwargs)

    # If result: compute predictions at observed points (safe & consistent)
    if result is not None and show_fit_points and kind in ("scatter", "both"):
        yhat_obs = result.predict(X.T)  # (d,n) convention
        ax.scatter(
            x1, x2, yhat_obs,
            alpha=max(0.25, alpha * 0.6),
            marker="^",
            label="fit (points)"
        )

    # Surface
    if kind in ("surface", "both"):
        if result is None:
            raise ValueError("multivariable_plot(kind='surface'/'both') requires result=")

        x1g = np.linspace(x1.min(), x1.max(), int(grid_size))
        x2g = np.linspace(x2.min(), x2.max(), int(grid_size))
        X1g, X2g = np.meshgrid(x1g, x2g)

        # Build grid in full feature space (ngrid, d)
        means = X.mean(axis=0)  # (d,)
        Xg = np.tile(means, (X1g.size, 1))
        Xg[:, i] = X1g.ravel()
        Xg[:, j] = X2g.ravel()

        Yg = result.predict(Xg.T).reshape(X1g.shape)  # -> (grid,grid)

        ax.plot_surface(X1g, X2g, Yg, alpha=surface_alpha, linewidth=0, antialiased=True)

    # Labels (generic for now)
    ax.set_xlabel(f"{dataset.xlabel}_{i+1}")
    ax.set_ylabel(f"{dataset.xlabel}_{j+1}")
    ax.set_zlabel(dataset.ylabel)

    # Title
    if title is None:
        if result is None:
            title = f"{dataset.name} (multivariable scatter)"
        else:
            title = f"{dataset.name} | {result.model_name} ({kind})"
            if d == 3:
                k = ({0, 1, 2} - {i, j}).pop()
                title += f" | fixed x_{k+1}=mean"

    ax.set_title(title)

    if grid:
        ax.grid(True)

    if legend:
        ax.legend()

    return fig, ax