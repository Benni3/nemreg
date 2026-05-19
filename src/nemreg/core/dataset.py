import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Union

"""
Helper functions
"""

import numpy as np

def _err_from_reps(values_2d: np.ndarray, *, mode: str, ddof: int, global_err: bool) -> Union[np.ndarray, float]:
    """
    values_2d: shape (n_points, n_reps)
    mode: "std" or "sem"
    global_err: if True -> return scalar; else -> return vector length n_points
    """
    if values_2d.ndim != 2:
        raise ValueError("replicates must be 2D: (n_points, n_reps)")

    n_reps = values_2d.shape[1]
    if n_reps < 2:
        # With 1 replicate, std/sem isn't defined. Return zeros as the safest default.
        base = np.zeros(values_2d.shape[0], dtype=float)
    else:
        base = np.std(values_2d, axis=1, ddof=ddof)

    if mode == "std":
        err_vec = base
    elif mode == "sem":
        err_vec = base / np.sqrt(max(n_reps, 1))
    else:
        raise ValueError("mode must be 'std' or 'sem'")

    if global_err:
        # One uncertainty for all points (e.g., instrument precision estimated from all reps)
        return float(np.mean(err_vec))
    return err_vec


"""
A dataclass storing information about the data given so the program computes once and just accesses later. 
"""

from dataclasses import dataclass
from typing import Optional, List, Union
import numpy as np

@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray
    z: Optional[np.ndarray] = None

    xerr: Optional[np.ndarray] = None
    yerr: Optional[np.ndarray] = None
    zerr: Optional[np.ndarray] = None

    xlabel: str = "x"
    ylabel: str = "y"
    zlabel: str = "z"

    xunit: str = ""
    yunit: str = ""
    zunit: str = ""

    name: str = "dataset"

    def __post_init__(self):
        # y always 1D (n,)
        self.y = np.asarray(self.y, dtype=float).ravel()

        # z optional 1D (n,)
        if self.z is not None:
            self.z = np.asarray(self.z, dtype=float).ravel()

        # x can be 1D (n,) or 2D (n, d)
        self.x = np.asarray(self.x, dtype=float)
        if self.x.ndim == 1:
            self.x = self.x.reshape(-1, 1)   # (n, 1)
        elif self.x.ndim == 2:
            pass                              # (n, d)
        else:
            raise ValueError("x must be 1D (n,) or 2D (n, d)")

        n, d = self.x.shape
        if d > 3:
            raise ValueError("x supports at most 3 features (d <= 3)")

        if self.y.shape[0] != n:
            raise ValueError("y length must match number of rows in x")

        if self.z is not None and self.z.shape[0] != n:
            raise ValueError("z length must match number of rows in x")

        # xerr: allow None, scalar, (n,) for 1D x, or (n,d) for multivariable
        if self.xerr is not None:
            self.xerr = np.asarray(self.xerr, dtype=float)
            if self.xerr.ndim == 0:
                self.xerr = np.full((n, d), float(self.xerr), dtype=float)
            elif self.xerr.ndim == 1:
                if d != 1:
                    raise ValueError("For multivariable x, xerr must be scalar or shape (n, d)")
                if self.xerr.shape != (n,):
                    raise ValueError("xerr must have shape (n,) for 1D x")
                self.xerr = self.xerr.reshape(n, 1)
            elif self.xerr.ndim == 2:
                if self.xerr.shape != (n, d):
                    raise ValueError("xerr must have shape (n, d)")
            else:
                raise ValueError("xerr must be scalar, 1D, or 2D")

        # yerr: allow None, scalar, or (n,)
        if self.yerr is not None:
            self.yerr = np.asarray(self.yerr, dtype=float)
            if self.yerr.ndim == 0:
                self.yerr = np.full(n, float(self.yerr), dtype=float)
            if self.yerr.shape != (n,):
                raise ValueError("yerr must be scalar or shape (n,)")

        # zerr: allow None, scalar, or (n,) — only meaningful if z exists
        if self.zerr is not None:
            if self.z is None:
                raise ValueError("zerr was provided but z is None")
            self.zerr = np.asarray(self.zerr, dtype=float)
            if self.zerr.ndim == 0:
                self.zerr = np.full(n, float(self.zerr), dtype=float)
            if self.zerr.shape != (n,):
                raise ValueError("zerr must be scalar or shape (n,)")

        # cached means
        self.x_mean = self.x.mean(axis=0)        # (d,)
        self.y_mean = float(self.y.mean())
        self.z_mean = float(self.z.mean()) if self.z is not None else None

    @property
    def n(self) -> int:
        # number of points (rows)
        return int(self.x.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.x.shape[1])

    @property
    def has_xerr(self) -> bool:
        return self.xerr is not None

    @property
    def has_yerr(self) -> bool:
        return self.yerr is not None

    @property
    def has_z(self) -> bool:
        return self.z is not None

    @property
    def has_zerr(self) -> bool:
        return self.zerr is not None

    @property
    def xerr0(self) -> np.ndarray:
        return self.xerr if self.xerr is not None else np.zeros_like(self.x)

    @property
    def yerr0(self) -> np.ndarray:
        return self.yerr if self.yerr is not None else np.zeros_like(self.y)

    @property
    def zerr0(self) -> np.ndarray:
        if self.z is None:
            return None
        return self.zerr if self.zerr is not None else np.zeros_like(self.z)

    @property
    def x_std(self):
        # per-feature std (d,)
        return self.x.std(axis=0, ddof=1) if self.n > 1 else np.zeros(self.n_features)

    @property
    def y_std(self) -> float:
        return float(np.std(self.y, ddof=1)) if self.n > 1 else 0.0

    @property
    def z_std(self) -> Optional[float]:
        if self.z is None:
            return None
        return float(np.std(self.z, ddof=1)) if self.n > 1 else 0.0

    def summary(self) -> str:
        extra = ", z" if self.z is not None else ""
        return f"{self.name}: {self.n} points, {self.n_features}D x, y{extra}"

    @classmethod
    def from_replicates(
        cls,
        x,
        Y,
        *,
        X=None,                 # optional x replicates, shape (n_points, n_reps)
        Z=None,                 # optional z replicates, shape (n_points, n_reps)

        yerr_mode="sem",        # "std" or "sem"
        xerr_mode="sem",        # "std" or "sem"
        zerr_mode="sem",        # "std" or "sem"

        yerr_global=False,      # True -> scalar error
        xerr_global=False,      # True -> scalar error
        zerr_global=False,      # True -> scalar error

        ddof=1,

        yerr=None,              # allow overriding with custom errors
        xerr=None,              # allow overriding with custom errors
        zerr=None,              # allow overriding with custom errors

        **kwargs
    ):
        # --- Y (required) ---
        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2:
            raise ValueError("Y must be 2D: (n_points, n_reps)")

        n_points = Y.shape[0]

        # --- X handling (optional replicates) ---
        if X is not None:
            X = np.asarray(X, dtype=float)
            if X.shape != Y.shape:
                raise ValueError("X must have same shape as Y: (n_points, n_reps)")
            x_mean = np.mean(X, axis=1)

            if xerr is None:
                xerr_calc = _err_from_reps(X, mode=xerr_mode, ddof=ddof, global_err=xerr_global)
                xerr = np.full(n_points, xerr_calc, dtype=float) if np.isscalar(xerr_calc) else xerr_calc
        else:
            x = np.asarray(x, dtype=float).ravel()
            if x.shape[0] != n_points:
                raise ValueError("x length must match Y.shape[0] (n_points)")
            x_mean = x
            # xerr stays as provided (often None)

        # --- y mean + yerr ---
        y_mean = np.mean(Y, axis=1)
        if yerr is None:
            yerr_calc = _err_from_reps(Y, mode=yerr_mode, ddof=ddof, global_err=yerr_global)
            yerr = np.full(n_points, yerr_calc, dtype=float) if np.isscalar(yerr_calc) else yerr_calc

        # --- Z handling (optional) ---
        z_mean = None
        if Z is not None:
            Z = np.asarray(Z, dtype=float)
            if Z.ndim != 2:
                raise ValueError("Z must be 2D: (n_points, n_reps)")
            if Z.shape[0] != n_points:
                raise ValueError("Z must have same n_points as Y")

            z_mean = np.mean(Z, axis=1)

            if zerr is None:
                zerr_calc = _err_from_reps(Z, mode=zerr_mode, ddof=ddof, global_err=zerr_global)
                zerr = np.full(n_points, zerr_calc, dtype=float) if np.isscalar(zerr_calc) else zerr_calc

        return cls(x=x_mean, y=y_mean, z=z_mean, xerr=xerr, yerr=yerr, zerr=zerr, **kwargs)

"""
Class for multiple curves
"""

@dataclass
class DatasetCollection:
    datasets: List[Dataset]
    name: str = "collection"

    def __post_init__(self):
        if not self.datasets:
            raise ValueError("DatasetCollection cannot be empty")
        if not all(isinstance(d, Dataset) for d in self.datasets):
            raise TypeError("All items in datasets must be Dataset instances")

    @property
    def n_curves(self) -> int:
        return len(self.datasets)

    def summary(self) -> str:
        return f"{self.name}: {self.n_curves} curves"

    def labels(self) -> List[str]:
        return [d.name for d in self.datasets]

    def require_same_x(self, *, atol: float = 0.0, rtol: float = 0.0) -> bool:
        """Return True if all curves share the same x array (within tolerance)."""
        x0 = self.datasets[0].x
        for d in self.datasets[1:]:
            if not np.allclose(d.x, x0, atol=atol, rtol=rtol):
                return False
        return True

    def stack_y(self, *, require_same_x: bool = True) -> np.ndarray:
        """
        Stack y arrays into shape (n_points, n_curves).
        Only valid if all datasets have same x and same length.
        """
        if require_same_x and not self.require_same_x():
            raise ValueError("Cannot stack: datasets do not share the same x values")
        n = self.datasets[0].n
        if any(d.n != n for d in self.datasets):
            raise ValueError("Cannot stack: datasets have different lengths")
        return np.column_stack([d.y for d in self.datasets])

    @classmethod
    def from_y_matrix(
        cls,
        x: np.ndarray,
        Y: np.ndarray,
        *,
        names: Optional[List[str]] = None,
        xlabel: str = "x",
        ylabel: str = "y",
        xunit: str = "",
        yunit: str = "",
        collection_name: str = "collection",
        yerr: Optional[np.ndarray] = None,
        xerr: Optional[np.ndarray] = None,
    ) -> "DatasetCollection":
        """
        Build multiple curves from a shared x and a Y matrix:
          x shape: (n,)
          Y shape: (n, m) where m = number of curves
        """
        x = np.asarray(x, dtype=float).ravel()
        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2:
            raise ValueError("Y must be 2D with shape (n_points, n_curves)")
        if Y.shape[0] != x.shape[0]:
            raise ValueError("Y first dimension must match x length")

        m = Y.shape[1]
        if names is None:
            names = [f"curve_{j+1}" for j in range(m)]
        if len(names) != m:
            raise ValueError("names must have length equal to number of curves")

        datasets = []
        for j in range(m):
            datasets.append(
                Dataset(
                    x=x,
                    y=Y[:, j],
                    xerr=xerr,
                    yerr=yerr,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    xunit=xunit,
                    yunit=yunit,
                    name=names[j],
                )
            )

        return cls(datasets=datasets, name=collection_name)