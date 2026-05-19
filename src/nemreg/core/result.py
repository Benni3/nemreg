from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Any, Tuple, Callable
import numpy as np

from nemreg.core.model import Model
from nemreg.stats.metrics import r2, rmse, chi2
from nemreg.core.expr import format_expr


@dataclass(frozen=True)
class FitResult:
    """
    Output of nemreg fitting (LM core).

    Stores BOTH:
      - the fitted parameters
      - the model (so the result can evaluate itself)

    This makes plotting + LaTeX later much cleaner.

    Attributes
    ----------
    model_name: str
    model: Model
        The fitted model definition (contains func + param_names).
    param_names: tuple[str, ...]
    popt: np.ndarray
        Best-fit parameters in the same order as param_names.
    pcov: np.ndarray
        Covariance matrix.
    perr: np.ndarray
        1-sigma standard errors (sqrt(diag(pcov))).
    yhat: np.ndarray
        Model predictions at the dataset x-values used in fit.
    residuals: np.ndarray
        y - yhat (unweighted)
    stats: dict
        r2, rmse, chi2, chi2_red, dof, n, p
    meta: dict
        Extra info (weighted/unweighted, bounds, lambda, etc.)
    """
    model_name: str
    model: Model
    param_names: Tuple[str, ...]
    popt: np.ndarray
    pcov: np.ndarray
    perr: np.ndarray
    yhat: np.ndarray
    residuals: np.ndarray
    stats: Dict[str, float]
    meta: Dict[str, Any]

    def params_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in zip(self.param_names, self.popt)}

    def predict(self, x, *, params: Optional[Sequence[float]] = None) -> np.ndarray:
        """
        Predict y for arbitrary x using the stored model.

        - For 1D models: x is typically (n,) (or (1,n) depending on caller)
        - For multivariable models in nemreg: x is typically (d,n)
        """
        p = self.popt if params is None else np.asarray(params, dtype=float).ravel()
        y = self.model.func(x, *p)
        return np.asarray(y, dtype=float).ravel()

    def predict_from_dataset_x(self, dataset: Any) -> np.ndarray:
        """
        Convenience: predict using a Dataset's x, respecting nemreg x conventions.

        Dataset.x is stored as (n,d). We pass to model.func:
          - d==1 -> (n,)
          - d>1  -> (d,n)
        """
        X = np.asarray(dataset.x, dtype=float)
        if X.ndim != 2:
            raise ValueError("predict_from_dataset_x(): dataset.x must be 2D (n,d)")
        n, d = X.shape
        if n <= 0:
            return np.asarray([], dtype=float)

        xdata = X[:, 0] if d == 1 else X.T
        return self.predict(xdata)

    def summary(self, digits: int = 6) -> str:
        lines = []
        lines.append(f"Model: {self.model_name}")
        lines.append(
            f"n={int(self.stats.get('n', -1))}, "
            f"p={int(self.stats.get('p', -1))}, "
            f"dof={int(self.stats.get('dof', -1))}"
        )

        for name, val, err in zip(self.param_names, self.popt, self.perr):
            lines.append(f"  {name} = {val:.{digits}g} ± {err:.{digits}g}")

        r2v = self.stats.get("r2", float("nan"))
        rmsev = self.stats.get("rmse", float("nan"))
        chi2v = self.stats.get("chi2", float("nan"))
        chi2rv = self.stats.get("chi2_red", float("nan"))
        lines.append(
            f"R²={r2v:.{digits}g}, RMSE={rmsev:.{digits}g}, "
            f"χ²={chi2v:.{digits}g}, χ²_red={chi2rv:.{digits}g}"
        )
        return "\n".join(lines)

    @property
    def func(self) -> Callable[..., np.ndarray]:
        """Quick access to the stored model function."""
        return self.model.func
    
    def expr_str(self, digits: int = 4) -> str:
        s = format_expr(self.model.expr, self.param_names, self.popt, digits=digits)
        return s if s is not None else "<no expression template>"

    def latex_str(self, digits: int = 4) -> str:
        tmpl = self.model.latex if self.model.latex is not None else self.model.expr
        s = format_expr(tmpl, self.param_names, self.popt, digits=digits)
        if s is None:
            return "<no latex template>"
        # minimal cleanup if using expr as latex
        s = s.replace("*", r"\,")
        s = s.replace("**", "^")
        return s

    