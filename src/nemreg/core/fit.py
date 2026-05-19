from __future__ import annotations

from typing import Optional, Sequence, Tuple, Any, Dict
import numpy as np
from scipy.optimize import curve_fit

from nemreg.core.result import FitResult
from nemreg.stats.metrics import r2, rmse, chi2


def _prepare_xdata(dataset) -> np.ndarray:
    """
    Dataset convention:
      dataset.x is stored as shape (n, d) (after your Dataset __post_init__)
      dataset.y is shape (n,)

    Model convention:
      - 1D models accept x as (n,) or (1,n)
      - multivariable models expect x as (d, n)

    So we pass:
      - if d==1: xdata = dataset.x[:,0]  -> (n,)
      - else:    xdata = dataset.x.T     -> (d,n)
    """
    X = np.asarray(dataset.x, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"fit(): dataset.x must be 2D (n,d). Got shape {X.shape}")

    n, d = X.shape
    if d == 1:
        return X[:, 0]
    return X.T


def _prepare_sigma(dataset) -> Tuple[Optional[np.ndarray], bool]:
    """
    Returns (sigma, weighted)
    sigma is 1D length n, or None.
    """
    if getattr(dataset, "yerr", None) is None:
        return None, False

    sigma = np.asarray(dataset.yerr, dtype=float).ravel()
    n = np.asarray(dataset.y, dtype=float).ravel().size

    if sigma.size != n:
        raise ValueError(f"fit(): yerr must have length n={n}, got {sigma.size}")

    if np.any(~np.isfinite(sigma)):
        raise ValueError("fit(): yerr contains non-finite values")
    if np.any(sigma <= 0):
        raise ValueError("fit(): yerr must be strictly positive for weighting")

    return sigma, True


def _choose_p0(model, xdata, y, p0: Optional[Sequence[float]]):
    """
    Precedence:
      1) user p0
      2) model.guess(x,y)
      3) ones
    Also validates length vs model.param_names.
    """
    n_params = len(model.param_names)

    if p0 is not None:
        p0_arr = np.asarray(p0, dtype=float).ravel()
        if p0_arr.size != n_params:
            raise ValueError(f"fit(): p0 length must be {n_params}, got {p0_arr.size}")
        return p0_arr

    if getattr(model, "guess", None) is not None:
        try:
            g = model.guess(xdata, y)
            g = np.asarray(g, dtype=float).ravel()
            if g.size != n_params:
                raise ValueError(
                    f"fit(): model.guess returned length {g.size}, expected {n_params}"
                )
            return g
        except Exception as e:
            raise ValueError(f"fit(): model.guess failed: {type(e).__name__}: {e}") from e

    return np.ones(n_params, dtype=float)


def fit(
    dataset,
    model,
    *,
    p0: Optional[Sequence[float]] = None,
    bounds: Tuple[Sequence[float], Sequence[float]] = (-np.inf, np.inf),
    maxfev: int = 100000,
    method: Optional[str] = None,
    absolute_sigma: Optional[bool] = None,
    **curve_fit_kwargs,
) -> FitResult:
    """
    Fit a Model to a Dataset using scipy.optimize.curve_fit.

    Parameters
    ----------
    dataset : Dataset
    model : Model
    p0 : initial guess (overrides model.guess)
    bounds : (lower, upper) bounds for parameters
    maxfev : curve_fit max function evals
    method : curve_fit method (None, 'trf', 'dogbox', 'lm')
    absolute_sigma : if None -> True when weighted (yerr), else False
    curve_fit_kwargs : passed through to curve_fit

    Returns
    -------
    FitResult
    """
    y = np.asarray(dataset.y, dtype=float).ravel()
    if y.ndim != 1:
        raise ValueError("fit(): dataset.y must be 1D")

    xdata = _prepare_xdata(dataset)
    n = y.size

    # sanity check x and y compatible
    if xdata.ndim == 1:
        if xdata.size != n:
            raise ValueError(f"fit(): x and y size mismatch: {xdata.size} vs {n}")
    else:
        # xdata is (d,n)
        if xdata.shape[1] != n:
            raise ValueError(f"fit(): x and y mismatch: x has n={xdata.shape[1]}, y has n={n}")

    sigma, weighted = _prepare_sigma(dataset)

    if absolute_sigma is None:
        absolute_sigma = bool(weighted)

    p0_use = _choose_p0(model, xdata, y, p0)

    # curve_fit call
    try:
        popt, pcov = curve_fit(
            model.func,
            xdata,
            y,
            p0=p0_use,
            sigma=sigma,
            absolute_sigma=absolute_sigma,
            bounds=bounds,
            maxfev=maxfev,
            method=method,
            **curve_fit_kwargs,
        )
    except Exception as e:
        raise RuntimeError(
            "fit(): curve_fit failed.\n"
            f"  model={getattr(model, 'name', 'unknown')}\n"
            f"  error={type(e).__name__}: {e}"
        ) from e

    popt = np.asarray(popt, dtype=float).ravel()
    pcov = np.asarray(pcov, dtype=float)

    # errors
    if pcov.shape[0] == pcov.shape[1] and pcov.shape[0] == popt.size and np.all(np.isfinite(pcov)):
        perr = np.sqrt(np.clip(np.diag(pcov), 0.0, np.inf))
    else:
        perr = np.full_like(popt, np.nan, dtype=float)

    # predictions + residuals on training data
    yhat = np.asarray(model.func(xdata, *popt), dtype=float).ravel()
    residuals = y - yhat

    p = popt.size
    dof = int(max(n - p, 0))

    stats: Dict[str, float] = {
        "n": float(n),
        "p": float(p),
        "dof": float(dof),
        "r2": r2(y, yhat),
        "rmse": rmse(y, yhat),
    }

    if sigma is not None:
        chi2_val = chi2(y, yhat, sigma)
        stats["chi2"] = chi2_val
        stats["chi2_red"] = chi2_val / dof if dof > 0 else float("nan")
    else:
        stats["chi2"] = float("nan")
        stats["chi2_red"] = float("nan")

    meta: Dict[str, Any] = {
        "weighted": weighted,
        "absolute_sigma": absolute_sigma,
        "bounds": bounds,
        "method": method,
        "maxfev": maxfev,
    }

    
    return FitResult(
        model_name=getattr(model, "name", "model"),
        model=model,
        param_names=tuple(getattr(model, "param_names", ())),
        popt=popt,
        pcov=pcov,
        perr=perr,
        yhat=yhat,
        residuals=residuals,
        stats=stats,
        meta=meta,
    )