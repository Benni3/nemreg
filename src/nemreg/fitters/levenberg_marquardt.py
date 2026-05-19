from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict, Any

import numpy as np

from nemreg.core.result import FitResult, _r2, _rmse, _chi2


def _prepare_xdata(dataset) -> np.ndarray:
    """
    Dataset.x stored as (n, d).
    Model convention:
      - d==1 -> pass (n,) to model.func
      - d>1  -> pass (d, n) to model.func
    """
    X = np.asarray(dataset.x, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"lm_fit(): dataset.x must be 2D (n,d). Got {X.shape}")
    n, d = X.shape
    if d == 1:
        return X[:, 0]
    return X.T


def _prepare_sigma(dataset) -> Tuple[Optional[np.ndarray], bool]:
    """
    Returns (sigma, weighted).
    sigma is 1D length n or None.
    """
    y = np.asarray(dataset.y, dtype=float).ravel()
    n = y.size

    if getattr(dataset, "yerr", None) is None:
        return None, False

    sigma = np.asarray(dataset.yerr, dtype=float).ravel()
    if sigma.size != n:
        raise ValueError(f"lm_fit(): yerr length must be n={n}, got {sigma.size}")

    if np.any(~np.isfinite(sigma)):
        raise ValueError("lm_fit(): yerr contains non-finite values")
    if np.any(sigma <= 0):
        raise ValueError("lm_fit(): yerr must be strictly positive for weighting")

    return sigma, True


def _choose_p0(model, xdata, y, p0: Optional[Sequence[float]]) -> np.ndarray:
    """
    Precedence:
      1) user p0
      2) model.guess(x,y)
      3) ones
    """
    n_params = len(model.param_names)

    if p0 is not None:
        p0 = np.asarray(p0, dtype=float).ravel()
        if p0.size != n_params:
            raise ValueError(f"lm_fit(): p0 length must be {n_params}, got {p0.size}")
        return p0

    if getattr(model, "guess", None) is not None:
        g = model.guess(xdata, y)
        g = np.asarray(g, dtype=float).ravel()
        if g.size != n_params:
            raise ValueError(f"lm_fit(): model.guess returned {g.size} values, expected {n_params}")
        return g

    return np.ones(n_params, dtype=float)


def _weighted_residuals(model, xdata: np.ndarray, y: np.ndarray, theta: np.ndarray, sigma: Optional[np.ndarray]) -> np.ndarray:
    """
    r(theta) = (y - f(x,theta)) / sigma   if sigma given
              (y - f(x,theta))           else
    """
    yhat = np.asarray(model.func(xdata, *theta), dtype=float).ravel()
    if yhat.shape != y.shape:
        raise ValueError(f"lm_fit(): model returned shape {yhat.shape}, expected {y.shape}")

    r = (y - yhat)
    if sigma is not None:
        r = r / sigma
    if not np.all(np.isfinite(r)):
        raise ValueError("lm_fit(): residuals contain non-finite values (check model domain / data).")
    return r


def _num_jacobian_residuals(
    model,
    xdata: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    sigma: Optional[np.ndarray],
    *,
    rel_step: float = 1e-6,
    abs_step: float = 1e-8,
) -> np.ndarray:
    """
    Numerical Jacobian of residuals:
      J[:,j] = (r(theta + h e_j) - r(theta)) / h

    Uses parameter-scaled step: h = max(abs_step, rel_step*max(1,|theta_j|))
    """
    r0 = _weighted_residuals(model, xdata, y, theta, sigma)
    n = r0.size
    p = theta.size
    J = np.empty((n, p), dtype=float)

    for j in range(p):
        h = max(abs_step, rel_step * max(1.0, abs(float(theta[j]))))
        t1 = theta.copy()
        t1[j] += h
        r1 = _weighted_residuals(model, xdata, y, t1, sigma)
        J[:, j] = (r1 - r0) / h

    if not np.all(np.isfinite(J)):
        raise ValueError("lm_fit(): Jacobian contains non-finite values.")
    return J


def _solve_damped(A: np.ndarray, g: np.ndarray, lam: float) -> np.ndarray:
    """
    Solve (A + lam*D) d = g where D = diag(A) (scale-aware damping).
    Falls back to lstsq if needed.
    """
    # Damping matrix: use diag(A); if diag has zeros, replace with 1
    diagA = np.diag(A).copy()
    diagA[~np.isfinite(diagA)] = 1.0
    diagA = np.where(diagA > 0, diagA, 1.0)
    D = np.diag(diagA)

    M = A + lam * D

    try:
        return np.linalg.solve(M, g)
    except np.linalg.LinAlgError:
        # robust fallback
        d, *_ = np.linalg.lstsq(M, g, rcond=None)
        return d


@dataclass
class LMOptions:
    max_iter: int = 200
    max_retries: int = 10

    lambda0: float = 1e-2
    lambda_up: float = 10.0
    lambda_down: float = 0.1
    lambda_min: float = 1e-20
    lambda_max: float = 1e20

    # finite-diff Jacobian steps
    rel_step: float = 1e-6
    abs_step: float = 1e-8

    # stopping criteria
    tol_step: float = 1e-10
    tol_cost: float = 1e-12
    tol_grad: float = 1e-10

    # if True, treat sigma as absolute (like curve_fit absolute_sigma=True)
    # affects covariance scaling
    absolute_sigma: Optional[bool] = None


def lm_fit(
    dataset,
    model,
    *,
    p0: Optional[Sequence[float]] = None,
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    options: Optional[LMOptions] = None,
) -> FitResult:
    """
    Levenberg–Marquardt nonlinear least squares fitter (weighted).

    Parameters
    ----------
    dataset : Dataset
    model   : Model (func(x, *params) -> y)
    p0      : optional initial parameters
    bounds  : optional (lower, upper) bounds for parameters (simple clamp)
             Note: this is not full bounded LM; we clamp candidate theta to bounds.
    options : LMOptions

    Returns
    -------
    FitResult
    """
    if options is None:
        options = LMOptions()

    y = np.asarray(dataset.y, dtype=float).ravel()
    n = y.size
    xdata = _prepare_xdata(dataset)
    sigma, weighted = _prepare_sigma(dataset)

    # covariance scaling choice
    absolute_sigma = options.absolute_sigma
    if absolute_sigma is None:
        absolute_sigma = bool(weighted)

    theta = _choose_p0(model, xdata, y, p0)
    p = theta.size

    # bounds handling (simple clamp)
    if bounds is not None:
        lo = np.asarray(bounds[0], dtype=float).ravel()
        hi = np.asarray(bounds[1], dtype=float).ravel()
        if lo.size == 1:
            lo = np.full(p, float(lo[0]))
        if hi.size == 1:
            hi = np.full(p, float(hi[0]))
        if lo.size != p or hi.size != p:
            raise ValueError(f"lm_fit(): bounds must be length {p} (or scalars).")
        if np.any(lo > hi):
            raise ValueError("lm_fit(): bounds invalid (lower > upper).")
        # clamp initial
        theta = np.minimum(np.maximum(theta, lo), hi)
    else:
        lo = hi = None

    # initial residuals/cost
    r = _weighted_residuals(model, xdata, y, theta, sigma)
    S = float(r @ r)

    lam = float(options.lambda0)
    lam = min(max(lam, options.lambda_min), options.lambda_max)

    # store for stopping checks
    last_S = S

    # main LM loop
    for it in range(options.max_iter):
        # compute Jacobian and normal eq pieces at current theta
        J = _num_jacobian_residuals(
            model, xdata, y, theta, sigma,
            rel_step=options.rel_step,
            abs_step=options.abs_step,
        )
        A = J.T @ J           # (p,p)
        g = J.T @ r           # (p,)

        grad_norm = float(np.linalg.norm(g, ord=np.inf))
        if grad_norm < options.tol_grad:
            break

        accepted = False
        theta_old = theta.copy()
        S_old = S

        # inner retry loop: increase lambda until step is accepted
        for retry in range(options.max_retries):
            # solve damped system
            dtheta = _solve_damped(A, g, lam)

            step_norm = float(np.linalg.norm(dtheta))
            if step_norm < options.tol_step:
                accepted = True  # tiny step => consider converged
                break

            theta_cand = theta + dtheta

            # clamp candidate to bounds (simple bounded behavior)
            if lo is not None:
                theta_cand = np.minimum(np.maximum(theta_cand, lo), hi)

            # evaluate candidate
            r_cand = _weighted_residuals(model, xdata, y, theta_cand, sigma)
            S_cand = float(r_cand @ r_cand)

            # ---- ACCEPT / REJECT happens HERE ----
            if S_cand < S:
                # ACCEPT
                theta = theta_cand
                r = r_cand
                S = S_cand

                lam = max(options.lambda_min, lam * options.lambda_down)
                accepted = True
                break
            else:
                # REJECT
                theta = theta_old  # explicit (keeps intent clear)
                S = S_old
                r = _weighted_residuals(model, xdata, y, theta, sigma)

                lam = min(options.lambda_max, lam * options.lambda_up)

        # if we never accepted any step, stop (stuck)
        if not accepted:
            break

        # stopping by cost improvement
        if abs(last_S - S) < options.tol_cost:
            break
        last_S = S

    # final predictions on training x
    yhat = np.asarray(model.func(xdata, *theta), dtype=float).ravel()
    residuals_unweighted = y - yhat

    # final Jacobian for covariance estimate
    J_final = _num_jacobian_residuals(
        model, xdata, y, theta, sigma,
        rel_step=options.rel_step,
        abs_step=options.abs_step,
    )
    A_final = J_final.T @ J_final

    # Covariance: (J^T J)^-1 * s^2
    # If sigma is absolute -> s^2 = 1
    # else -> s^2 = chi2/dof
    dof = max(n - p, 0)

    try:
        Ainv = np.linalg.inv(A_final)
    except np.linalg.LinAlgError:
        Ainv = np.linalg.pinv(A_final)

    if sigma is not None:
        chi2 = _chi2(y, yhat, sigma)
        s2 = 1.0 if absolute_sigma else (chi2 / dof if dof > 0 else np.nan)
        pcov = Ainv * s2
    else:
        chi2 = float("nan")
        pcov = Ainv * (np.sum(residuals_unweighted**2) / dof if dof > 0 else np.nan)

    perr = np.sqrt(np.clip(np.diag(pcov), 0.0, np.inf)) if pcov.shape == (p, p) else np.full(p, np.nan)

    stats: Dict[str, float] = {
        "n": float(n),
        "p": float(p),
        "dof": float(dof),
        "r2": _r2(y, yhat),
        "rmse": _rmse(y, yhat),
        "chi2": float(chi2),
        "chi2_red": float(chi2 / dof) if (sigma is not None and dof > 0) else float("nan"),
    }

    meta: Dict[str, Any] = {
        "algorithm": "levenberg_marquardt",
        "weighted": weighted,
        "absolute_sigma": bool(absolute_sigma),
        "lambda0": float(options.lambda0),
        "lambda_final": float(lam),
        "max_iter": int(options.max_iter),
        "max_retries": int(options.max_retries),
        "rel_step": float(options.rel_step),
        "abs_step": float(options.abs_step),
        "bounds": bounds,
    }

    return FitResult(
        model_name=getattr(model, "name", "model"),
        param_names=tuple(model.param_names),
        popt=np.asarray(theta, dtype=float),
        pcov=np.asarray(pcov, dtype=float),
        perr=np.asarray(perr, dtype=float),
        yhat=np.asarray(yhat, dtype=float),
        residuals=np.asarray(residuals_unweighted, dtype=float),
        stats=stats,
        meta=meta,
    )