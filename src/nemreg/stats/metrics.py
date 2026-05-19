from __future__ import annotations

import numpy as np


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, float).ravel()
    yhat = np.asarray(yhat, float).ravel()
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, float).ravel()
    yhat = np.asarray(yhat, float).ravel()
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def chi2(y: np.ndarray, yhat: np.ndarray, sigma: np.ndarray) -> float:
    y = np.asarray(y, float).ravel()
    yhat = np.asarray(yhat, float).ravel()
    sigma = np.asarray(sigma, float).ravel()
    return float(np.sum(((y - yhat) / sigma) ** 2))


__all__ = ["r2", "rmse", "chi2"]