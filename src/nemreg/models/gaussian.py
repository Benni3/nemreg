import numpy as np
from nemreg.core.model import Model


def gaussian():
    """
    Gaussian with offset:
        y = A * exp(-(x-mu)^2 / (2*sigma^2)) + B

    Params:
        A, mu, sigma, B
    """

    def f(x, A, mu, sigma, B):
        x = np.asarray(x, dtype=float)

        # Accept (n,) or (1,n)
        if x.ndim == 2:
            if x.shape[0] != 1:
                raise ValueError("gaussian expects 1D x")
            x = x[0]

        sigma = np.asarray(sigma, dtype=float)
        return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + B

    def guess(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.ndim == 2:
            x = x.reshape(-1)

        # Sort for stable heuristics
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]

        B0 = float(np.min(y))
        y0 = y - B0

        # Amplitude guess
        A0 = float(np.max(y0))
        if not np.isfinite(A0) or A0 == 0:
            A0 = 1.0

        # Center guess: x at max y
        mu0 = float(x[np.argmax(y0)]) if x.size else 0.0

        # Sigma guess: weighted std around mu using y0 as weights
        w = np.clip(y0, 0, np.inf)
        if np.sum(w) > 0:
            sigma0 = float(np.sqrt(np.sum(w * (x - mu0) ** 2) / np.sum(w)))
        else:
            sigma0 = float(0.2 * (x.max() - x.min())) if x.size > 1 else 1.0

        if not np.isfinite(sigma0) or sigma0 <= 0:
            sigma0 = 1.0

        return [A0, mu0, sigma0, B0]

    return Model(
        name="gaussian",
        func=f,
        param_names=("A", "mu", "sigma", "B"),
        guess=guess,
        expr="A * exp(-(x-mu)^2 / (2*sigma^2)) + B",
        latex=r"A \exp{- \frac{(x - mu)^2}{2 sigma^2}} + B"
    )