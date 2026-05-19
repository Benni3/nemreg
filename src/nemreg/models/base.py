import numpy as np
from nemreg.core.model import Model


def base(base: float):
    """
    y = A * base^(k*x) + B

    User chooses base.
    """

    if base <= 0:
        raise ValueError("base must be positive")

    ln_base = np.log(base)

    def f(x, A, k, B):
        x = np.asarray(x, dtype=float)

        if x.ndim == 2:
            if x.shape[0] != 1:
                raise ValueError("exponential_base expects 1D x")
            x = x[0]

        return A * np.exp(k * ln_base * x) + B

    def guess(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.ndim == 2:
            x = x.reshape(-1)

        B0 = float(np.min(y))
        y_adj = y - B0

        mask = y_adj > 0

        if np.count_nonzero(mask) >= 2:
            lx = x[mask]
            ly = np.log(y_adj[mask])

            A_mat = np.vstack([lx, np.ones_like(lx)]).T
            slope, intercept = np.linalg.lstsq(A_mat, ly, rcond=None)[0]

            k_est = slope / ln_base
            A_est = np.exp(intercept)

            return [A_est, k_est, B0]

        return [1.0, 1.0, float(np.mean(y))]

    return Model(
        name=f"exponential_base_{base}",
        func=f,
        param_names=("A", "k", "B"),
        guess=guess,
        expr="A * base^(k*x) + B",
        latex=r"A base^{k x} + B"
    )