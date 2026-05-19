import numpy as np
from nemreg.core.model import Model

"""
Later adding the nonlinear term

y = C_0 + C_1 * x^n_1 + ... + C_k * x^n_k

Making the models span a bigger area
"""

def power():
    """
    Model:
        y = C * x^n + B

    Parameters:
        C, n, B
    """

    def f(x, C, n, B):
        x = np.asarray(x, dtype=float)

        # Accept (n,) or (1,n) from multivariable convention
        if x.ndim == 2:
            if x.shape[0] != 1:
                raise ValueError("power_offset expects 1D x")
            x = x[0]

        return C * (x ** n) + B

    def guess(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.ndim == 2:
            x = x.reshape(-1)

        # Basic safe initial guesses
        B0 = float(np.min(y))   # offset approx lowest value
        y_adj = y - B0

        mask = (x > 0) & (y_adj > 0) & np.isfinite(x) & np.isfinite(y_adj)

        if np.count_nonzero(mask) >= 2:
            lx = np.log(x[mask])
            ly = np.log(y_adj[mask])

            A = np.vstack([lx, np.ones_like(lx)]).T
            n_est, lnC_est = np.linalg.lstsq(A, ly, rcond=None)[0]
            C_est = float(np.exp(lnC_est))

            return [C_est, float(n_est), B0]

        # fallback if log-fit impossible
        return [1.0, 1.0, float(np.mean(y))]

    return Model(
        name="power_offset",
        func=f,
        param_names=("C", "n", "B"),
        guess=guess,
        expr="C * x^n + B",
        latex=r"C x^n + B"
    )