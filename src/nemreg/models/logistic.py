import numpy as np
from nemreg.core.model import Model


def logistic():
    """
    Logistic (sigmoid):
        y = L / (1 + exp(-k*(x-x0))) + B

    Params:
        L, k, x0, B
    """

    def f(x, L, k, x0, B):
        x = np.asarray(x, dtype=float)

        if x.ndim == 2:
            if x.shape[0] != 1:
                raise ValueError("logistic expects 1D x")
            x = x[0]

        return L / (1 + np.exp(-k * (x - x0))) + B

    def guess(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.ndim == 2:
            x = x.reshape(-1)

        L0 = float(np.max(y) - np.min(y))
        B0 = float(np.min(y))
        x00 = float(np.median(x))
        k0 = 1.0

        return [L0, k0, x00, B0]

    return Model(
        name="logistic",
        func=f,
        param_names=("L", "k", "x0", "B"),
        guess=guess,
        expr="L / (1 + exp(-k*(x-x0))) + B",
        latex=r"\frac{L}{(1 + \exp{-k (x-x0)})} + B"
    )