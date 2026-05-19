import numpy as np
from nemreg.core.model import Model


def rational():
    """
    Rational (1/1) model:
        y = (a*x + b) / (c*x + d)

    Params:
        a, b, c, d
    """

    def f(x, a, b, c, d):
        x = np.asarray(x, dtype=float)

        if x.ndim == 2:
            if x.shape[0] != 1:
                raise ValueError("rational expects 1D x")
            x = x[0]

        return (a * x + b) / (c * x + d)

    def guess(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.ndim == 2:
            x = x.reshape(-1)

        # Safe generic guess
        # Start near y ≈ (a/d) x + (b/d), so set d=1
        b0 = float(np.mean(y))
        a0 = 0.0
        c0 = 0.0
        d0 = 1.0
        return [a0, b0, c0, d0]

    return Model(
        name="rational",
        func=f,
        param_names=("a", "b", "c", "d"),
        guess=guess,
        expr="(a*x + b) / (c*x + d)",
        latex=r"\frac{a x + b}{c x + d}"
    )