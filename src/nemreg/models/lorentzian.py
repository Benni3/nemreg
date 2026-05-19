import numpy as np
from nemreg.core.model import Model


def lorentzian():
    """
    Lorentzian peak:
        y = A / (1 + ((x-x0)/gamma)^2) + B

    Params:
        A, x0, gamma, B
    """

    def f(x, A, x0, gamma, B):
        x = np.asarray(x, dtype=float)

        if x.ndim == 2:
            if x.shape[0] != 1:
                raise ValueError("lorentzian expects 1D x")
            x = x[0]

        return A / (1 + ((x - x0) / gamma) ** 2) + B

    def guess(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.ndim == 2:
            x = x.reshape(-1)

        B0 = float(np.min(y))
        A0 = float(np.max(y) - B0)
        x00 = float(x[np.argmax(y)])
        gamma0 = float(0.2 * (x.max() - x.min()))

        if gamma0 <= 0:
            gamma0 = 1.0

        return [A0, x00, gamma0, B0]

    return Model(
        name="lorentzian",
        func=f,
        param_names=("A", "x0", "gamma", "B"),
        guess=guess,
        expr="A / (1 + ((x-x0)/gamma)^2) + B",
        latex=r"\frac{A}{(1 + (\frac{(x-x0)}{gamma})^2 )} + B"
    )