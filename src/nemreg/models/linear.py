import numpy as np
from nemreg.core.model import Model


def linear():
    """
    Linear model:
        y = A*x + B
    """

    def f(x, A, B):
        return A * x + B

    def guess(x, y):
        # Simple slope/intercept estimate using endpoints
        if len(x) < 2:
            return [1.0, 0.0]

        A = (y[-1] - y[0]) / (x[-1] - x[0]) if (x[-1] - x[0]) != 0 else 1.0
        B = y[0] - A * x[0]
        return [A, B]

    return Model(
        name="linear",
        func=f,
        param_names=("A", "B"),
        guess=guess,
        expr="A*x + B",
        latex=r"A x + B",
    )