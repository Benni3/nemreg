import numpy as np
from nemreg.core.model import Model


def arctan():
    """
    y = A * arctan(B*x + C) + D
    """

    def f(x, A, B, C, D):
        x = np.asarray(x, dtype=float)
        if x.ndim == 2:
            if x.shape[0] != 1:
                raise ValueError("arctan expects 1D x")
            x = x[0]
        return A * np.arctan(B * x + C) + D

    return Model(
        name="arctan",
        func=f,
        param_names=("A", "B", "C", "D"),
        expr="A * arctan(B*x + C) + D",
        latex=r"A \arctan(B x + C) + D"
    )