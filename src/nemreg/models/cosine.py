import numpy as np
from nemreg.core.model import Model


def cosine():
    """
    y = A * cos(omega*x + phi) + B
    """

    def f(x, A, omega, phi, B):
        x = np.asarray(x, dtype=float)
        if x.ndim == 2:
            if x.shape[0] != 1:
                raise ValueError("cosine expects 1D x")
            x = x[0]
        return A * np.cos(omega * x + phi) + B

    return Model(
        name="cosine",
        func=f,
        param_names=("A", "omega", "phi", "B"),
        expr="A cos(omega x + phi) + B",
        latex=r"A \cos(omega + phi) + B"
    )