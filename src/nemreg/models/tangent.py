import numpy as np
from nemreg.core.model import Model

"""
Later add restriction to the model as it is not valid near the asymptotes
"""

def tangent():
    """
    y = A * tan(omega*x + phi) + B
    """

    def f(x, A, omega, phi, B):
        x = np.asarray(x, dtype=float)
        if x.ndim == 2:
            if x.shape[0] != 1:
                raise ValueError("tangent expects 1D x")
            x = x[0]
        return A * np.tan(omega * x + phi) + B

    return Model(
        name="tangent",
        func=f,
        param_names=("A", "omega", "phi", "B"),
        expr="A * tan(omega*x + phi) + B",
        latex="A \tan{omega x + phi} + B"

    )