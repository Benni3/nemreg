import numpy as np
from nemreg.core.model import Model

"""
Later add restriction to the model as it is not valid when the abs is over 1 
"""

def arccos():
    """
    y = A * arccos(B*x + C) + D
    """

    def f(x, A, B, C, D):
        x = np.asarray(x, dtype=float)
        if x.ndim == 2:
            if x.shape[0] != 1:
                raise ValueError("arccos expects 1D x")
            x = x[0]
        return A * np.arccos(B * x + C) + D

    return Model(
        name="arccos",
        func=f,
        param_names=("A", "B", "C", "D"),
        expr="A * arccos(B*x + C) + D",
        latex=r"A \arccos(B x + C) + D"
    )