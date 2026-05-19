import numpy as np
from nemreg.core.model import Model


def surface():
    """
    2D polynomial surface:
        z = a*x + b*y + c + d*x^2 + e*y^2 + f*x*y

    Expects x passed as shape (2, n): [x_row, y_row]
    Params:
        a, b, c, d, e, f
    """

    def f(X, a, b, c, d, e, fxy):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] != 2:
            raise ValueError("surface expects X shape (2, n)")

        x = X[0]
        y = X[1]
        return a * x + b * y + c + d * x**2 + e * y**2 + fxy * x * y

    def guess(X, z):
        # Simple safe guess: intercept = mean(z)
        z = np.asarray(z, dtype=float).ravel()
        return [0.0, 0.0, float(np.mean(z)), 0.0, 0.0, 0.0]

    return Model(
        name="surface",
        func=f,
        param_names=("a", "b", "c", "d", "e", "f"),
        guess=guess,
        expr="a*x + b*y + c + d*x^2 + e*y^2 + f*x*y",
        latex=r"a x + b y + c + d x^2 + e y^2 + f x y"
    )