import numpy as np
from nemreg.core.model import Model


def logarithmic():
    """
    y = A * ln(x) + B

    Params:
        A, B
    """

    def f(x, A, B):
        x = np.asarray(x, dtype=float)

        if x.ndim == 2:
            if x.shape[0] != 1:
                raise ValueError("logarithmic expects 1D x")
            x = x[0]

        return A * np.log(x) + B

    def guess(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.ndim == 2:
            x = x.reshape(-1)

        mask = x > 0
        if np.count_nonzero(mask) >= 2:
            lx = np.log(x[mask])
            A_mat = np.vstack([lx, np.ones_like(lx)]).T
            A_est, B_est = np.linalg.lstsq(A_mat, y[mask], rcond=None)[0]
            return [float(A_est), float(B_est)]

        return [1.0, float(np.mean(y))]

    return Model(
        name="logarithmic",
        func=f,
        param_names=("A", "B"),
        guess=guess,
        expr="A * ln(x) + B",
        latex=r"A \ln(x) + B"
    )