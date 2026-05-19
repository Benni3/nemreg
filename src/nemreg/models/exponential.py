import numpy as np
from nemreg.core.model import Model


def exponential():
    """
    Model:
        y = A * exp(k*x) + B

    Parameters:
        A, k, B
    """

    def f(x, A, k, B):
        x = np.asarray(x, dtype=float)

        # Accept (n,) or (1,n)
        if x.ndim == 2:
            if x.shape[0] != 1:
                raise ValueError("exponential_offset expects 1D x")
            x = x[0]

        return A * np.exp(k * x) + B

    def guess(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.ndim == 2:
            x = x.reshape(-1)

        # Rough offset estimate
        B0 = float(np.min(y))

        y_adj = y - B0

        mask = (y_adj > 0) & np.isfinite(x) & np.isfinite(y_adj)

        if np.count_nonzero(mask) >= 2:
            lx = x[mask]
            ly = np.log(y_adj[mask])

            # ln(y-B) = kx + ln(A)
            A_mat = np.vstack([lx, np.ones_like(lx)]).T
            k_est, lnA_est = np.linalg.lstsq(A_mat, ly, rcond=None)[0]
            A_est = float(np.exp(lnA_est))

            return [A_est, float(k_est), B0]

        # fallback guess
        return [1.0, -1.0, float(np.mean(y))]

    return Model(
        name="exponential_offset",
        func=f,
        param_names=("A", "k", "B"),
        guess=guess,
        expr="A * exp(k*x) + B",
        latex=r"A \exp{k x} + B"
    )