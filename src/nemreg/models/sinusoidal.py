import numpy as np
from nemreg.core.model import Model


def sinusoidal():
    """
    Sinusoidal model:
        y = A * sin(omega*x + phi) + B

    Params:
        A, omega, phi, B
    """

    def f(x, A, omega, phi, B):
        x = np.asarray(x, dtype=float)

        if x.ndim == 2:
            if x.shape[0] != 1:
                raise ValueError("trigenometric expects 1D x")
            x = x[0]

        return A * np.sin(omega * x + phi) + B

    def guess(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.ndim == 2:
            x = x.reshape(-1)

        # Sort for frequency estimation
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]

        B0 = float(np.mean(y))
        y0 = y - B0
        A0 = float(0.5 * (np.max(y0) - np.min(y0)))
        if not np.isfinite(A0) or A0 == 0:
            A0 = 1.0

        # FFT frequency guess
        omega0 = 1.0
        if x.size >= 4:
            dx = np.median(np.diff(x))
            if np.isfinite(dx) and dx > 0:
                yf = np.fft.rfft(y0)
                freqs = np.fft.rfftfreq(x.size, d=dx)
                if freqs.size > 1:
                    k = np.argmax(np.abs(yf[1:])) + 1
                    f0 = freqs[k]
                    omega0 = float(2 * np.pi * f0) if f0 > 0 else 1.0

        phi0 = 0.0
        return [A0, omega0, phi0, B0]

    return Model(
        name="trigenometric",
        func=f,
        param_names=("A", "omega", "phi", "B"),
        guess=guess,
        expr="A * sin(omega*x + phi) + B",
        latex=r"A \sin{omega x + phi} + B"

    )