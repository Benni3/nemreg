import numpy as np
from nemreg.core.model import Model


def damped_oscillation():
    """
    Damped oscillation:
        y = A * exp(-gamma*t) * cos(omega*t + phi) + B

    Params:
        A, gamma, omega, phi, B
    """

    def f(t, A, gamma, omega, phi, B):
        t = np.asarray(t, dtype=float)

        # Accept (n,) or (1,n)
        if t.ndim == 2:
            if t.shape[0] != 1:
                raise ValueError("damped_oscillation expects 1D t")
            t = t[0]

        return A * np.exp(-gamma * t) * np.cos(omega * t + phi) + B

    def guess(t, y):
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)

        if t.ndim == 2:
            t = t.reshape(-1)

        # Sort by t for nicer heuristics
        idx = np.argsort(t)
        t = t[idx]
        y = y[idx]

        B0 = float(np.mean(y))
        y0 = y - B0

        # Amplitude guess: half peak-to-peak
        A0 = float(0.5 * (np.max(y0) - np.min(y0)))
        if not np.isfinite(A0) or A0 == 0:
            A0 = 1.0

        # Frequency guess: FFT on detrended signal
        omega0 = 1.0
        if t.size >= 4:
            dt = np.median(np.diff(t))
            if np.isfinite(dt) and dt > 0:
                yf = np.fft.rfft(y0)
                freqs = np.fft.rfftfreq(t.size, d=dt)  # Hz
                # ignore DC
                if freqs.size > 1:
                    k = np.argmax(np.abs(yf[1:])) + 1
                    f0 = freqs[k]
                    omega0 = float(2 * np.pi * f0) if f0 > 0 else 1.0

        # Damping guess: small positive
        gamma0 = 0.0  # start undamped; optimizer can increase

        # Phase guess: align first point (rough)
        phi0 = 0.0

        return [A0, gamma0, omega0, phi0, B0]

    return Model(
        name="damped_oscillation",
        func=f,
        param_names=("A", "gamma", "omega", "phi", "B"),
        guess=guess,
        expr="A * exp(-gamma*t) * cos(omega*t + phi) + B",
        latex=r"A \exp^{-gamma t} \cos(omega t + phi) + B"
    )