# tests/test_fit_multivariable.py
import numpy as np
import nemreg as nr

def test_fit_multivariable_plane_like_model():
    rng = np.random.default_rng(1)

    n = 60
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    X = np.column_stack([x1, x2])  # (n,2)

    a_true, b_true, c_true = 1.5, -2.0, 0.3
    sigma = 0.2
    y = a_true*x1 + b_true*x2 + c_true + rng.normal(0, sigma, size=n)

    ds = nr.Dataset(x=X, y=y, yerr=sigma, name="mv")

    def plane(x, a, b, c):
        x = np.asarray(x, float)
        # multivariable convention: (d,n)
        if x.ndim != 2 or x.shape[0] != 2:
            raise ValueError("Expected x as (2,n)")
        x1, x2 = x[0], x[1]
        return a*x1 + b*x2 + c

    model = nr.Model(name="plane", func=plane, param_names=("a", "b", "c"))

    res = nr.fit(ds, model)

    assert res.stats["r2"] > 0.9
    yhat = res.predict_from_dataset_x(ds)
    assert yhat.shape == (n,)