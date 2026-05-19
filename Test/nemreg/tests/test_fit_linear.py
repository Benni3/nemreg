# tests/test_fit_linear.py
import numpy as np
import nemreg as nr

def test_fit_linear_returns_model_and_good_r2():
    rng = np.random.default_rng(0)

    n = 50
    x = np.linspace(-5, 5, n)
    A_true, B_true = 2.0, -1.0
    sigma = 0.5

    y = A_true*x + B_true + rng.normal(0, sigma, size=n)
    ds = nr.Dataset(x=x, y=y, yerr=sigma, name="lin")

    def linear_func(x, A, B):
        x = np.asarray(x, float)
        if x.ndim == 2:  # (d,n) convention possible
            x = x[0]
        return A*x + B

    model = nr.Model(name="linear", func=linear_func, param_names=("A", "B"))

    res = nr.fit(ds, model)

    # must contain model
    assert res.model is model
    assert res.model_name == "linear"

    # stats must exist
    assert "r2" in res.stats
    assert res.stats["r2"] > 0.9

    # predictions must have correct shape
    yhat = res.predict_from_dataset_x(ds)
    assert yhat.shape == (n,)