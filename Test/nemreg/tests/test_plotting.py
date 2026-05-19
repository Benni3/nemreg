# tests/test_plotting.py
import numpy as np
import nemreg as nr

def test_plotting_dispatcher_does_not_crash():
    x = np.linspace(0, 5, 20)
    y = 2*x + 1
    ds = nr.Dataset(x=x, y=y, yerr=0.1, name="plot")

    def linear(x, A, B):
        x = np.asarray(x, float)
        if x.ndim == 2:
            x = x[0]
        return A*x + B

    model = nr.Model(name="linear", func=linear, param_names=("A","B"))
    res = nr.fit(ds, model)

    # data only
    fig, ax = nr.plot(mode="data", dataset=ds)
    assert fig is not None and ax is not None

    # data + fit
    fig, ax = nr.plot(mode="result_data", dataset=ds, result=res)
    assert fig is not None and ax is not None

    # result only (fit)
    fig, ax = nr.plot(mode="result", dataset=ds, result=res, plot_mode="fit")
    assert fig is not None and ax is not None

    # alias: plot_mode should also work (your dispatcher maps it)
    fig, ax = nr.plot(mode="result", dataset=ds, result=res, plot_mode="fit")
    assert fig is not None and ax is not None

    # residuals
    fig, ax = nr.plot(mode="result", dataset=ds, result=res, plot_mode="residuals")
    assert fig is not None and ax is not None